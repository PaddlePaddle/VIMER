""" trainer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import math
import logging
import argparse
import functools
import numpy as np
import paddle as P
import cv2
from visualdl import LogWriter

from src.utils.save_load import load_model, save_model


class Trainer(object):
    """
    Trainer class
    """
    def __init__(self, config, model, optimizer, data_loader,
                 valid_data_loader=None, eval_classes=None,
                 lr_scheduler=None, distributed=False, 
                 eval_only=False, weights=None):
        '''
        :param config:
        :param model:
        :param optimizer:
        :param data_loader:
        :param valid_data_loader:
        :param lr_scheduler:
        '''
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.eval_classes = eval_classes
        self.lr_scheduler = lr_scheduler
        self.distributed = distributed
        self.len_step = len(self.data_loader)
        self.len_step_eval = len(self.valid_data_loader)
        self.forward_step = 1
        self.eval_only = eval_only

        global_config = config['global']
        self.epochs = global_config.get('epoch_num', 0)
        self.valid_step_interval = global_config.get('valid_step_interval', -1)
        self.log_step_interval = global_config.get('log_step_interval', -1)
        self.save_model_dir = global_config.get('save_model_dir', './output')
        self.eval_with_gt_bbox = global_config.get('eval_with_gt_bbox', False)
     
        # VisualDL config
        self.log_dir = os.path.join(self.save_model_dir, 'log_dir')
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.log_writer = LogWriter(self.log_dir)

        """
        Full training logic, including train and validation.
        """
        self.valid_model = self.model

        if self.eval_only:
            # eval
            para_dict = P.load(weights)
            self.model.set_dict(para_dict)  
            val_dict = self._valid(epoch=1)
            logging.info('[Epoch Validation] {}'.format(val_dict))
            logging.info("End Valiadtion!")
        else:  
            # train
            load_model(self.config, self.model)
            if self.distributed:
                from paddle.distributed import fleet
                logging.info('init parallel model')
                self.optimizer = fleet.distributed_optimizer(self.optimizer)
                self.model = fleet.distributed_model(self.model)
                self.model = P.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            for epoch in range(1, self.epochs + 1):
                # ensure distribute worker sample different data,
                # set different random seed by passing epoch to sampler
                self._train_epoch(epoch)
                self._save_model('model_epoch_%03d' % (epoch))
            logging.info("End training!")
        
    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log dict that contains average loss and metric in this epoch.
        '''
        self.model.train()
        ## step iteration start ##
        average_loss = 0.0
        item_losses = {'det_loss':[0, 0], 'recg_loss':[0, 0], 'cls_loss':[0, 0]}
        # training
        for step_idx, input_data in enumerate(self.data_loader(), start=1):
            output = self.model(*input_data,
                is_train=True,
                feed_names=self.config['train']['feed_names'])
            # calculate loss
            loss = output['loss']
            average_loss += loss.numpy()[0]
            if self.forward_step > 1:
                loss /= self.forward_step
            for item in item_losses.keys():
                if item in output.keys():
                    item_loss = output[item].numpy()[0]
                    if item_loss > 1e-7:
                        item_losses[item][0] += item_loss
                        item_losses[item][1] += 1
            # backward
            loss.backward()
            if step_idx % self.forward_step == 0:
                self.optimizer.step()
                self.optimizer.clear_grad()
            self.lr_scheduler.step()
    
            # print log
            if self.log_step_interval > 0 and step_idx % self.log_step_interval == 0:
                format = 'Epoch:[{}/{}] Train Step:[{}/{}] Total Loss: {:.5f} '.format(
                    epoch, self.epochs,
                    step_idx, self.len_step,
                    average_loss / self.log_step_interval)
            
                format += 'current_lr: {}'.format(self.optimizer.get_lr())
                format += '\n --- '
                for k, v in item_losses.items():
                    format += k.upper() + ': {:.5f}'.format(v[0] / (1e-7 + v[1])) + ', '
                logging.info(format[:-2])
                average_loss = 0.0
                item_losses = {'det_loss':[0, 0], 'recg_loss':[0, 0], 'cls_loss':[0, 0]}
            
            if self.do_validation and self.valid_step_interval > 0 and step_idx % self.valid_step_interval == 0:
                self._save_model('model_epoch_%03d_step_%07d' % (epoch, step_idx))
                '''
                try:
                    val_dict = self._valid(epoch=epoch)
                except:
                    continue
                logging.info('[Epoch Validation] Epoch:[{}/{}] {}'.format(
                        epoch, self.epochs, val_dict))
                '''
        for k, v in item_losses.items():
            self.log_writer.add_scalar(tag='{}'.format(k), step=epoch, value=(v[0] / (1e-7 + v[1])))

        item_losses = {'det_loss':[0, 0], 'recg_loss':[0, 0], 'cls_loss':[0, 0]}

    @P.no_grad()
    def _valid(self, epoch):
        '''
         Validate after training an epoch or regular step, this is a time-consuming procedure if validation data is big.
        :param max_step: Integer, do max times for training.
        :return: A dict that contains information about validation and the loss
        '''
        self.valid_model.eval()
        for eval_class in self.eval_classes.values():
            eval_class.reset()

        total_time = 0.0
        total_frame = 0.0

        for step_idx, input_data in enumerate(self.valid_data_loader()):
            start = time.time()
            output = self.model(*input_data, 
                is_train=False,
                eval_with_gt_bbox = self.eval_with_gt_bbox,
                feed_names=self.config['eval']['feed_names'])
            total_time += time.time() - start
    
            for task in self.eval_classes:
                gts = output[task + '_gts']
                preds = output[task + '_preds']
                for pred, gt in zip(preds, gts):
                    self.eval_classes[task].update(pred, gt)

            total_frame += input_data[0].shape[0]

        metrics = 'fps: {:.2f}\n'.format(total_frame / total_time)

        for key, val in self.eval_classes.items():
            val_result = val.accumulate()
            if isinstance(val_result, dict):
                metrics += '{}:\n'.format(key)
                for sub_key, sub_val in val_result.items():
                    metrics += '| {}: {} |'.format(sub_key, sub_val) 
                    self.log_writer.add_scalar(tag='{}_{}'.format(key, sub_key), step=epoch, value=sub_val)
                metrics += '\n'
            else:
                metrics += '\n{}:\n{}\n'.format(key, val_result)

        return metrics

    def _save_model(self, name):
        """save model
        """
        if P.distributed.get_rank() != 0:
            return
        save_model(
                model=self.model,
                optimizer=self.optimizer,
                model_path=self.save_model_dir,
                prefix=name)
