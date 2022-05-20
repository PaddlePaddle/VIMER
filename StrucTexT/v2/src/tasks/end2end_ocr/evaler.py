""" evaler.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import logging
import numpy as np
import paddle as P
import cv2

from tqdm import trange

class Evaler:
    """
    Evaler class
    """

    def __init__(self, config, model, data_loader, eval_classes):
        '''
        :param config:
        :param model:
        :param data_loader:
        :param eval_classes:
        '''
        self.config = config
        self.model = model
        self.valid_data_loader = data_loader
        self.eval_classes = eval_classes
        self.len_step = len(self.data_loader)

        self.init_model = config['init_model']
        self.config = config['eval']

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        self._resume_model()
        self.model.eval()
        for eval_class in self.eval_classes.values():
            eval_class.reset()

        total_time = 0.0
        total_frame = 0.0
        t = trange(self.len_step)
        loader = self.valid_data_loader()
        for step_idx in t:
            t.set_description('deal with %i' % step_idx)
            input_data = next(loader)
            start = time.time()
            feed_names = self.config['feed_names']
            output = self.model(*input_data, feed_names=feed_names, is_train=False)
            total_time += time.time() - start

            ####### Eval ##########
            for key, val in self.eval_classes.items():
                gts = output[key + '_gts']
                preds = output[key + '_preds']
                for pred, gt in zip(preds, gts):
                    val.update(pred, gt)
            #########################
            total_frame += input_data[0].shape[0]
        metrics = 'fps : {}'.format(total_frame / total_time)
        for key, val in self.eval_classes.items():
            val_result = val.accumulate()
            if isinstance(val_result, dict):
                metrics += '{}:\n'.format(key)
                for sub_key, sub_val in val_result.items():
                    metrics += '| {}: {} |'.format(sub_key, sub_val) 
                metrics += '\n'
            else:
                metrics += '\n{}:\n{}\n'.format(key, val_result)
        print('[Eval Validation] {}'.format(val_dict))

    def _resume_model(self):
        '''
        Resume from saved model
        :return:
        '''
        para_path = self.init_model
        if os.path.exists(para_path):
            para_dict = P.load(para_path)
            self.model.set_dict(para_dict)
            logging.info('Load init model from %s', para_path)
