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
from src.postprocess.table_postprocess import TablePostProcess

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
        self.len_step = len(self.valid_data_loader)

        self.init_model = config['init_model']
        self.config = config['eval']
        self.postprocess = TablePostProcess()

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        self._resume_model()
        self.model.eval()

        total_time = 0.0
        total_frame = 0.0
        t = trange(self.len_step)
        loader = self.valid_data_loader()
        for step_idx in t:
            t.set_description('deal with %i' % step_idx)
            input_data = next(loader)
            feed_names = self.config['feed_names']
            start = time.time()
            output = self.model(*input_data, feed_names=feed_names)
            total_time += time.time() - start
            ######### Eval ##########
            label = output['label'][0]
            pred = self.postprocess(output)
            for key, val in self.eval_classes.items():
                val.update(pred, label)
            #########################
            total_frame += input_data[0].shape[0]
        metrics = 'fps : {}'.format(total_frame / total_time)
        for key, val in self.eval_classes.items():
            metrics += '\n{}:\n'.format(key) + str(val.accumulate())
            val.reset()
        print('[Eval Validation] {}'.format(metrics))

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
