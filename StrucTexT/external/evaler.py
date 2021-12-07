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
from tqdm import trange

class Evaler:
    """
    Evaler class
    """

    def __init__(self, config, model, data_loader, eval_classes=None):
        '''
        :param config:
        :param model:
        :param data_loader:
        '''
        self.model = model
        self.eval_classes = eval_classes
        self.valid_data_loader = data_loader
        self.len_step = len(self.valid_data_loader)

        self.init_model = config['init_model']
        self.valid_config = config['eval']

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
            t.set_description('evaluate with example %i' % step_idx)
            input_data = next(loader)
            start = time.time()
            feed_names = self.valid_config['feed_names']
            output = self.model(*input_data, feed_names=feed_names)
            total_time += time.time() - start

            ####### Eval ##########
            for key, val in self.eval_classes.items():
                if 'entity' in key and 'label_prim' in output.keys():
                    label = output['label_prim'].numpy()
                    logit = output['logit_prim'].numpy()
                    mask = output.get('mask', None)
                else:
                    label = output['label'].numpy()
                    logit = output['logit'].numpy()
                    mask = output.get('mask', None)
                mask = None if mask is None else mask.numpy()
                val(logit, label, mask)
            #########################
            total_frame += input_data[0].shape[0]
        metrics = 'fps : {}'.format(total_frame / total_time)
        for key, val in self.eval_classes.items():
            metrics += '\n{}:\n'.format(key) + str(val.get_metric())
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
