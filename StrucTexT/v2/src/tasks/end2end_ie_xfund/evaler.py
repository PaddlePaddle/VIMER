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
from src.postprocess.ocr_postprocess import split_anno_by_symbol, grouping

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

    @P.no_grad()
    def run(self):
        '''
        print evaluation results
        '''
        pass

    def _resume_model(self):
        '''
        Resume from saved model
        :return:
        '''
        pass
