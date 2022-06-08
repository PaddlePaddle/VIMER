"""Data Reader functions."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# # vim:fenc=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import cv2
import json
import gzip
import logging
import numpy as np
import paddle
import paddle.fluid as fluid

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from glob import glob
from itertools import accumulate
from src.data import build_transform
from src.data.dataset import BaseDataset

class Dataset(BaseDataset):
    """LabelingSegmentDataset"""
    def __init__(self, config, feed_names, train_mode=True):
        """__init__"""
        self.config = config

        batch_size = config['batch_size']
        data_path = config['data_path']
        image_path = config['image_path']

        assert os.path.isdir(image_path)

        if os.path.isfile(data_path):
            labels = [e.strip().split() for e in open(data_path).readlines()]
            #labels = [[os.path.join(image_path, os.path.basename(image)), int(label)] for image, label in labels]
            labels = [[os.path.join(image_path, image), int(label)] for image, label in labels]
            label_list = list(filter(lambda e: os.path.exists(e[0]), labels))
            label_list = [label_list]
        else:
            raise ValueError('error in load data_path for rvl-cdip: ', data_path)

        self.transform = build_transform(config['transform'])
        super(Dataset, self).__init__(
                dataset_list=label_list,
                feed_names=feed_names,
                batch_size=batch_size,
                train_mode=train_mode,
                collect_batch=True,
                shuffle=True)

    def _convert_examples(self, examples):
        """
        convert_example_to_feature
        :param examples: batch_samples for document
        """
        feature = examples[0]
        image = feature['image']
        if self.transform:
            transform_out = self.transform({'image': image})
            feature['image'] = transform_out['image']

        return feature

    def _read_data(self, example):
        image_path, label = example
        example = {'label': label}
        try:
            image = cv2.imread(image_path)
        except:
            logging.debug('Dataset... Error in read %s', image_path)
            return None
        if image is None:
            logging.debug('Dataset... Error in load image for %s', image_path)
            return None
        example['image'] = image

        return example
