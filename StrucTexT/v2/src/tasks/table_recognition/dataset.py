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

        if os.path.isfile(data_path):
            with open(data_path) as f:
                data_list = [[x.strip(), image_path] for x in f.readlines()]
                label_list = [data_list]
        else:
            raise ValueError('error in load data_path: ', data_path)

        self.transform = build_transform(config['transform'])
        super(Dataset, self).__init__(
                dataset_list=label_list,
                feed_names=feed_names,
                batch_size=batch_size,
                train_mode=train_mode,
                collect_batch=False,
                shuffle=True)

    def _convert_examples(self, examples):
        """
        convert_example_to_feature
        :param examples: batch_samples for document
        """
        images = []
        labels = []
        for example in examples:
            label = example['html']['structure']['tokens']
            label = list(filter(lambda x: ('head' not in x) and ('body' not in x), label))
            example = {'image': example['image']}
            if self.transform:
                feature = self.transform(example)
            images.append(feature['image'])
            labels.append(label)
        return {'image': np.asarray(images), 'label': labels}

    def _read_data(self, example):
        data, image_path = example
        example = json.loads(data)
        filename = example['filename']
        image_path = os.path.join(image_path, filename)
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
