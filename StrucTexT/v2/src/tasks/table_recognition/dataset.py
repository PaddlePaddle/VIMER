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
            with open(data_path, 'r', 'utf-8') as f:
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
        example = examples[0]
        bboxes = np.array(example['cell_bboxes']).reshape(-1, 4).tolist()
        row_col_indexs = np.array(example['row_col_indexes']).reshape(-1, 2)
        xs = np.array(example['xs']).reshape(-1)
        ys = np.array(example['ys']).reshape(-1)
        row_link_map_up = np.array(example['row_link_map_up'])
        row_link_map_down = np.array(example['row_link_map_down'])
        col_link_map_left = np.array(example['col_link_map_left'])
        col_link_map_right = np.array(example['col_link_map_right'])

        xs_cls = np.array(example['xs_cls']).reshape(-1)
        ys_cls = np.array(example['ys_cls']).reshape(-1)
        feature = {'image': example['image'],
                   'bboxes': bboxes,
                   'row_col_indexs': row_col_indexs.tolist(),
                   'xs': xs,
                   'ys': ys,
                   'xs_cls': xs_cls,
                   'ys_cls': ys_cls,
                   'row_link_map_up': row_link_map_up,
                   'row_link_map_down': row_link_map_down,
                   'col_link_map_left': col_link_map_left,
                   'col_link_map_right':col_link_map_right}
        if self.transform:
            feature = self.transform(example)
        return feature

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
