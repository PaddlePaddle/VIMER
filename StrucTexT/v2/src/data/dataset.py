"""Data Reader functions."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# # vim:fenc=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import random
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from itertools import accumulate
from paddle.io import WeightedRandomSampler
from abc import ABCMeta, abstractmethod

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from functools import reduce, partial
from collections import namedtuple
from paddle.fluid import core

class BaseDataset(paddle.io.Dataset):
    """BaseDataset"""
    __metaclass__ = ABCMeta
    def __init__(self,
            dataset_list=None,
            feed_names=None,
            batch_size=1,
            train_mode=True,
            shuffle=False,
            weightedsampler=None,
            collect_batch=True,
            **kwargs):
        """__init__"""
        super(BaseDataset, self).__init__()

        self.feed_names = feed_names
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.collect_batch = collect_batch

        assert(len(dataset_list) > 0)
        assert(isinstance(dataset_list, list)), 'dataset_list must be a 2d-list'
        assert(isinstance(feed_names, list)), 'feed_names must be a 1d-list'
        assert(weightedsampler is None or isinstance(weightedsampler, list) \
                and len(weightedsampler) == len(dataset_list)), \
                'weightedsampler must be a list with same len of dataset_list'

        self.data_list = []
        self.data_index = [[] for x in range(len(dataset_list))]
        for source, data_source in enumerate(dataset_list):
            for index, example in enumerate(data_source):
                self.data_list.append([example, source])
                self.data_index[source].append(index)

        dataset_size = len(self.data_list)
        self.len_step = dataset_size // self.batch_size
        logging.info('Dataset... {} Load {:,} mini_batch data with batch_size {}.'.format(
            'train' if train_mode else 'test', self.len_step, self.batch_size))

        self.weightedsampler = None
        self.list_index = list(range(dataset_size))
        if weightedsampler:
            self.acc_len = [0] + list(accumulate([len(x) for x in self.data_index]))
            self.weightedsampler = WeightedRandomSampler(
                    weights=weightedsampler,
                    num_samples=1e8)
        elif shuffle:
            random.shuffle(self.list_index)
        if collect_batch:
            self.len_step = len(self.data_list)
            self.batch_size = 1


    @abstractmethod
    def _read_data(self, example):
        """
        Implemention of reading data.
        Args:
            data_path:
            image_path:
        Returns:
            A dict of structural data.
        """
        pass


    @abstractmethod
    def _convert_examples(self, examples):
        """
        Implemention of reading data.
        Args:
            examples: a dict of structural data.
        Returns:
            A dict mapping keys to the feed_names of model.
        """
        pass

    def __len__(self):
        return self.len_step

    def __getitem__(self, idx):
        examples = []
        while len(examples) < self.batch_size:
            i_start, i_end = idx * self.batch_size, (idx + 1) * self.batch_size
            batch_samples = self.list_index[i_start:i_end]
            for index in batch_samples:
                if self.weightedsampler:
                    bucket_index = next(self.weightedsampler)
                    index = index % (self.acc_len[bucket_index + 1] - self.acc_len[bucket_index])
                    index += self.acc_len[bucket_index]
                example, source = self.data_list[index]
                example = self._read_data(example)
                if example is None:
                    logging.error('Dataset... Cannot read example %s', example)
                    continue
                example['source'] = source
                examples.append(example)
                if len(examples) >= self.batch_size:
                    break
            idx = np.random.randint(self.__len__())

        features = self._convert_examples(examples)
        if features is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        input_data = []
        for name in self.feed_names:
            feature = features[name]
            if self.collect_batch and feature.shape[0] == 1:
                feature = feature[0]
            input_data.append(feature)
        return input_data
