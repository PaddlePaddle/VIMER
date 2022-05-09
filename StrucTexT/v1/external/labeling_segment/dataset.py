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
import glob
import gzip
import json
import codecs
import random
import logging
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from itertools import accumulate
from functools import reduce, partial
from collections import namedtuple
from paddle.fluid import core
from model.ernie.tokenizing_ernie import ErnieTokenizer
from utils.imaug.operators import DecodeImage, NormalizeImage

Feature = namedtuple("Feature", [
    'images', 'sentence', 'sentence_bboxes', 'sentence_ids', 'sentence_pos',
    'sentence_mask', 'label', 'label_mask'])

def image_preprocess(im, input_size):
    """ image_preprocess """
    if isinstance(input_size, list) or isinstance(input_size, tuple):
        input_size = input_size[0]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    new_h = int(im_shape[0] * im_scale)
    new_w = int(im_shape[1] * im_scale)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    im_padded = np.zeros((input_size, input_size, 3), dtype=np.float32)
    im_padded[:new_h, :new_w, :] = im
    im_padded = im_padded.transpose((2, 0, 1))
    return im_padded, im_scale


def read_example_from_file(label_file):
    """ read_example_from_file """
    if not os.path.exists(label_file):
        logging.warning('The file (%s) is not existed!', label_file)
        return None
    example = None
    try:
        with gzip.open(label_file, 'rb') as f:
            example = json.loads(f.read().decode('utf-8'))
    except:
        return None
    return example


def convert_examples_to_features(config, examples, tokenizer, transform=None):
    """
    convert_example_to_feature
    :param config: configure
    :param examples: batch_samples for document
    :param tokenizer : convert subword to vocab_id
    :param transform: process function for image
    """
    def _flatten_2d_list(c):
        return [i for item in c for i in item]
    vocab = tokenizer.vocab
    input_size = config.get('image_shape')
    max_seqlen = config.get('max_seqlen')
    sep_token = vocab[config['dict_params'].get('sep_token')]
    pad_token = vocab[config['dict_params'].get('pad_token')]
    cls_token = vocab[config['dict_params'].get('cls_token')]
    img_token = vocab[config['dict_params'].get('img_token')]
    mask_token = vocab[config['dict_params'].get('mask_token')]

    pad_bbox = [0, 0, 0, 0]
    batch_size = len(examples)

    images = [e['image'] for e in examples]
    label = np.zeros((batch_size, max_seqlen), dtype='int64')
    label_mask = np.zeros((batch_size, max_seqlen), dtype='bool')

    sentence = np.ones((batch_size, max_seqlen), dtype='int32') * pad_token #文本和图像特征
    sentence_pos = np.zeros((batch_size, max_seqlen), dtype='int32') # 单词绝对位置（subwords共享）
    sentence_ids = np.zeros((batch_size, max_seqlen), dtype='int32') # 句子绝对位置（文本和图像tokens共享）
    sentence_mask = np.ones((batch_size, max_seqlen), dtype='int32') * 2 # 0 for word_token, 1 for line_token, 2 for pad_token
    sentence_bboxes = np.zeros((batch_size, max_seqlen, 4), dtype='float32')

    ## sample continuous lines block
    line_nums = []
    for idx in range(batch_size):
        image = images[idx]
        line_bboxes = examples[idx]['line_bboxes']
        seq_tokens = examples[idx]['tokens']
        cls = examples[idx]['cls']
        token_bboxes = examples[idx].get('token_bboxes', [[c] * len(l) for c, l in zip(line_bboxes, seq_tokens)])
        acc_len = list(accumulate([len(_flatten_2d_list(c)) for c in seq_tokens]))
        buf_a = [l for c, l in enumerate(acc_len, 3) if c + l < max_seqlen] # [cls] + ... + [doc] + num(line)
        line_len = len(buf_a)
        line_bboxes = line_bboxes[:line_len]
        seq_tokens = seq_tokens[:line_len]
        tokens_bboxe = token_bboxes[:line_len]
        # assign label
        label[idx, :line_len] = cls[:line_len]
        label_mask[idx, :line_len] = True
        # contain [doc]
        line_bboxes = [[0, 0, image.shape[1] - 1, image.shape[0] - 1]] + line_bboxes
        line_len = len(buf_a) + 1

        if transform:
            image = transform(image)
        image, im_scale = image_preprocess(image, input_size)
        images[idx] = image
        # expand token
        extend_tokens = [[['[CLS]']]] + seq_tokens # [cls], seq1, sep2, seq3, [doc], ...
        # expand bbox
        extend_token_bboxes = [[pad_bbox]] + token_bboxes # [pad_bbox], token_bboxes, token_bboxes, ...
        extend_token_bboxes = _flatten_2d_list(extend_token_bboxes) # concat all sequences
        # generate sent_pos
        extend_sent_pos = [[i] * len(_flatten_2d_list(c)) for i, c in enumerate(extend_tokens)]
        extend_sent_pos = _flatten_2d_list(extend_sent_pos) # concat all sequences

        extend_tokens = _flatten_2d_list(extend_tokens) # concat all sequences
        extend_token_bboxes = [[bbox] * len(id) for bbox, id in zip(extend_token_bboxes, extend_tokens)] # expand to subword, which share the word bbox
        extend_token_bboxes = _flatten_2d_list(extend_token_bboxes) # concat all subword_bboexes

        # generate token_pos
        extend_token_pos = [[i] * len(c) for i, c in enumerate(extend_tokens)] # expand to subword_pos
        extend_token_pos = _flatten_2d_list(extend_token_pos) # concat all subword_pos
        extend_tokens = _flatten_2d_list(extend_tokens) # expand to subwords
        try:
            extend_tokens = tokenizer.convert_tokens_to_ids(extend_tokens)
        except:
            return None

        line_nums.append(line_len)
        token_len = len(extend_tokens)

        assert(len(line_bboxes) == line_len)
        assert(len(extend_token_bboxes) == token_len)
        assert(len(extend_token_pos) == token_len)
        assert(len(extend_sent_pos) == token_len)
        assert(max(extend_sent_pos) + 1 == line_len)

        sentence[idx, :token_len] = extend_tokens
        sentence_ids[idx, :token_len] = extend_sent_pos
        sentence_pos[idx, :token_len] = extend_token_pos
        sentence_mask[idx, :token_len] = 0
        sentence_bboxes[idx, :token_len] = extend_token_bboxes

        sentence[idx, token_len:token_len + line_len] = img_token
        sentence_ids[idx, token_len:token_len + line_len] = np.arange(line_len, dtype='int64')
        sentence_mask[idx, token_len:token_len + line_len] = 1
        sentence_bboxes[idx, token_len:token_len + line_len] = line_bboxes
        sentence_bboxes[idx] *= im_scale


    max_vislen = max(line_nums) - 1
    label = label[:, :max_vislen]
    label_mask = label_mask[:, :max_vislen]

    feature = Feature(
        images=np.asarray(images),
        sentence=sentence,
        sentence_ids=sentence_ids,
        sentence_pos=sentence_pos,
        sentence_mask=sentence_mask,
        sentence_bboxes=sentence_bboxes,
        label=label,
        label_mask=label_mask)

    return feature


class Dataset(paddle.io.Dataset):
    """LabelingSegmentDataset"""
    def __init__(self, config, feed_names, train_mode):
        """__init__"""
        super(Dataset, self).__init__()
        self.config = config
        self.feed_names = feed_names
        self.train_mode = train_mode
        assert(config.get('dict_path', None) is not None)
        dict_params = config['dict_params']
        self.tokenizer = ErnieTokenizer.init(config['dict_path'], **dict_params)

        data_path = config.get('data_path')
        image_path = self.config.get('image_path', '')
        image_list = []
        label_list = []
        data_list = []

        if os.path.isdir(image_path):
            image_list.append(image_path)
        elif image_path != '':
            logging.error('error in load image_path: ', image_path)

        if os.path.isfile(data_path):
            with codecs.open(data_path, 'r', 'utf-8') as f:
                label_list = list(filter(lambda x: len(x) == 2, [x.strip().split('\t') for x in f.readlines()]))
                image_list = [x[0] for x in label_list]
                label_list = [x[1] for x in label_list]
        elif os.path.isdir(data_path):
            label_list.append(data_path)
        else:
            logging.error('error in load data_path: ', data_path)

        if len(image_list) > 1 and len(image_list) != len(label_list):
            logging.error('the image_path is not assigned to data_path')
            return
        elif len(image_list) == 1:
            image_list = image_list * len(label_list)
        elif len(image_list) == 0:
            image_list = [[]] * len(label_list)

        dataset_index = -1
        for data_path, image_path in zip(label_list, image_list):
            dataset_index += 1
            if not os.path.isdir(data_path):
                raise ValueError('The %s in data_path should be a folder containing .label files' % data_path)
            sub_data = glob.glob(os.path.join(data_path, '*.label'))
            logging.debug('load {:,} label files from {}'.format(len(sub_data), data_path))
            sub_data = [[item, image_path, str(dataset_index)] for item in sub_data]
            data_list.extend(sub_data)

        self.data_list = data_list
        self.batch_size = config.get('batch_size', 1)
        self.len_step = len(data_list) // self.batch_size
        logging.info('{}: Load {:,} mini_batch data with batch_size {}.'.format(
                'train' if train_mode else 'test', self.len_step, self.batch_size))
        if config.get('shuffle', False):
            random.shuffle(self.data_list)

        transform_config = config.get('transform')
        decode_image = DecodeImage(transform_config['img_mode'],
                                   transform_config['channel_first'])
        normalize_image = NormalizeImage(transform_config['scale'],
                                         transform_config['mean'],
                                         transform_config['std'],
                                         transform_config['order'])
        self.transform = lambda x: normalize_image(decode_image({'image': x}))['image']

    def __len__(self):
        return self.len_step

    def __getitem__(self, idx):
        features = None
        while features is None:
            examples = []
            while len(examples) < self.batch_size:
                i_start, i_end = idx * self.batch_size, (idx + 1) * self.batch_size
                batch_samples = self.data_list[i_start:i_end]
                for sample in batch_samples:
                    label, image_dir, data_source = sample
                    example = read_example_from_file(label)
                    if example is None or len(example['line_bboxes']) < 1:
                        continue
                    if 'image' in example.keys() and example['image'] is None:
                        logging.debug('Not find image in %s', image_path)
                        continue
                    elif 'image_name' in example.keys():
                        image_name = example['image_name']
                        image_path = os.path.join(image_dir, image_name)
                        if not os.path.exists(image_path.encode('utf-8')):
                            logging.debug('Not find %s', image_path)
                            continue
                        try:
                            image = cv2.imread(image_path)
                        except:
                            logging.debug('Error in read %s', image_path)
                            continue
                        if image is None:
                            logging.debug('Error in load image for %s', label)
                            continue
                        example['image'] = image
                    else:
                        logging.debug('Not image find in %s', image_path)
                        continue
                    example['data_source'] = int(data_source)
                    examples.append(example)
                    if len(examples) >= self.batch_size:
                        break
                idx = np.random.choice(self.len_step, 1)[0]

            features = convert_examples_to_features(
                                        self.config,
                                        examples,
                                        self.tokenizer,
                                        self.transform)
            if features is not None:
                break
            idx = np.random.choice(self.len_step, 1)[0]
        samples_data = [getattr(features, name) for name in self.feed_names]
        return samples_data
