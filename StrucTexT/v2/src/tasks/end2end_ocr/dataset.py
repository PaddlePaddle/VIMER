""" Data Reader functions. """
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
import copy
import glob
import codecs
import logging
import numpy as np

import paddle
import paddle.fluid as fluid

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from src.data import build_transform
from src.data.dataset import BaseDataset


Lexicon_Table_95 = ['!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', \
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', \
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', \
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
    'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ']


TEXT_CLASSES = {
    "question": 0,
    "answer": 1,
    "header": 2,
    "other": 3
}


class LabelConverter(object):
    """convert between text and lexicon index"""
    def __init__(self, seq_len=50, lexicon=None, recg_loss='CE'):
        """initialize
        Input:
            seq_len: the max-size sequence length
            lexion: the lexion info of recognition task
        """
        if lexicon is None:
            lexicon = Lexicon_Table_95
        self.recg_loss = recg_loss
        tokens = ['[PAD]', '[STOP]']
        self.idx2char = list(tokens) + list(lexicon)
        self.seq_len = seq_len

        self.char2idx = {}
        for i, char in enumerate(self.idx2char):
            self.char2idx[char] = i

    def encode(self, text, ignore_tag):
        """ encode character into index
        Input:
            text: the transcript of ground truth <String>
            ignore_tag: the flag to ignore the text <Bool>
        Output:
            new_text_idx: the text index of the input text with token index <List>
        """
        if ignore_tag:
            text = ""
        text = text.upper()
        text = list(text) + ['[STOP]']
        text_idx = [self.char2idx[char] for char in text] # for 95 classes
        new_text_idx = [self.char2idx['[PAD]']] * self.seq_len

        text_len = len(text)
        if text_len > self.seq_len:
            new_text_idx = text_idx[:self.seq_len]
        else:
            new_text_idx[:text_len] = text_idx

        return new_text_idx

    def decode(self, text_idx):
        """ convert text-index into text-label.
        Input:
            text_idx: the text index of predicted text <List>
        Output:
            text: the predicted text <String>
        """
        if self.recg_loss == "CE":
            text = ''.join([self.idx2char[idx] for idx in text_idx])
        elif self.recg_loss == "CTC":
            new_text_idx = []
            for i, t in enumerate(text_idx):
                if t != 0 and (i == 0 or t != text_idx[i - 1]):
                    new_text_idx.append(t)
            text = ''.join([self.idx2char[idx] for idx in new_text_idx])

        if text.find('[STOP]') != -1:
            text = text[:text.find('[STOP]')]
        return text


def _sort_box_with_list(anno, left_right_first=False):
    """sort bbox"""
    def compare_key(x):
        """ compare_key """
        poly = x[0]
        poly = np.array(poly, dtype=np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(poly)
        center = rect[0]
        # from left to right
        if left_right_first:
            return center[0], center[1]
        else:
        # from top to bottom
            return center[1], center[0]
    anno = sorted(anno, key=compare_key)
    return anno


def _bbox2poly(bbox):
    """ _bbox2poly """
    poly = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
    return poly


def _parse_ann_info_funsd(anno_path):
    """load annos from anno_path
    Input:
        anno_path: absolute path of annoataion file <Str>
    Output:
        res: (poly, transcript, text_class, ignore_tag) <Tuple>
    """
    res = []
    with codecs.open(anno_path, 'r', 'utf-8') as f:
        data = json.load(f)

    ## funsd word level
    for line in data['form']:
        for word in line['words']:
            ignore_tag = False
            box, transcript = word['box'], word['text']
            poly = _bbox2poly(list(map(float, box)))

            if len(transcript) == 0:
                transcript = ""
            else:
                for char in transcript:
                    if char not in Lexicon_Table_95:
                        transcript = ""
                        break
            res.append((poly, transcript, 0, ignore_tag))

    if len(res) == 0:
        return None
    return res


class Dataset(BaseDataset):
    """TextSpotting Dataset
    Input:
        config: train_config['dataset']  <Dict>
        feed_names: the training/testing fields <List>
    """
    def __init__(self, config, feed_names, train_mode=True):
        self.config = config

        batch_size = config['batch_size']
        data_path = config['data_path']
        image_path = config['image_path']

        assert os.path.isdir(image_path)

        if os.path.isdir(data_path):
            labels = glob.glob(data_path + '/*.*')
            label_list = [[label_path, image_path] for label_path in labels]
            label_list = [label_list]
        else:
            raise ValueError('error in load data_path for funsd: ', subdata_label_path)

        self.transform = build_transform(config['transform'])
        super(Dataset, self).__init__(
                dataset_list=label_list,
                feed_names=feed_names,
                batch_size=batch_size,
                train_mode=train_mode,
                collect_batch=True,
                shuffle=True)

        self.seq_len = config.get('max_seq_len', 50)
        self.recg_loss = config.get('recg_loss', 'CE')
        self.label_converter = LabelConverter(
            seq_len=self.seq_len,
            recg_loss=self.recg_loss)

    def _convert_examples(self, examples):
        """convert example to field
        """
        config = self.config
        example = examples[0]
        anno_path = example['boxes_and_texts_file']
        anno = _parse_ann_info_funsd(anno_path)
        if anno is None:
            return None

        # sort the box based on the position
        anno = _sort_box_with_list(anno)
        image = example['image']
        image_name = example['image_name']

        polys = []
        texts = []
        classes = []
        ignore_tags = []

        for poly, text, text_class, ignore_tag in anno:
            polys.append(poly)
            texts.append(self.label_converter.encode(text, ignore_tag))
            ignore_tags.append(ignore_tag)
            classes.append(text_class)

        label_word = {'image': image}
        label_word['polys'] = np.array(polys, dtype=np.float32).reshape(-1, 4, 2)
        label_word['texts'] = np.array(texts, dtype=np.int64)
        label_word['classes'] = np.array(classes, dtype=np.int64)
        label_word['ignore_tags'] = np.array(ignore_tags, dtype=np.bool)

        data = self.transform(label_word)
        data['org_image'] = image
        data['image_name'] = image_name
        return data

    def _read_data(self, example):
        """load image from image path and return image with data path
        Input:
            example: ['X51005268200.txt', '../text_spotting_data/sroie/train/image/'] <List>
        Output:
            example: readed image and label path <Dict>
        """

        data_path, image_path = example
        image_name = os.path.basename(data_path).replace('.json', '.png')
        image_path = os.path.join(image_path, image_name)

        if not os.path.exists(image_path):
            logging.warning('Dataset... The file (%s) is not existed!', image_path)
            return None
        try:
            image = cv2.imread(image_path)
        except:
            logging.debug('Dataset... Error in read %s', image_path)
            return None
        if image is None:
            logging.debug('Dataset... Error in load image for %s', image_path)
            return None

        example = {'image': image,
                   'boxes_and_texts_file': data_path,
                   'image_name': image_name}

        return example
