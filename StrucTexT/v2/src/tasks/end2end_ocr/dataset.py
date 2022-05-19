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
import gzip
import codecs
import logging
import numpy as np
from copy import deepcopy

import paddle
import paddle.fluid as fluid

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from glob import glob
from src.data import build_transform
from src.data.dataset import BaseDataset


Lexicon_Table_95 = ['!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', \
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', \
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', \
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', \
    'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ']


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


def _parse_ann_info_data(anno_path):
    """load annos from anno_path
    Input:
        anno_path: absolute path of annoataion file <Str>
    Output:
        res: (poly, transcript, text_class, ignore_tag) <Tuple>
    """
    res = []
    with gzip.open(label_file, 'rb') as f:
            example = json.loads(f.read().decode('utf-8'))
    with codecs.open(anno_path, 'r', 'utf-8') as fp:    # fix some ascii bug
        data = json.load(fp)

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


def _parse_ann_info_funsd_line(anno_path): 
    """load annos from anno_path
    Input:
        anno_path: absolute path of annoataion file <Str>   
    Output:
        res: (poly, transcript, text_class, ignore_tag) <Tuple>
    """
    res = []

    with codecs.open(anno_path, 'r', 'utf-8') as fp:    # fix some ascii bug
        data = json.load(fp)

    # funsd word level
    for line in data['form']:
        ignore_tag = False
        box, transcript, label = line['box'], line['text'], line['label']
        box = list(map(float, box))
        poly = self._bbox2poly(box)

        if transcript in ['*', '###'] or len(transcript) == 0:
            # ignore_tag = True
            transcript = ""
        else:
            for char in transcript:
                if char not in Lexicon_Table_95:
                    # ignore_tag = True
                    transcript = ""
                    break
        try:
            text_class = TEXT_CLASSES[label]
        except:
            text_class = 3
        res.append((poly, transcript, text_class, ignore_tag))

    if len(res) == 0:
        return None
    return res

class Dataset(BaseDataset):
    """ END2END IE DATASET """
    def __init__(self, config, feed_names, train_mode=True):
        """__init__"""
        self.config = config
        self.tokenizer = ErnieTokenizer.init(
                config['dict_path'], **config['dict_params'])
        vocab = self.tokenizer.vocab
        pad_token = vocab[config['dict_params'].get('pad_token')]

        batch_size = config['batch_size']
        data_path = config['data_path']
        dict_params = config['dict_params']
        self.max_seqlen = config['max_seqlen']

        if os.path.isdir(data_path):
            image_path = config['image_path']
            labels = glob(os.path.join(data_path, '*.*'))
            label_list = [[label_path, image_path] for label_path in labels]
            label_list = [label_list]
        elif os.path.isfile(data_path):
            with open(data_path, 'r', 'utf-8') as f:
                data_list = list(filter(lambda x: len(x) == 2,
                    [x.strip().split('\t') for x in f.readlines()]))
                label_list = []
                for data_path, image_path in data_list:
                    labels = glob(os.path.join(data_path, '*.*'))
                    sub_data = [[label_path, image_path] for label_path in labels]
                    label_list.append(sub_data)
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

        self.seq_len = config.get('max_seq_len', 50)
        self.recg_loss = config.get('recg_loss', 'CE')
        self.label_converter = LabelConverter(
            seq_len=self.seq_len, 
            recg_loss=self.recg_loss)

    def _convert_examples(self, examples):
        """convert example to field
        """
        def _flatten_2d_list(c):
            return [i for item in c for i in item]

        config = self.config
        data = {'batch_size': len(examples)}

        sentence = np.ones((batch_size, max_seqlen), dtype='int32') * pad_token
        sentence_mask = np.zeros((batch_size, max_seqlen), dtype='int32')

        for idx, example in enumerate(examples):
            example = examples[idx]
            cls = example['cls']
            line_bb = example['line_bboxes']
            token_bb = example['token_bboxes']
            tokens = example['tokens']


            #### word parse ####
            words = _flatten_2d_list(tokens)
            word_bbs = _flatten_2d_list(token_bb)
            word_cls = _flatten_2d_list([[c] * len(line) for c, line in zip(cls, tokens)])
            polys, texts, classes, ignore_tags = [], [], [], []
            for box, word, c in zip(word_bbs, words, word_cls)
                ignore_tag = False
                classes.append(c)
                polys.append(_bbox2poly(box))
                #text = ''.join([w.replace('##', '') for w in word])
                #texts.append(label_converter.encode(text, ignore_tag))
                texts.append(self.tokenizer.convert_tokens_to_ids(word[0]))
                ignore_tags.append(ignore_tag)
            label_word = {}
            label_word['polys'] = np.array(polys, dtype=np.float32).reshape(-1, 4, 2)
            label_word['texts'] = np.array(texts, dtype=np.int64)
            label_word['classes'] = np.array(classes, dtype=np.int64)
            label_word['ignore_tags'] = np.array(ignore_tags, dtype=np.bool)

            #### line parse ####
            polys, texts, classes, ignore_tags = [], [], [], []
            for box, line, c in zip(line_bb, tokens, cls):
                ignore_tag = False
                classes.append(c)
                polys.append(_bbox2poly(box))
                text = = [''.join(w.replace('##', '').upper() for w in line]
                text = ' '.join(text)
                texts.append(label_converter.encode(text, ignore_tag))
                ignore_tags.append(ignore_tag)

            label_line = {}
            label_line['polys'] = np.array(polys_line, dtype=np.float32).reshape(-1, 4, 2)
            label_line['texts'] = np.array(texts_line, dtype=np.int64)
            label_line['classes'] = np.array(classes_line, dtype=np.int64)
            label_line['ignore_tags'] = np.array(ignore_tags_line, dtype=np.bool)

            example['labels'] = [label_word, label_line] #位置不允许调换

            #### preprocess ####
            transform_out = self.transform(example)
            if transform_out is None:
                return None
            if 'labels' in transform_out.keys():
                label_word = transform_out['labels'][0]
                label_line = transform_out['labels'][1]
                for key, val in label_word.items():
                    if key not in data.keys():
                        data[key] = []
                    data[key].append(val)
                for key, val in label_line.items():
                    new_key = key + '_line'
                    if new_key not in data.keys():
                        data[new_key] = []
                    data[new_key].append(val)

            for key, val in transform_out.items():
                if key == 'labels':
                    continue
                if key not in data.keys():
                    data[key] = []
                data[key].append(val)

        return data

    def _read_data(self, example):
        data_path, image_path = example
        if not os.path.exists(data_path):
            logging.warning('Dataset... The file (%s) is not existed!', data_path)
            return None
        try:
            with gzip.open(data_path, 'rb') as f:
                example = json.loads(f.read().decode('utf-8'))
        except:
            return None
        if len(example['line_bboxes']) < 1:
            logging.debug('Dataset... The number of line_bboxes is zero')
            return None
        if 'image_name' not in example.keys():
            logging.debug('Dataset... Not find image_name')
            return None
        image_path = os.path.join(image_path, example['image_name'])
        if not os.path.exists(image_path.encode('utf-8')):
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
        example['image'] = image
        return example
