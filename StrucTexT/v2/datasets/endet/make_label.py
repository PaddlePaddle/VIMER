# !/usr/bin/env python3
""" make endet data.py """
import io
import os
import re
import sys
import six
import gzip
import glob
import ujson
import logging
import argparse
import random as r
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from collections import namedtuple
# from model.ernie.tokenizing_ernie import ErnieTokenizer
from src.StrucTexT.ernie import ErnieTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

whit_space_pat = re.compile(r'\S+')
pat = re.compile(r'([a-zA-Z0-9]+|\S)')

def merge_subword(tokens):
    """
    :param tokens:
    :return: merged_tokens
    """
    ret = []
    for token in tokens:
        if token.startswith("##"):
            real_token = token[2:]
            if len(ret):
                ret[-1] += real_token
            else:
                ret.append(real_token)
        else:
            ret.append(token)
    return ret


def parse_txt(line, tokenizer, take_subword=False):
    """
    char tokenizer (wordpiece english)
    normed txt(space seperated or not) => list of word-piece
    """
    pos = []
    ret_line = []
    if len(line) == 0:
        return ret_line
    for sub in pat.finditer(line):
        s, e = sub.span()
        sen = line[s:e]
        pos.append([s, e])
        if take_subword:
            sub_words = []
            for ll in tokenizer.tokenize(sen):
                sub_words.append(ll)
            ret_line.append(sub_words)
        else:
            for ll in tokenizer.tokenize(sen):
                ret_line.append(ll)

    if r.random() < 1e-7:
        print('****', file=sys.stderr)
        print(line, file=sys.stderr)
        print('|'.join(ret_line), file=sys.stderr)
        print('****', file=sys.stderr)

    return ret_line, pos


def read_example_from_file(label_json, image_path, kv_dict, tokenizer):
    """
    convert label_file to pickle example
    """
    image_name = os.path.basename(image_path)

    tokens = []
    line_bboxes = []
    cls_labels = []
    txt_pos_dict = {}

    label_json = ujson.loads(label_json)
    label_polys = label_json["boxes"]
    for one_poly in label_polys:
        t_poly = one_poly["box"]
        t_cls = one_poly["class"]

        flatten_poly = []
        for item in t_poly:
            flatten_poly += item
        if t_cls not in txt_pos_dict:
            txt_pos_dict[t_cls] = [flatten_poly]
        else:
            txt_pos_dict[t_cls].append(flatten_poly)

    print_str = ""
    for k, v in txt_pos_dict.items():
        if str(k) not in kv_dict:
            continue
        token_text = kv_dict[str(k)]
        token = [parse_txt(token_text, tokenizer)[0]]
        tokens.append(token)
        line_bboxes.append(v)
        cls_labels.append(int(k))
        print_str += "#" + str(len(v))
        if len(v) > 10:
            print("image name = {}, tokens len = {}, line len = {}".format(image_name, len(tokens), print_str))

    # print("image name = {}, tokens len = {}, line len = {}".format(image_name, len(tokens), print_str))

    example = {'image_name': image_name,
                'line_bboxes': line_bboxes,
                'token_bboxes': line_bboxes,
                'tokens': tokens,
                'token_classes': cls_labels
                }
    return example


def build_dataset(config, image_dir, label_dir, out_dir, kv_file, \
        label_list=None, no_check_image=False, aggregate_num=-1):
    """
    extract & parse doc_label
    """
    if aggregate_num < 1:
        os.makedirs(out_dir, exist_ok=True)
    dict_params = config['dict_params']
    tokenizer = ErnieTokenizer.init(config['dict_path'], **dict_params)
    if isinstance(label_list, str) and \
       os.path.exists(label_list) and \
       label_list.endswith('.txt'):
        with open(label_list) as f:
            files = f.readlines()
    else:
        files = glob.glob(os.path.join(label_dir, '*.txt'))
    logging.info('load %d txt files' % len(files))

    ## read kv_file
    kv_dict = {}
    with open(kv_file, "r") as fid:
        for t_line in fid.readlines():
            t_seg = t_line.strip().split(" ")
            kv_dict[t_seg[1]] = t_seg[0]

    index = 0
    examples = []
    for ex_index, file in enumerate(files, 1):
        file = file.strip()
        file = file.replace("\"\"", "#")
        file = file.replace("\"", "")
        file = file.replace("#", "\"")

        i_pos = file.find("{\"boxes")
        label_json = file[i_pos:]
        file_name = os.path.basename(file[:(i_pos - 1)])

        image_path = os.path.join(image_dir, file_name)
        if not no_check_image and not os.path.exists(image_path):
            is_find = False
            for suffix in ['jpg', 'bmp', 'png', 'jpeg', 'tif']:
                image_name = image_path + '.' + suffix
                if os.path.exists(image_name):
                    is_find = True
                    image_path = image_name
                    break
                image_name = image_path + '.' + suffix.upper()
                if os.path.exists(image_name):
                    is_find = True
                    image_path = image_name
                    break
            if not is_find:
                logging.warning('The image_path (%s) is not existed!', image_path)
                continue
        example = read_example_from_file(label_json, image_path, kv_dict, tokenizer)
        if example is None:
            continue
        if aggregate_num < 1:
            save_path = os.path.basename(image_path) + '.label'
            save_path = os.path.join(out_dir, save_path)
            with gzip.open(save_path, 'wb') as w:
                w.write(ujson.dumps(example).encode('utf-8'))
        else:
            examples.append(example)
            if len(examples) % aggregate_num == 0:
                logging.info("Extracting example %d", ex_index)
                logging.info("Save examples at %s", out_dir)
                with gzip.open(out_dir + ('.%03d' % index), 'wb') as w:
                    examples = ujson.dumps(examples)
                    w.write(examples.encode('utf-8'))
                examples = []
                index += 1
    if aggregate_num > 0:
        with gzip.open(out_dir + ('.%03d' % index), 'wb') as w:
            examples = ujson.dumps(examples)
            w.write(examples.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretrain Data Maker')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--kv_file', type=str, required=True)
    parser.add_argument('--label_list', type=str, default='')
    parser.add_argument('--no_check_image', action='store_true')

    args = parser.parse_args()
    for arg, value in sorted(six.iteritems(vars(args))):
        logging.warning('%s: %s' % (arg, value))
    assert os.path.exists(args.config_file)
    config = ujson.loads(open(args.config_file).read())

    build_dataset(config['eval']['dataset']['pretrain']['ENDET'], args.image_dir, args.label_dir, \
            args.out_dir, args.kv_file, args.label_list, args.no_check_image)
    logging.info('done: %s' % args.out_dir)
