""" make_funsd_data.py """
import io
import os
import re
import sys
import cv2
import glob
import gzip
import json
import codecs
import logging
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from collections import namedtuple
from model.ernie.tokenizing_ernie import ErnieTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

whit_space_pat = re.compile(r'\S+')
pat = re.compile(r'([a-zA-Z0-9]+|\S)')
cls_dict = {'other': 0, 'header': 1, 'question': 2, 'answer': 3}

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

def parse_txt(line, tokenizer):
    """
    char tokenizer (wordpiece english)
    normed txt(space seperated or not) => list of word-piece
    """
    ret_line = []
    line = line.lower()
    if len(line) == 0:
        return ret_line
    for sub in pat.finditer(line):
        s, e = sub.span()
        sen = line[s:e]
        for ll in tokenizer.tokenize(sen):
            ret_line.append(ll)

    #ret_line = tokenizer.convert_tokens_to_ids(ret_line)
    return ret_line


def read_example_from_file(label, image_name, tokenizer):
    """
    convert label_file to pickle example
    """

    ids = []
    tokens = []
    line_bboxes = []
    token_bboxes = []
    cls_label = []
    linkings = []
    for idx, line in enumerate(label):
        id = line['id']
        line_bbox = line['box']
        cls = cls_dict.get(line['label'])
        linking = line['linking']
        for w in line['words']:
            if len(w['text'].strip()) == 0:
                w['text'] = '[UNK]'
        text = [x['text'] for x in line['words']]
        token_bbox = [x['box'] for x in line['words']]
        token = [parse_txt(word, tokenizer) for word in text]

        ids.append(id)
        tokens.append(token)
        token_bboxes.append(token_bbox)
        line_bboxes.append(line_bbox)
        linkings.append(linking)
        cls_label.append(cls)
    example = {'ids': ids,
               'cls': cls_label,
               'tokens': tokens,
               'line_bboxes': line_bboxes,
               'token_bboxes': token_bboxes,
               'linkings': linkings,
               'image_name': image_name}
    return example


def build_dataset(label_dir, out_dir, config):
    """
    extract & parse doc_label
    """
    os.makedirs(out_dir, exist_ok=True)
    dict_params = config['dict_params']
    tokenizer = ErnieTokenizer.init(config['dict_path'], **dict_params)
    label_files = glob.glob(os.path.join(label_dir, '*.json'))
    for label_file in label_files:
        image_name = os.path.basename(label_file).replace('json', 'png')
        with codecs.open(label_file, 'r', 'utf-8') as f:
            label = json.load(f)['form']
        example = read_example_from_file(label, image_name, tokenizer)
        if example is None:
            raise ValueError('Failed to parse label of (%s)', image_path)
        save_path = os.path.join(out_dir, image_name + '.label')
        with gzip.open(save_path, 'wb') as w:
            w.write(json.dumps(example).encode('utf-8'))
        logging.info('Parsing %s', image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FUNSD Data Maker')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    assert os.path.exists(args.config_file)
    config = json.loads(open(args.config_file).read())

    build_dataset(args.label_dir, args.out_dir, config['eval']['dataset'])
    logging.info('done: %s' % args.out_dir)
