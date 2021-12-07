""" make_ephoie_data.py """
import io
import os
import re
import sys
import cv2
import gzip
import glob
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
cls_dict = {'other': 0, 'grade': 1, 'subject': 2, \
        'school': 3, 'test time': 4, 'class': 5, \
        'name': 6, 'examination number': 7, 'score': 8, \
        'seat number': 9, 'student number': 10}
types_dict = {'VALUE': 0, 'KEY': 1, 'NONE': 2}

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


def read_example_from_file(label, kvpair, image_name, tokenizer):
    """
    convert label_file and kvpair to pickle example
    """

    ids = []
    types = []
    tokens = []
    line_bboxes = []
    token_bboxes = []
    cls_label = []
    entities = {}
    for k, v in kvpair.items():
        entities[cls_dict.get(k)] = v
    for id, line in label.items():
        id = int(id)
        cls = list(map(lambda x: x if x <= 10 else 0, line['tag']))
        text = list(line['string'])
        char_num = len(text)
        text_quad = line['box']
        line_type = types_dict[line['class']]
        assert (len(cls) == char_num and char_num > 0), 'The number of charactor(%d) and \
                cls_label(%d) must be equal in %s' % (char_num, len(cls), image_name)
        token = [parse_txt(char, tokenizer) for char in text]
        line_bbox = [text_quad[0], text_quad[1], text_quad[4], text_quad[5]]

        line_width = text_quad[2] - text_quad[0]
        line_height = text_quad[7] - text_quad[1]
        if len(text) > 1 and line_height > 2 * line_width:
            char_dx1 = (text_quad[6] - text_quad[0]) // char_num
            char_dy1 = (text_quad[7] -  text_quad[1]) // char_num
            char_dx2 = (text_quad[4] - text_quad[2]) // char_num
            char_dy2 = (text_quad[5] -  text_quad[3]) // char_num
        else:
            char_dx1 = (text_quad[2] - text_quad[0]) // char_num
            char_dy1 = (text_quad[3] -  text_quad[1]) // char_num
            char_dx2 = (text_quad[4] - text_quad[6]) // char_num
            char_dy2 = (text_quad[5] -  text_quad[7]) // char_num
        token_bbox = [[text_quad[0] + i * char_dx1,
                       text_quad[1] + i * char_dy1,
                       text_quad[4] - (char_num - i - 1) * char_dx2,
                       text_quad[5] - (char_num - i - 1) * char_dy2]
                       for i in range(char_num)]

        ids.append(id)
        types.append(line_type)
        tokens.append(token)
        token_bboxes.append(token_bbox)
        line_bboxes.append(line_bbox)
        cls_label.append(cls)
    example = {'ids': ids,
               'token_cls': cls_label,
               'types': types,
               'tokens': tokens,
               'entities': entities,
               'line_bboxes': line_bboxes,
               'token_bboxes': token_bboxes,
               'image_name': image_name}
    return example


def build_dataset(label_dir, label_file, kvpair_dir, out_dir, config):
    """
    extract & parse doc_label
    """
    os.makedirs(out_dir, exist_ok=True)
    dict_params = config['dict_params']
    tokenizer = ErnieTokenizer.init(config['dict_path'], **dict_params)
    if os.path.isfile(label_file):
        with open(label_file) as f:
            label_files = [os.path.join(label_dir, x.strip()) for x in f.readlines()]
    else:
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        label_name = os.path.basename(label_file)
        image_name = label_name.replace('txt', 'jpg')
        kvpair_file = os.path.join(kvpair_dir, label_name)
        if not os.path.isfile(kvpair_file):
            raise ValueError('%s is not found' % kvpair_file)
        with codecs.open(label_file, 'r', 'utf-8') as f:
            label = json.load(f)
        with codecs.open(kvpair_file, 'r', 'utf-8') as f:
            kvpair = json.load(f)
        example = read_example_from_file(label, kvpair, image_name, tokenizer)
        if example is None:
            raise ValueError('Failed to parse label of (%s)', image_path)
        save_path = os.path.join(out_dir, image_name + '.label')
        with gzip.open(save_path, 'wb') as w:
            w.write(json.dumps(example).encode('utf-8'))
        logging.info('Parsing %s', image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EPHOIE Data Maker')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--label_file', type=str, default='')
    parser.add_argument('--kvpair_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    assert os.path.exists(args.config_file)
    config = json.loads(open(args.config_file).read())

    build_dataset(args.label_dir, args.label_file, args.kvpair_dir,
            args.out_dir, config['eval']['dataset'])
    logging.info('done: %s' % args.out_dir)
