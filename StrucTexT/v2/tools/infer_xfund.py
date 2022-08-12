""" infer_xfund.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import json
import logging
import argparse
import functools
import importlib
import numpy as np
import paddle as P
import time

LOG_FORMAT = '%(asctime)s - %(levelname)s: %(message)s' # [%(filename)s:%(lineno)d]'
logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt='%Y-%m-%d %H:%M:%S')

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from src.utils.utility import add_arguments, print_arguments
from src.data import build_dataloader
from src.tasks import build_task
from src.metrics import build_metric
from src.postprocess.ocr_postprocess import grouping

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# sysconf
# base
parser = argparse.ArgumentParser('launch for eval')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--task_type', type=str, required=True)
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)

args = parser.parse_args()
print_arguments(args)
config = json.loads(open(args.config_file).read())

ALL_MODULES = ['document_classify', 'end2end_ocr', 'end2end_ie', 'table_recognition', 'end2end_ie_xfund']
if args.task_type not in ALL_MODULES:
    raise ValueError('Not valid task_type %s in %s' % (args.task_type, str(ALL_MODULES)))

# modules
place = P.set_device('gpu')
Model, Dataset, Evaler = build_task(args.task_type)

def infer(config):
    """ infer """
    # program
    eval_config = config['eval']
    model_config = config['architecture']

    label_path = args.label_path
    image_path = args.image_path
    weights_path = args.weights_path

    assert weights_path.endswith('.pdparams') and \
            os.path.isfile(weights_path), \
            'the weights_path %s is not existed!' % weights_path
    assert os.path.exists(label_path), 'the label_dir %s is not existed!' % label_path
    assert os.path.isdir(image_path), 'the image_dir %s is not existed!' % image_path

    eval_config['dataset']['data_path'] = label_path
    eval_config['dataset']['image_path'] = image_path


    eval_dataset = Dataset(
            eval_config['dataset'],
            eval_config['feed_names'],
            False)

    eval_loader = build_dataloader(
            config['eval'],
            eval_dataset,
            'Eval',
            place,
            False)
    #model
    model = Model(model_config, eval_config['feed_names'])

    # load params
    para_path = weights_path
    if os.path.exists(para_path):
        para_dict = P.load(para_path)
        model.set_dict(para_dict)
        logging.info('Load init model from %s', para_path)

    model.eval()

    loader = eval_loader()
    for step_idx in range(len(eval_loader)):
        print('deal with %i' % step_idx)
        input_data = next(loader)
        feed_names = eval_config['feed_names']
        output = model(*input_data, feed_names=feed_names)
        ######### infer ##########
        image_name = input_data[-2][0]

        ratio_info = input_data[-1].numpy().squeeze()
        _, _, ratio_h, ratio_w = ratio_info[0], ratio_info[1], ratio_info[2], ratio_info[3]

        preds = output['line_preds']
        os.makedirs("./pred_result/", exist_ok=True) 

        with open ("./pred_result/" + os.path.splitext(image_name)[0] + ".txt", 'w') as f:
            for pred in preds[0]:
                pts = np.array(pred[0], np.float32).reshape((-1, 1, 2))
                pts[:, :, 0] /= ratio_w
                pts[:, :, 1] /= ratio_h
                pts = pts.astype(np.int32)
                cls = pred[1]

                pts = pts.reshape(-1)
                pts = list(map(str, pts))
                cstr = os.path.splitext(image_name)[0]+'_'+str(num).zfill(3)+".jpg" + '\t' + " ".join(pts) + '\t' + str(cls) + '\n'
                f.write(cstr)
                num+=1

#start to infer
infer(config)
