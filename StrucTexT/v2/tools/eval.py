""" eval.py """
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

ALL_MODULES = ['document_classify', 'end2end_ocr']
if args.task_type not in ALL_MODULES:
    raise ValueError('Not valid task_type %s in %s' % (args.task_type, str(ALL_MODULES)))

# modules
place = P.set_device('gpu')
Model, Dataset, Evaler = build_task(args.task_type)

def eval(config):
    """ eval """
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

    config['init_model'] = weights_path
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

    #metric
    eval_classes = build_metric(config['metric'])

    #start
    logging.info('eval start...')
    eval = Evaler(config=config,
                  model=model,
                  data_loader=eval_loader,
                  eval_classes=eval_classes)

    eval.run()
    logging.info('eval end...')

#start to eval
eval(config)
