"""save_load"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import errno
import pickle
import six
import logging

import paddle

__all__ = ['load_model', 'save_model']


def _mkdir_if_not_exist(path):
    '''
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logging.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_model(config, model, optimizer=None):
    '''
    load model from checkpoint or pretrained_model
    '''
    global_config = config['global']
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}
    if checkpoints and len(checkpoints) > 0:
        if checkpoints.endswith('.pdparams'):
            checkpoints = checkpoints.replace('.pdparams', '')
        assert os.path.exists(checkpoints + '.pdparams'), \
            'The {}.pdparams does not exists!'.format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + '.pdparams')
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logging.warning('{} not in loaded params!'.format(
                    key))
                continue
            pre_value = params[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logging.warning(
                    'The shape of model params {} {} not matched with loaded params shape {} !'.
                    format(key, value.shape, pre_value.shape))
        model.set_state_dict(new_state_dict)

        if optimizer is not None:
            if os.path.exists(checkpoints + '.pdopt'):
                optim_dict = paddle.load(checkpoints + '.pdopt')
                optimizer.set_state_dict(optim_dict)
            else:
                logging.warning(
                    '{}.pdopt is not exists, params of optimizer is not loaded'.
                    format(checkpoints))

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1
        logging.info('resume from {}'.format(checkpoints))
    elif pretrained_model and len(pretrained_model) > 0:
        if isinstance(pretrained_model, str):
            load_pretrained_params(model, pretrained_model)
        if isinstance(pretrained_model, dict):
            pretrained_model = [{k:v} for k, v in pretrained_model.items()]
        if isinstance(pretrained_model, list):
            for item in pretrained_model:
                if 'replace_segs' in item:
                    replace_segs = item.pop('replace_segs')
                else:
                    replace_segs = []
                append_prefix = list(item)[0]
                model_path = item[append_prefix]
                load_pretrained_params(model, model_path, append_prefix, replace_segs)
    else:
        logging.info('train from scratch')
    return best_model_dict


def load_pretrained_params(model, path, append_prefix='', replace_segs=[]):
    """ load_pretrained_params """
    if path.endswith('.pdparams'):
        path = path.replace('.pdparams', '')
    assert os.path.exists(path + '.pdparams'), \
        'The {}.pdparams does not exists!'.format(path)

    params = paddle.load(path + '.pdparams')
    state_dict = model.state_dict()
    new_state_dict = {}
    for param_name in params.keys():
        name = param_name
        for seg in replace_segs:
            key = list(seg)[0]
            val = seg[key]
            name = name.replace(key, val)
        name = append_prefix + name
        if name in state_dict.keys():
            if list(state_dict[name].shape) == list(params[param_name].shape):
                logging.debug('Loading... find weights `%s` to `%s` in %s', param_name, name, path)
                new_state_dict[name] = params[param_name]
            else:
                logging.warning(
                        'Loading... The shape of model params {} {} not matched with loaded params {} {} in {}!'.
                        format(name, state_dict[name].shape, param_name, params[param_name].shape, path))
        else:
            logging.warning('Loading... ignore weights `%s` in %s', param_name, path)

    model.set_state_dict(new_state_dict)
    logging.info('load pretrain successful from {}'.format(path))
    return model


def save_model(model,
               optimizer,
               model_path,
               is_best=False,
               prefix='',
               **kwargs):
    '''
    save model to the target path
    '''
    if paddle.distributed.get_rank() != 0:
        return
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(model.state_dict(), model_prefix + '.pdparams')
    paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')

    # save metric and config
    with open(model_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logging.info('Saving... save best model is to {}'.format(model_prefix))
    else:
        logging.info('Saving... save model in {}'.format(model_prefix))
