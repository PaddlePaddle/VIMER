""" __init__ """
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import copy

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from .ops.operators import *
from .ops.batch_operators import *
from .ops.autoaugment import *
from .ops.cutout import *
from .ops.grid import *
from .ops.randaugment import *
from .ops.hide_and_seek import *
from .ops.random_erasing import *
from .ops.timm_autoaugment import *

from .east_process import *
from .db_process import *

__all__ = ['build_transform']

def create_operators(op_param_list, module=None, global_config=None):
    """
    create operators based on the config
    Args:
        op_param_list(list): a dict list, used to create some operators
        module(str|module): the module or path
        global_config(dict): update params
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    if isinstance(module, str):
        module = importlib.import_module(module)
    for operator in op_param_list:
        assert isinstance(operator,
                dict) and len(operator) == 1, "format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None and isinstance(param, dict):
            param.update(global_config)
        if module:
            op = getattr(module, op_name)(**param)
        else:
            op = eval(op_name)(**param)
        ops.append(op)
    return ops


def build_transform(op_param_list,
        preprocess_fun=None,
        postprocess_fun=None,
        global_config=None):
    """ build_transform """
    op_list = create_operators(op_param_list, global_config=global_config)
    def transform(data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        if preprocess_fun:
            data = preprocess_fun(data)
        for op in ops:
            data = op(data)
            if data is None:
                return None
        if postprocess_fun:
            data = postprocess_fun(data)
        return data
    transformer = lambda x: transform(x, op_list)

    return transformer
