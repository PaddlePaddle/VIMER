""" utility.py """
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the 'License');
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an 'AS IS' BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import inspect
import six
import importlib
import distutils.util

__all__ = ['get_apis',
           'create_operators',
           'print_arguments',
           'add_arguments']

def get_apis(current_module):
    """
    go through and return all the functions and classes for given module
    Args:
        current_module(module): a instance of module
    """
    current_func = sys._getframe().f_code.co_name
    api = []
    obj_list = inspect.getmembers(current_module,
                                  inspect.isclass) + \
               inspect.getmembers(current_module,
                                  inspect.isfunction)
    for _, obj in obj_list:
        api.append(obj.__name__)
    api.remove(current_func)
    return api

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
        if global_config is not None:
            param.update(global_config)
        if module:
            op = getattr(module, op_name)(**param)
        else:
            op = eval(op_name)(**param)
        ops.append(op)
    return ops


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument('name', default='Jonh', type=str, help='User name.')
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument('name', str, 'Jonh', 'User name.', parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        '--' + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)
