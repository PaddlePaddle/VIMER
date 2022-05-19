import os
import sys

from .none_op import *
from .v2net import *

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

__all__ = ['build_backbone']

support_dict = ['NoneOp', 'V2Net']

def build_backbone(module_name, config):
    assert module_name in support_dict, Exception(
        "backbone only support {}, {} is unexpected".format(support_dict, module_name))
    module_class = eval(module_name)(**config)
    return module_class
