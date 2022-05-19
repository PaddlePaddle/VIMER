import os
import sys

from .detr_head import *

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

__all__ = ['build_head']

support_dict = ['DETRHead']

def build_head(module_name, config):
    assert module_name in support_dict, Exception(
        "downstream head only support {}, {} is unexpected".format(support_dict, module_name))
    module_class = eval(module_name)(**config)
    return module_class
