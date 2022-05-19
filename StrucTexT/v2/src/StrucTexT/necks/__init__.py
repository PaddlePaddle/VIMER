""" build_neck """
import os
import sys
from .db_fpn import FPN as DB_FPN
from .east_fpn import FPN as EAST_FPN
from .text_grid import TexTGrid

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

__all__ = ['build_neck']

support_dict = ['DB_FPN', 'EAST_FPN', 'TexTGrid']

def build_neck(module_name, config):
    """ build_neck """
    assert module_name in support_dict, Exception(
        "neck only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
