""" build_loss """
import os
import sys

from .iou_loss import IouLoss, GIoULoss, DIouLoss

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

__all__ = ['build_loss']

support_dict = ['IouLoss', 'GIoULoss', 'DIouLoss']

def build_loss(module_name, config):
    """ build_loss """
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
