# Copyright (c) 
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging
import numpy as np
import pickle
# import torch
# import torch.distributed as dist
import paddle
import paddle.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size():
    """get_world_size
    """
    return dist.get_world_size()


def get_rank():
    """get_rank
    """
    return dist.get_rank()


def get_local_rank():
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    return dist.ParallelEnv().local_rank


def get_local_size():
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    return dist.ParallelEnv().local_rank


def is_main_process(): 
    """judge whether the current process is the main process
    """
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    pass


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    pass


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    pass


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    pass


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    pass
