# Copyright (c) 
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging
import numpy as np
import pickle

import paddle
import paddle.distributed as dist

_LOCAL_PROCESS_GROUP = None


def get_world_size(dp_group=None):
    """get_world_size
    """
    if dp_group is None:
        return dist.get_world_size()
    else:
        return dp_group.nranks


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
    # paddle.distributed.barrier()
    pass

def gather_v(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: tensor
        dst (int): destination rank
        group: 

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size(group) > 1:
        data_list = []
        paddle.distributed.all_gather(data_list, data, group)
    else:
        data_list = [data]
    return data_list


def gather(datas, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        datas: 列表数据有2种类型，
        类型1：list=[paddle.Tensor, paddle.Tensor, ...,  paddle.Tensor]
        类型2：list=[
                    {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, 
                    {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor},
                    ...
                    {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}
                    ]
    Returns:
        经过gathered后的数据data_list，
        对于类型1，data_list=[[paddle.Tensor,...], [paddle.Tensor,...], ..., ]
        对于类型2，data_list=[
                    [{'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, ... ], #长度为world-size
                    [{'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, ... ], 
                    ...
                    [{'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, {'key1':paddle.Tensor, 'key2':paddle.Tensor, ...,  'keyn':paddle.Tensor}, ... ], 
                    ]
        list[data]: 
    """
    if get_world_size(group) > 1:
        data_list = []
        for data in datas:
            if isinstance(data, paddle.Tensor):
                gathered_data = []
                paddle.distributed.all_gather(gathered_data, data, group)
                data_list.append(gathered_data)
            elif isinstance(data, dict):
                global_dict = {}
                for key, value in data.items():
                    gathered_value = []
                    paddle.distributed.all_gather(gathered_value, value, group)
                    global_dict[key] = gathered_value
                    
                gathered_data = []
                for i in range(len(gathered_value)):
                    local_dict = {}
                    for key in global_dict.keys():
                        local_dict[key] = global_dict[key][i]
                    gathered_data.append(local_dict)
                data_list.append(gathered_data)
    else:
        data_list = [datas]
    return data_list

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    return 0 #TODO add shared_random_seed


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
