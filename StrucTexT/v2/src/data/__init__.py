""" build_dataloader"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import paddle
import signal
import numpy as np
import paddle as P
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from data.dataset import BaseDataset
from utils.utility import get_apis
from data.transform import build_transform

__all__ = ['build_transform', 'build_dataloader']


def _term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


signal.signal(signal.SIGINT, _term_mp)
signal.signal(signal.SIGTERM, _term_mp)

def build_dataloader(config, dataset, mode, device, distributed=False):
    """ build_dataloader """

    loader_config = config['loader']
    datset_config = config['dataset']
    collect_batch = loader_config['collect_batch']
    num_workers = loader_config['num_workers']
    drop_last = loader_config.get('drop_last', True)
    shuffle = loader_config.get('shuffle', False)
    use_shared_memory = loader_config.get('use_shared_memory', False)

    if not distributed and not collect_batch:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=None,
            places=device,
            timeout=60,
            num_workers=num_workers,
            return_list=True)
        return data_loader
    if distributed and not collect_batch:
        batch_size = 1
        collate_fn = lambda x: x[0]
    else:
        batch_size = datset_config['batch_size']
        collate_fn = None

    batch_sampler = DistributedBatchSampler \
            if distributed else BatchSampler
    batch_sampler = batch_sampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        places=device,
        timeout=60,
        num_workers=num_workers,
        return_list=True)

    return data_loader
