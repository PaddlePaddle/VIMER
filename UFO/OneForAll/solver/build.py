"""solve/build.py
"""
from typing import Optional, Dict, List, Any, Set, Type
import re
import copy
import math

import paddle


def build_lr_optimizer_lazy(**kwargs):
    """build_lr_optimizer_lazy
    """
    optimizer_type = kwargs.get('optimizer_type', 'SGD') 
    model = kwargs['model']
    momentum = kwargs['momentum']
    weight_decay = kwargs['weight_decay']
    lr_multiplier = kwargs['lr_multiplier']
    grad_clip_enabled = kwargs.get('grad_clip_enabled', True)
    grad_clip_norm = kwargs.get('grad_clip_norm', 5.0)

    if optimizer_type == 'SGD':
        return  paddle.optimizer.Momentum(
                learning_rate=lr_multiplier,
                momentum=momentum,
                parameters=model.parameters(),
                weight_decay=weight_decay,
                grad_clip= paddle.nn.ClipGradByNorm(grad_clip_norm) if grad_clip_enabled else None)
    elif optimizer_type == 'Adam':
        return paddle.optimizer.Adam(learning_rate=lr_multiplier, 
                beta1=0.9, beta2=0.999, 
                epsilon=1e-08, 
                parameters=model.parameters(), 
                weight_decay=weight_decay, 
                grad_clip=paddle.nn.ClipGradByNorm(grad_clip_norm) if grad_clip_enabled else None, 
                name=None, 
                lazy_mode=False 
                )
    elif optimizer_type == 'AdamW':
        return paddle.optimizer.AdamW(learning_rate=lr_multiplier, 
                beta1=0.9, beta2=0.999, 
                epsilon=1e-08, 
                parameters=model.parameters(), 
                weight_decay=weight_decay, 
                lr_ratio=None, 
                apply_decay_param_fun=None, 
                grad_clip=paddle.nn.ClipGradByNorm(grad_clip_norm) if grad_clip_enabled else None, 
                lazy_mode=False, 
                multi_precision=False, 
                name=None )
    else:
        raise ValueError()


def build_lr_scheduler_lazy(**kwargs):
    """build_lr_scheduler_lazy
    """
    warmup_iters = kwargs['warmup_iters']
    warmup_method = kwargs['warmup_method']
    max_iters = kwargs['max_iters']
    # warmup_factor TODO add warmup_factor
    sched =  kwargs['sched']
    eta_min = kwargs['eta_min']
    base_lr = kwargs['base_lr']
    if warmup_method == 'linear' and sched == 'CosineAnnealingLR': 
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            paddle.optimizer.lr.CosineAnnealingDecay(base_lr, max_iters, eta_min), 
            warmup_iters, 
            0., 
            base_lr)
    else:
        raise ValueError("Unknown warmup and sched method : {} and {}".format(warmup_method, sched))
    return lr_scheduler
