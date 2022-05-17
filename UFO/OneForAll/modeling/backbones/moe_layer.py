# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.utils import global_scatter, global_gather
from paddle.distributed import alltoall, all_gather
from paddle.distributed import fleet
from paddle.autograd import PyLayer
from paddle.distributed.fleet.meta_parallel.pp_utils.utils import _hp_recompute
from paddle.distributed.fleet.utils import recompute
from paddle import fluid
from paddle.nn.initializer import Constant
import copy
from paddle.nn.initializer import Constant

__all__ = ["MoeLayer"]

def hardpolicy(policy_task, tau, training):                                                                                               
    """hardpolicy                                                                                                                         
        params: policy_task                                                                                                               
                tau                                                                                                                       
                training                                                                                                                  
        returns: possiblity                                                                                                               
    """                                                                                                                                   
    eps = 1e-5
    gamma = 0.9999
    u = paddle.rand(policy_task.shape)
    u = paddle.clip(u, min=eps)
    u = paddle.clip(u, max=gamma)
    gumbels = -1 * paddle.log(-1 * paddle.log(u))                                                                                         
                                                                                                                                          
    if training:                                                                                                                          
        gumbels = (policy_task + gumbels) / tau                                                                                           
    else:                                                                                                                                 
        gumbels = (policy_task + 0) / tau                                                                                                 
                                                                                                                                          
    y_soft = paddle.nn.functional.softmax(gumbels, axis=-1)                                                                               
    # Straight through.                                                                                                                   
    base_value = paddle.zeros_like(policy_task, 'float32')                                                                                
    base_value.stop_gradient=True                                                                                                         
    overwrite_value = paddle.ones_like(policy_task, 'float32')                                                                            
    overwrite_value.stop_gradient=True                                                                                                    
    overwrite_index = paddle.argsort(y_soft)[:, -1:]                                                                                      
    overwrite_index.stop_gradient=True                                                                                                    
    base_index = paddle.tile(paddle.Tensor(np.array([[0, 1]])), (policy_task.shape[0], 1))                                                
    y_hard = paddle.where(base_index == overwrite_index, overwrite_value, base_value)                                                     
    possiblity = y_hard - y_soft.detach() + y_soft                                                                                        
    return possiblity

class MOEScatter(PyLayer):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

    @staticmethod
    def forward(ctx,
                inp,
                local_expert_count,
                global_expert_count,
                world_size,
                group=None):
        local_input_buf = inp
        if world_size > 1:
            global_input_buf = global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            global_input_buf = local_input_buf

        ctx.moe_args = inp.shape[0], world_size, group

        variables = (local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, grad):
        (local_expert_count, global_expert_count) = ctx.saved_tensor()
        (inp_batch_size, world_size, group) = ctx.moe_args

        origin_shape = grad.shape
        if world_size > 1:
            local_grad_in = global_gather(
                grad.reshape([0, -1]), local_expert_count, global_expert_count, group=group)
        else:
            local_grad_in = grad

        origin_shape[0] = -1
        grad_in = local_grad_in
        return grad_in.reshape(origin_shape), None, None


class MOEGather(PyLayer):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(ctx,
                global_output_buf,
                local_expert_count,
                global_expert_count,
                world_size,
                group=None):
        if world_size > 1:
            local_output_buf = global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            local_output_buf = global_output_buf

        output = local_output_buf

        ctx.moe_args = (global_output_buf.shape[0], world_size, group)
        variables = (local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        local_expert_count, global_expert_count = ctx.saved_tensor()
        fwd_batch_size, world_size, group = ctx.moe_args

        origin_shape = grad_out.shape
        grad_out_buf = grad_out

        if world_size > 1:
            global_grad_out_buf = global_scatter(
                grad_out_buf.reshape([0, -1]),
                local_expert_count,
                global_expert_count,
                group=group)
        else:
            global_grad_out_buf = grad_out_buf

        origin_shape[0] = -1
 
        return global_grad_out_buf.reshape(origin_shape), None, None

def _alltoall(in_tensor_list, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = len(in_tensor_list)
    return paddle._C_ops.alltoall(in_tensor_list, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id)

def prepare_forward(t, moe_group):
    with paddle.no_grad():
        task_id = paddle.unique(t)

        #NOTE: train logic!
        if task_id.shape[0] > 1:
            local_expert_count = paddle.concat([ (t == task_id[i]).astype("int64").sum() for i in range(task_id.shape[0])])
            global_expert_count = _alltoall(local_expert_count,group=moe_group)
        else:
            #NOTE: test logic
            assert len(task_id.shape) == 1, "the input of doing test has error!"
            local_expert_count = paddle.zeros([moe_group.nranks], dtype="int64") 
            local_expert_count[task_id[0].item()] = t.shape[0] 
            if paddle.distributed.get_rank() % moe_group.nranks == task_id[0].item():
                global_expert_count = paddle.ones([moe_group.nranks], dtype="int64") 
                global_expert_count *= t.shape[0]
            else:
                global_expert_count = paddle.zeros([moe_group.nranks], dtype="int64") 

    return local_expert_count, global_expert_count

class MoeLayer(nn.Layer):
    def __init__(self,
                 specific_experts,
                 common_expert,
                 moe_group=None,
                 gate_config=None,
                 mp_group=None,
                 use_checkpointing=False,
                 **kwargs):
        super(MoeLayer, self).__init__()

        if gate_config is None:
            gate_config = dict()

        # only support mp/dp
        self.group = moe_group

        self.world_size = 1
        if self.group is not None:
            self.world_size = self.group.nranks

        assert specific_experts is not None
        
        assert len(specific_experts) == 1, "the number of experts must be 1 now."
        self.num_expert = len(specific_experts)
        self.specific_experts = nn.LayerList(specific_experts)
        self.expert_weight = self.create_parameter(shape=[len(self.specific_experts), 2], default_initializer=Constant(value=0.))
        self.expert_weight.name = self.expert_weight.name + "exp_weight"
        self.common_expert = common_expert

        self.training = True
        self.use_checkpointing=use_checkpointing
        
        self.cache_count = None

    def forward(self, inp, t=None, tau=5, monitor=None):
        # inp shape: b * s * m
        # t: task id in task_MOE

        assert len(inp.shape) == 3
        origin_shape = inp.shape
        inp = inp.reshape_([0, -1])
        deal_shape = inp.shape  #[1024, 257, 768]

        #local_expert_count, global_expert_count = prepare_forward(t, moe_group)
        with paddle.no_grad():
            if self.cache_count is None:
                local_expert_count, global_expert_count = monitor._get_expert_count(t)
                self.cache_count = local_expert_count, global_expert_count
            else:
                local_expert_count, global_expert_count = self.cache_count

        global_samples = global_expert_count.sum().item()
        x = MOEScatter.apply(inp, local_expert_count, global_expert_count, monitor.moe_group.nranks, monitor.moe_group)
        #rank0: [1024,1024,1024,1024] [4096,257*768]
        cur_shape = x.shape # [4096, 257 * 768]
        if global_samples > 0:
            policy_task = self.expert_weight                                                                                             
            policy_task = paddle.tile(policy_task, (global_samples, 1))
            possiblity = hardpolicy(policy_task, tau, self.training)  

            
            x = x.reshape_([-1, origin_shape[1], origin_shape[2]]) #[4096, 257, 768]
            if self.use_checkpointing:
                common_outputs = recompute(self.common_expert, x)
                specific_outputs = recompute(self.specific_experts[0], x)
            else:
                common_outputs = self.common_expert(x)
                specific_outputs = self.specific_experts[0](x)
            
            #four task res
            #[4096, 257, 768]
            combine_output = common_outputs * possiblity[:, 0].unsqueeze(1).unsqueeze(2) \
                                     + specific_outputs * possiblity[:, 1].unsqueeze(1).unsqueeze(2) 
            combine_output = combine_output.reshape_(cur_shape) #[4096, 257*768]
        else:
            combine_output = x
            
        x = MOEGather.apply(combine_output, local_expert_count, global_expert_count, monitor.moe_group.nranks, monitor.moe_group)
        
        #[1024, 257, 768]
        x = x.reshape_(origin_shape)

        return x


