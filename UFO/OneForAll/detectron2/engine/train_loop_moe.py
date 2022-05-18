# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
from typing import List, Mapping, Optional
import pickle

import paddle
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients, _apply_collective_grads

from utils import comm
# import fastreid.engine
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from .train_loop import HookBase, TrainerBase

__all__ = ["SimpleTrainer"]


def all_reduce_parameters(params, group, monitor=None, comm_flag=True):
    """
      group: used to communicate.
      monitor: used to control the number of ranks in comunications.
    """
    if group.nranks < 2:
        return
    with paddle.framework.no_grad():
        div_factor = group.nranks if monitor is None else monitor.moe_group.nranks
        for p in params:
            if p.grad is None:
                continue
            if comm_flag:
                paddle.distributed.all_reduce(p.grad, use_calc_stream=True, group=group)
                p._reset_grad_inplace_version(True)
            paddle.fluid.framework._dygraph_tracer().trace_op(
                type="elementwise_div",
                inputs={'X': p.grad,
                        'Y': paddle.to_tensor(div_factor, dtype=p.grad.dtype)},
                outputs={'Out': p.grad},
                attrs={'axis': -1})

class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, monitor=None):
        """
        Args:
            model: 
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: optimizer.
            dp_group: data parallel group.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.grad_scaler = None
        self.monitor = monitor

        self.sub_group = monitor.sub_group
        self.dp_group = monitor.dp_group
        self.moe_group = monitor.moe_group

        self.other_param, expert_weight_param, self.specific_expert_param, common_expert_param, batch_norm_nouse = self.parameters_classify(
            model)
        self.initial_param()


    def initial_param(self):
        # NOTE: other_param is shared in data parallel, they should be same in all dp ranks.
        #      expert_weight_param and specific_expert_param are not shared in data parallel.

        if self.sub_group is not None and self.sub_group.nranks > 1:
            for param in self.specific_expert_param:
                paddle.distributed.broadcast(param.detach(), src=self.sub_group.ranks[0], group=self.sub_group, use_calc_stream=True)

        for param in self.other_param:
            paddle.distributed.broadcast(param.detach(), src=self.moe_group.ranks[0], group=self.moe_group, use_calc_stream=True)

        if self.dp_group is not None and self.dp_group.nranks > 1:
            for p in self.model.parameters():
                paddle.distributed.broadcast(param.detach(), src=self.dp_group.ranks[0], group=self.dp_group, use_calc_stream=True) 

        print("param initialize!")

    def parameters_classify(self, model):
        print("all params: ", len(model.parameters()))

        common_expert_param = []
        specific_expert_param = []
        expert_weight_param = []
        other_param = []
        batch_norm_nouse = []

        for param in model.parameters():
            if "common_expert" in param.name:
                common_expert_param.append(param)
            elif "specific_expert" in param.name:
                specific_expert_param.append(param)
            elif "exp_weight" in param.name:
                expert_weight_param.append(param)
            elif "batch_norm" in param.name and param.stop_gradient == True:
                batch_norm_nouse.append(param)
            else:
                other_param.append(param)

        print("raw_other_params: ", len(other_param))
        other_param.extend(common_expert_param)

        specific_expert_param.extend(expert_weight_param)

        print("=========param info=========")
        print("other param:", len(other_param))
        print("expert_weight_param: ", len(expert_weight_param))
        print("specific_expert_param: ", len(specific_expert_param))
        print("common_expert_param: ", len(common_expert_param))
        print("============================")
        return other_param, expert_weight_param, specific_expert_param, common_expert_param, batch_norm_nouse

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.clear_grad()
        loss_dict = self.model(data, self.monitor)
        if isinstance(loss_dict, paddle.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        losses.backward()
        # NOTE: shared param should be all_reduced in task MOE.
        all_reduce_parameters(self.other_param, self.moe_group, monitor=self.monitor)
        
        if self.sub_group is None:
            all_reduce_parameters(self.specific_expert_param, self.moe_group, monitor=self.monitor, comm_flag=False)
        else:
            all_reduce_parameters(self.specific_expert_param, self.sub_group, monitor=self.monitor)

        #Note all_redcue in outsize dp.
        all_reduce_parameters(self.model.parameters(), self.dp_group)

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
            self,
            loss_dict,
            data_time,
            prefix="",
    ):
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
            loss_dict,
            data_time,
            prefix="",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        # metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.

        metrics_dict = {}
        for k, v in loss_dict.items():
            v_list = comm.gather_v(v)
            metrics_dict[k] = np.mean([v.detach().cpu().numpy() for v in v_list])
        # metrics_dict["data_time"] = data_time
        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            # data_time = metrics_dict["data_time"]
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            # metrics_dict = {
            #     k: np.mean([x[k].detach().cpu().numpy() for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            # }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    "Loss became infinite or NaN at iteration={storage.iter}!\n"
                    "loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def set_state_dict(self, state_dict):
        super().set_state_dict(state_dict)
        self.optimizer.set_state_dict(state_dict["optimizer"])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None, dp_group=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: 
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        # if isinstance(model,  paddle.DataParallel):
        #     assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        # assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        if grad_scaler is None:
            grad_scaler = paddle.amp.GradScaler(init_loss_scaling=1024.0)
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert NotImplementedError()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def set_state_dict(self, state_dict):
        super().set_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
