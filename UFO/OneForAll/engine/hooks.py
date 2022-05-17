"""paddle authors
"""
import logging
from collections import Counter

from detectron2.engine.train_loop import HookBase


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scale = 0

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        # largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        # if largest_group == 1:
        #     # If all groups have one parameter,
        #     # then find the most common initial LR, and use it for summary
        #     lr_count = Counter([g["lr"] for g in optimizer.param_groups])
        #     lr = lr_count.most_common()[0][0]
        #     for i, g in enumerate(optimizer.param_groups):
        #         if g["lr"] == lr:
        #             self._best_param_group_id = i
        #             break
        # else:
        #     for i, g in enumerate(optimizer.param_groups):
        #         if len(g["params"]) == largest_group:
        #             self._best_param_group_id = i
        #             break

    def before_step(self):
        """before_step
        """
        if self.trainer.grad_scaler is not None:
            self._scale = self.trainer.grad_scaler._scale

    def after_step(self):
        """after_step
        """
        lr = self._optimizer.get_lr()
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        if self.trainer.grad_scaler is None or self._scale == self.trainer.grad_scaler._scale:
            self._scheduler.step()
            lr = self._optimizer.get_lr()

        next_iter = self.trainer.iter + 1

    def state_dict(self):
        """state_dict
        """
        # state_dict = {}
        # for key, value in self._scheduler.items():
        #     if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
        #         state_dict[key] = value.state_dict()
        # return state_dict
        return self._scheduler.state_dict()

    def set_state_dict(self, state_dict):
        """set_state_dict
        """
        # print(state_dict)
        # for key, value in state_dict.items():
        #     logger = logging.getLogger(__name__)
        #     logger.info("Loading scheduler from state_dict ...")
        #     self._scheduler[key].load_state_dict(value)
        self._scheduler.set_state_dict(state_dict)
