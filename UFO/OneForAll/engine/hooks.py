"""paddle authors
"""
import logging
from collections import Counter

from detectron2.engine.train_loop import HookBase


class LRScheduler(HookBase):
    """
    A hook which executes a builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (optim.Optimizer):
            scheduler (optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scale = 0

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
        return self._scheduler.state_dict()

    def set_state_dict(self, state_dict):
        """set_state_dict
        """
        self._scheduler.set_state_dict(state_dict)
