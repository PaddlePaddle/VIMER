"""utils/events.py
"""
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
from fvcore.common.history_buffer import HistoryBuffer

from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.file_io import PathManager


class CommonMetricSacredWriter(EventWriter):
    """CommonMetricSacredWriter
    """
    def __init__(self, _run, max_iter=None, window_size=20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._run = _run
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA

    def _get_eta(self, storage):
        if self._max_iter is None:
            return None
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration - 1)
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return eta_seconds / 60. / 60.  # hours
        except KeyError:
            # estimate eta on our own - more noisy
            eta = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                        iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta = eta_seconds / 60. / 60.  # hours
            self._last_write = (iteration, time.perf_counter())
            return eta

    def write(self):
        """write
        """
        storage = get_event_storage()
        iteration = storage.iter

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        if data_time is not None:
            self._run.log_scalar('data_time', data_time, iteration)

        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        if iter_time is not None:
            self._run.log_scalar('iter_time', iter_time, iteration)

        try:
            lr = storage.history("lr").latest()
        except KeyError:
            lr = None
        if lr is not None:
            self._run.log_scalar('lr', lr, iteration)

        eta = self._get_eta(storage)
        if eta is not None:
            self._run.log_scalar('eta', eta, iteration)

        max_mem_mb = None
        if eta is not None:
            self._run.log_scalar('max_mem_mb', max_mem_mb, iteration)

        for k, v in storage.histories().items():
            if "loss" in k:
                self._run.log_scalar(k, v.median(self._window_size), iteration)
