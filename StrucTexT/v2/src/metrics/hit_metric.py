""" hit_metric.py """
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np

class HitMetric(paddle.metric.Metric):
    """ HitMetric """
    def __init__(self, k=1, main_indicator='Hit'):
        self.k = k
        self.main_indicator = main_indicator + '@{}'.format(k)
        self.reset()

    def name(self):
        """ name """
        return self.__class__.__name__

    def update(self, preds, labels, masks=None):
        """ update """
        if masks is None:
            masks = np.ones_like(labels)
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            for pred, label, mask in zip(pred_b, label_b, mask_b):
                if np.sum(mask.astype('int32')) == 0 or np.sum(label) == 0:
                    continue
                mask = np.where(mask)
                pred = pred[mask]
                label = label[mask]
                pid = np.argsort(pred)[::-1]
                l = label[pid]
                self.all_num += 1
                if l[:self.k].sum() > 0:
                    self.hit += 1

    def accumulate(self):
        """
        return metrics
        """
        res = {self.main_indicator: self.hit / max(1, self.all_num)}
        return res

    def reset(self):
        """ clear count """
        self.hit = 0.0
        self.all_num = 0
