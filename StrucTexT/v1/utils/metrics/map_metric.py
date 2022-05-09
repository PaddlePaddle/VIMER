""" map_metric.py """
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

import numpy as np

class MapMetric(object):
    """ MapMetric """
    def __init__(self, main_indicator='mAP'):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, labels, masks=None):
        if masks is None:
            masks = np.ones_like(labels)
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            for pred, label, mask in zip(pred_b, label_b, mask_b):
                if np.sum(mask.astype('int32')) == 0 or np.sum(label) == 0:
                    continue
                pid = np.argsort(pred[mask])[::-1]
                l = label[mask][pid]
                pos = np.where(l == 1)[0] + 1
                ap = [np.sum(l[:i]) * 1.0 / i for i in pos]
                self.map += np.mean(ap)
                self.all_num += 1

    def get_metric(self):
        """
        return metrics
        """
        res = {self.main_indicator: self.map / max(1, self.all_num)}
        return res

    def reset(self):
        """ clear count """
        self.map = 0.0
        self.all_num = 0
