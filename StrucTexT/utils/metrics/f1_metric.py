""" f1_metric.py """
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

class F1Metric(object):
    """ F1Metric """
    def __init__(self, threshold, main_indicator='F1'):
        self.main_indicator = main_indicator
        self.threshold = threshold
        self.reset()

    def __call__(self, preds, labels, masks=None):
        if masks is None:
            masks = np.ones_like(labels)
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            for pred, label, mask in zip(pred_b, label_b, mask_b):
                if np.sum(mask.astype('int32')) == 0:
                    continue
                pos_p = np.array(pred[mask] > self.threshold, dtype='int32')
                pos_l = label[mask]
                self.acc[0] += np.sum(pos_p * pos_l) # true positive
                self.acc[1] += np.sum((1 - pos_p) * pos_l) # false positive
                self.acc[2] += np.sum(pos_p * (1 - pos_l)) # false negative
                self.acc[3] += np.sum((1 - pos_p) * (1 - pos_l)) # true negative

    def get_metric(self):
        """
        return metrics
        """
        res = {}
        p = self.acc[0] / max(1e-6, self.acc[0] + self.acc[2])
        r = self.acc[0] / max(1e-6, self.acc[0] + self.acc[1])
        f = 2 * r * p / max(1e-6, r + p)
        res['EP'] = p
        res['ER'] = r
        res['EF'] = f
        return res

    def reset(self):
        """ clear count """
        self.acc = np.zeros(4)
