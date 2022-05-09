""" cls_metric.py """
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

class ClsMetric(object):
    """ ClsMetric """
    def __init__(self, main_indicator='acc'):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, labels, masks=None):
        if masks is None:
            masks = np.ones_like(labels)
        masks = masks.astype('int32')
        correct = np.array(preds == labels, dtype='int32')
        correct *= masks
        all = np.ones_like(labels)
        all *= masks
        self.correct_num += np.sum(correct)
        self.all_num += np.sum(all)

    def get_metric(self):
        """
        return metrics
        """
        res = {self.main_indicator: self.correct_num / max(1, self.all_num)}
        return res

    def reset(self):
        """ clear count """
        self.correct_num = 0.0
        self.all_num = 0
