""" table_metric.py """
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
from .teds_utils import TEDS

class TableMetric(paddle.metric.Metric):
    """ ClsMetric """
    def __init__(self, main_indicator='teds', structure_only=True):
        self.structure_only = structure_only
        self.main_indicator = main_indicator
        self.reset()
        self.teds = TEDS(n_jobs=16, structure_only = True)

    def name(self):
        """ name """
        return self.__class__.__name__

    def update(self, pred_html, gt_html, masks=None):
        """ update """
        html_pred = ''.join(pred_html)
        html_gt = ''.join(gt_html)
        html_gt = '<html><body><table>' + html_gt + '</table></body></html>'
        html_pred = '<html><body><table>' + html_pred + '</table></body></html>'
        teds_item = self.teds.evaluate(html_pred, html_gt)
        self.teds_list.append(teds_item)
        self.all_num += 1

    def accumulate(self):
        """
        return metrics
        """
        res = {self.main_indicator: sum(self.teds_list) / max(1, len(self.teds_list))}
        return res

    def reset(self):
        """ clear count """
        self.teds_list = []
        self.all_num = 0
