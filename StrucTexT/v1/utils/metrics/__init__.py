""" metric """
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ['build_metric']


def build_metric(config):
    """ build_metric """
    from .f1_metric import F1Metric
    from .map_metric import MapMetric
    from .hit_metric import HitMetric
    from .mrank_metric import MrankMetric
    from .tabcls_metric import TabClsMetric
    from .cls_metric import ClsMetric
    from .rescore_metric import ReScoreMetric
    support_dict = ['TabClsMetric', 'F1Metric', 'MapMetric', 'HitMetric', 'MrankMetric', 'ClsMetric', 'ReScoreMetric']

    module_classes = {}
    config = copy.deepcopy(config)
    for item in config:
        name = item.pop('name')
        module_name = item.pop('type')
        assert module_name in support_dict, Exception(
                'metric only support {}'.format(support_dict))
        module_classes[name] = eval(module_name)(**item)
    return module_classes
