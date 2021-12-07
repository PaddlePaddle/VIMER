""" tabcls_metric.py """
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

import cv2
import tabulate
import numpy as np

class TabClsMetric(object):
    """ TabClsMetric """
    def __init__(self, label_names, ignore_idx=[], main_indicator=''):
        self.main_indicator = main_indicator
        self.ignore_idx = ignore_idx
        self.label_names = label_names
        self.reset()

    def __call__(self, preds, labels, masks=None):
        if self.main_indicator == 'entity':
            assert (masks is not None), 'The entity calculation requires mask'
            self._entity(preds, labels, masks)
        elif self.main_indicator == 'token':
            assert (masks is not None), 'The token calculation requires mask'
            self._token(preds, labels, masks)
        else:
            self.main_indicator = 'line'
            self._segment(preds, labels)

    def _entity(self, preds, labels, masks):
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            mask_b = np.where(mask_b > 0)
            pred_b = pred_b[mask_b]
            label_b = label_b[mask_b]
            for idx, name in enumerate(self.label_names):
                pred = np.array(pred_b == idx, dtype='uint8')
                label = np.array(label_b == idx, dtype='uint8')
                self.acc2[idx, 1] += 1
                self.acc2[idx, 2] += 1
                if np.all(pred == label):
                    self.acc2[idx, 0] += 1

    def _segment(self, preds, labels):
        for idx, name in enumerate(self.label_names):
            pos_p = np.array(preds == idx, dtype='int32')
            pos_l = np.array(labels == idx, dtype='int32')
            self.acc[idx, 0] += np.sum(pos_p * pos_l)
            self.acc[idx, 1] += np.sum(pos_p)
            self.acc[idx, 2] += np.sum(pos_l)

    def _token(self, preds, labels, masks):
        for pred_b, label_b, mask_b in zip(preds, labels, masks):
            mask_b = np.where(mask_b > 0)
            pred_b = pred_b[mask_b]
            label_b = label_b[mask_b]
            for idx, name in enumerate(self.label_names):
                pred = np.array(pred_b == idx, dtype='uint8')
                label = np.array(label_b == idx, dtype='uint8')
                num_pred, idx_pred, stats_pred, _ = cv2.connectedComponentsWithStats(pred)
                num_label, idx_label, stats_label, _ = cv2.connectedComponentsWithStats(label)
                self.acc2[idx, 1] += num_pred - 1
                self.acc2[idx, 2] += num_label - 1
                for i, x in enumerate(stats_pred[1:]):
                    for j, y in enumerate(stats_label[1:]):
                        if x[0] == y[0] and x[1] == y[1] and \
                           x[2] == y[2] and x[3] == y[3]:
                               self.acc2[idx, 0] += 1

    def get_metric(self):
        """
        return metrics
        """

        data_list = [['name', 'mEP', 'mER', 'mEF', 'label_num']]
        if self.main_indicator in ['entity', 'token']:
            acc = self.acc2
        else:
            acc = self.acc
        for idx, (name, item_acc) in enumerate(zip(self.label_names, acc)):
            if idx in self.ignore_idx:
                continue
            n = int(item_acc[-1])
            p = item_acc[0] / max(1e-6, item_acc[1])
            r = item_acc[0] / max(1e-6, item_acc[2])
            f = 2 * r * p / max(1e-6, r + p)
            data_list.append([name, p, r, f, n])
        p = np.mean([x[1] for x in data_list[1:] if x[-1] > 0])
        r = np.mean([x[2] for x in data_list[1:] if x[-1] > 0])
        f = np.mean([x[3] for x in data_list[1:] if x[-1] > 0])
        data_list.append([self.main_indicator + '-level Macro F1', p, r, f, ''])

        acc = [x for i, x in enumerate(acc) if i not in self.ignore_idx]
        p = np.sum([x[0] for x in acc]) / np.sum([x[1] for x in acc])
        r = np.sum([x[0] for x in acc]) / np.sum([x[2] for x in acc])
        f = 2 * r * p / max(1e-6, r + p)
        data_list.append([self.main_indicator + '-level Micro F1', p, r, f, ''])

        form = tabulate.tabulate(data_list, tablefmt='grid', headers='firstrow')
        return form

    def reset(self):
        """ clear count """
        self.acc = np.zeros((len(self.label_names), 3))
        self.acc2 = np.zeros((len(self.label_names), 3))
