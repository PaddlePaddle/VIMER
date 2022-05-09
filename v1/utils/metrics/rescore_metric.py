""" rescore_metric.py """
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
import numpy as np


class ReScoreMetric(object):
    """ ReScoreMetric """
    def __init__(self, mode="boundaries", threshold=0.5, main_indicator=''):
        self.main_indicator = main_indicator
        self.mode = mode
        self.threshold = threshold
        self.reset()

    def __call__(self, preds, labels, masks=None):

        pred_relations = []
        gt_relations = []
        for pred_b, label_b in zip(preds, labels):
            pred_relation= []
            gt_relation = []

            rows, cols = label_b.shape
            for row in range(rows):
                for col in range(cols):
                    if label_b[row, col] == 1:
                        rel = {"head": (row, row+1),
                               "tail": (col, col+1),
                               "head_type": -1, #NOT CARE
                               "tail_type": -1, #NOT CARE
                               "type": 1}
                        gt_relation.append(rel)

            pred_b = np.array(pred_b > self.threshold, dtype='int32')

            rows, cols = pred_b.shape
            for row in range(rows):
                for col in range(cols):
                    if pred_b[row, col] == 1:
                        rel = {"head": (row, row+1),
                                "tail": (col, col+1),
                                "head_type": -1, #NOT CARE
                                "tail_type": -1, #NOT CARE
                                "type": 1}
                        pred_relation.append(rel)

            pred_relations.append(pred_relation)
            gt_relations.append(gt_relation)

        score = self.re_score(pred_relations, gt_relations, mode=self.mode)

        self.acc[0] += score['ALL']['tp']
        self.acc[1] += score['ALL']['fp']
        self.acc[2] += score['ALL']['fn']
        self.acc[3] += score['ALL']['p']
        self.acc[4] += score['ALL']['r']
        self.acc[5] += score['ALL']['f1']
        self.acc[6] += score['ALL']['Macro_f1']
        self.acc[7] += score['ALL']['Macro_p']
        self.acc[8] += score['ALL']['Macro_r']

        if score['ALL']['tp'] + score['ALL']['fp'] + score['ALL']['fn'] != 0:
            self.all_num += 1

    def get_metric(self):
        """
        return metrics
        """
        res = {}
        res['TP_NUM'] = self.acc[0]
        res['FP_NUM'] = self.acc[1]
        res['FN_NUM'] = self.acc[2]
        res['PRECISION'] = self.acc[3] / self.all_num
        res['RECALL'] = self.acc[4] / self.all_num
        res['F1-SCORE'] = self.acc[5] / self.all_num
        res['Macro_f1'] = self.acc[6] / self.all_num
        res['Macro_p'] = self.acc[7] / self.all_num
        res['Macro_r'] = self.acc[8] / self.all_num
        return res

    def reset(self):
        """ clear count """
        self.acc = np.zeros(9)
        self.all_num = 0

    def re_score(self, pred_relations, gt_relations, mode="strict"):
        """Evaluate RE predictions
        Args:
            pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
            gt_relations (list) :    list of list of ground truth relations
                rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                        "tail": (start_idx (inclusive), end_idx (exclusive)),
                        "head_type": ent_type,
                        "tail_type": ent_type,
                        "type": rel_type}
            vocab (Vocab) :         dataset vocabulary
            mode (str) :            in 'strict' or 'boundaries'"""

        assert mode in ["strict", "boundaries"]

        relation_types = [v for v in [0, 1] if not v == 0]
        scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

        # Count GT relations and Predicted relations
        n_sents = len(gt_relations)
        n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
        n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

        # Count TP, FP and FN per type
        for pred_sent, gt_sent in zip(pred_relations, gt_relations):
            for rel_type in relation_types:
                # strict mode takes argument types into account
                if mode == "strict":
                    pred_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in pred_sent
                        if rel["type"] == rel_type
                    }
                    gt_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in gt_sent
                        if rel["type"] == rel_type
                    }

                # boundaries mode only takes argument spans into account
                elif mode == "boundaries":
                    pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                    gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

                scores[rel_type]["tp"] += len(pred_rels & gt_rels)
                scores[rel_type]["fp"] += len(pred_rels - gt_rels)
                scores[rel_type]["fn"] += len(gt_rels - pred_rels)

        # Compute per entity Precision / Recall / F1
        for rel_type in scores.keys():
            if scores[rel_type]["tp"]:
                scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
                scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
            else:
                scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

            if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
                scores[rel_type]["f1"] = (
                    2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
                )
            else:
                scores[rel_type]["f1"] = 0

        # Compute micro F1 Scores
        tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
        fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
        fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

        else:
            precision, recall, f1 = 0, 0, 0

        scores["ALL"]["p"] = precision
        scores["ALL"]["r"] = recall
        scores["ALL"]["f1"] = f1
        scores["ALL"]["tp"] = tp
        scores["ALL"]["fp"] = fp
        scores["ALL"]["fn"] = fn

        # Compute Macro F1 Scores
        scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
        scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
        scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

        # print(f"RE Evaluation in *** {mode.upper()} *** mode")

        # print(
        #     "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(
        #         n_sents, n_rels, n_found, tp
        #     )
        # )
        # print(
        #     "\tALL\t TP: {};\tFP: {};\tFN: {}".format(scores["ALL"]["tp"], scores["ALL"]["fp"], scores["ALL"]["fn"])
        # )
        # print("\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(precision, recall, f1))
        # print(
        #     "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
        #         scores["ALL"]["Macro_p"], scores["ALL"]["Macro_r"], scores["ALL"]["Macro_f1"]
        #     )
        # )

        # for rel_type in relation_types:
        #     print(
        #         "\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
        #             rel_type,
        #             scores[rel_type]["tp"],
        #             scores[rel_type]["fp"],
        #             scores[rel_type]["fn"],
        #             scores[rel_type]["p"],
        #             scores[rel_type]["r"],
        #             scores[rel_type]["f1"],
        #             scores[rel_type]["tp"] + scores[rel_type]["fp"],
        #         )
        #     )

        return scores

