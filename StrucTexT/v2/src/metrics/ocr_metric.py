"""
Function: evaluation for chinsese dataset
Date: 2021-12-28
"""

import Polygon
import sys
import os
import numpy as np
import math
import glob
import codecs
import json
import editdistance
import math
import paddle


class PredPoly(object):
    """
    pass
    """
    def __init__(self, xs, ys):
        self.poly = Polygon.Polygon(tuple(zip(xs, ys)))


class GtPoly(object):
    """
    pass
    """
    def __init__(self, xs, ys, is_ignore=False):
        self.poly = Polygon.Polygon(tuple(zip(xs, ys)))
        self.ignore = is_ignore


def iou(poly1, poly2):
    """
    Compute the iou between poly1 and poly2.
    """
    intersec = poly1.area() + poly2.area() - (poly1 + poly2).area()
    union = (poly1 + poly2).area()
    return intersec / max(1e-5, union)


def pts2PolyPred(poly):
    """
    Convert points to PolyPred.
    """
    xs = poly[0::2]
    ys = poly[1::2]
    pred_poly = PredPoly(xs, ys)
    return pred_poly


def pts2PolyGt(poly):
    """
    Convert points to PolyGt.
    """
    xs = poly[0::2]
    ys = poly[1::2]
    gt_poly = GtPoly(xs, ys, False)
    return gt_poly


def eval_one_file(pred, gt, iou_thresh):
    """
    Evaluate the performance of polygon detection for one file (image).
    """
    hit_count = 0
    hit_str_num = 0
    fp_count = 0
    ned_count = 0
    gt_match_record = np.zeros(len(gt))

    pred = sorted(pred, key=lambda x: x[2], reverse=True)

    for pred_pts in pred:
        pred_poly = pts2PolyPred(pred_pts[0])
        pred_str = pred_pts[1].lower()
        pred_label = pred_pts[2]
        max_iou = 0
        max_iou_idx = -1
        # Compute the iou between gt_poly and pred_poly
        for idx in range(len(gt)):
            gt_poly = pts2PolyGt(gt[idx][0])
            if gt_match_record[idx] == 1:
                continue
            cur_iou = iou(gt_poly.poly, pred_poly.poly)
            if cur_iou > max_iou:
                max_iou = cur_iou
                max_iou_idx = idx

        if max_iou > iou_thresh:
            hit_count += 1
            gt_match_record[max_iou_idx] = 1
            gt_label = gt[max_iou_idx][1]
            gt_str = gt[max_iou_idx][2].lower()
            if gt_label == -1 or gt_label == pred_label:
                if gt_str == pred_str:
                    hit_str_num += 1
                ned = 1 - editdistance.eval(pred_str, gt_str) / max(1, max(len(gt_str), len(pred_str)))
                ned_count += ned
        else:
            fp_count += 1

    return {"hit_count": hit_count,
            "hit_str_num": hit_str_num,
            "fp_count": fp_count,
            "ned_count": ned_count}


class OCRMetric(paddle.metric.Metric):
    """ OCRMetric """
    def __init__(self, iou_thresh=0.5, main_indicator='1-NED'):
        self.main_indicator = main_indicator
        self.iou_thresh = iou_thresh
        self.reset()

    def name(self):
        """ name """
        return self.__class__.__name__

    def update(self, preds, labels):
        eval_result = eval_one_file(preds, labels, self.iou_thresh)
        self.gt_count += len(labels)
        self.fp_count += eval_result["fp_count"]
        self.ned_count += eval_result["ned_count"]

    def accumulate(self):
        """
        return metrics
        """
        ned_e2e = float(self.ned_count / max(self.fp_count + self.gt_count, 1))
        res = {self.main_indicator: ned_e2e}
        return res

    def reset(self):
        """ clear count """
        self.gt_count = 0
        self.fp_count = 0
        self.ned_count = 0
