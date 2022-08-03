""" demo.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import cv2
import glob
import json
import codecs
import shutil
import random
import logging
import argparse
import functools
import importlib
import numpy as np
import paddle as P
from Polygon import Polygon

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../src')))

def preprocess(im, input_size=1024):
    """preprocess image"""
    h, w, _ = im.shape
    resize_w = w
    resize_h = h

    if resize_h > resize_w:
        ratio = float(input_size) / resize_h
    else:
        ratio = float(input_size) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    # normalize img
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    im = im / 255
    im -= img_mean
    im /= img_std
    im = im.transpose((2, 0, 1)).astype(np.float32)
    im = im[np.newaxis, :]

    return im, ratio_h, ratio_w


def draw_ocr(draw_img, bbox, text):
    """draw ocr"""
    bbox_color = (255, 0, 255)
    text_color = (255, 255, 0)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

    bbox_np = np.array(bbox).reshape(-1, 2)
    cv2.polylines(draw_img, [bbox_np], True, bbox_color, 2)

    if bbox_np[0][1] - text_size[1] - 3 < 0:
        cv2.rectangle(draw_img,
                    (bbox_np[0][0], bbox_np[0][1] + 2),
                    (bbox_np[0][0] + text_size[0], bbox_np[0][1] + text_size[1] + 3),
                    color=text_color,
                    thickness=-1)
        cv2.putText(draw_img, text,
                    (bbox_np[0][0], bbox_np[0][1] + text_size[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1)
    else:
        cv2.rectangle(draw_img,
                    (bbox_np[0][0], bbox_np[0][1] - text_size[1] - 3),
                    (bbox_np[0][0] + text_size[0], bbox_np[0][1] - 3),
                    color=text_color,
                    thickness=-1)
        cv2.putText(draw_img, text,
                    (bbox_np[0][0], bbox_np[0][1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1)
    return draw_img


def split_instance(bbox, text, symbol):
    """split bbox and
    """
    begin = text.index(symbol) + 1
    ratio = begin / len(text)
    text1 = text[:begin]
    text2 = text[begin:]

    pt1_x = int((bbox[2] - bbox[0]) * ratio + bbox[0])
    pt1_y = int((bbox[3] - bbox[1]) * ratio + bbox[1])

    pt2_x = int((bbox[4] - bbox[6]) * ratio + bbox[6])
    pt2_y = int((bbox[5] - bbox[7]) * ratio + bbox[5])

    bbox1 = [bbox[0], bbox[1], pt1_x, pt1_y, pt2_x, pt2_y, bbox[6], bbox[7]]
    bbox2 = [pt1_x, pt1_y, bbox[2], bbox[3], bbox[4], bbox[5], pt2_x, pt2_y]

    return {"bbox1": bbox1,
            "text1": text1,
            "bbox2": bbox2,
            "text2": text2}


def split_anno_by_symbol(anno):
    """some bbox annotations in funsd datset are confusing, we use specific postprocess to handle this problme
    Args:
        anno: list of Tuple (bbox, text, socre)
    Return:
        res: same type as anno
    """
    res = []
    for bbox, text, score in anno:
        cnt1 = text.count("-")
        cnt2 = text.count(".")
        cnt3 = text.count("/")
        if cnt1 == 1 and len(text) > 6 and text[-1] !=  "-":
            # 450-3785 -> 450- 3785
            res_dict = split_instance(bbox, text, "-")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        elif cnt1 == 2 and text[-1] == "-":
            # 212-450- -> 212- 450-
            res_dict = split_instance(bbox, text, "-")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        elif cnt1 == 2 and text[-1] != "-":
            # B43-720-9290 -> B43- 720- 9290
            res_dict = split_instance(bbox, text, "-")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res_dict = split_instance(res_dict["bbox2"], res_dict["text2"], "-")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        elif cnt2 == 2 and len(text) == 4:
            # D.C. -> D. C.
            res_dict = split_instance(bbox, text, ".")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        elif cnt3 == 1 and text[-1] != "/" and text.replace("/", "").isdigit():
            # 21/80 21/ 80
            res_dict = split_instance(bbox, text, "/")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        elif cnt3 == 2 and len(text) == 8:
            # 10/26/92 -> 10/ 26/ 92
            res_dict = split_instance(bbox, text, "/")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res_dict = split_instance(res_dict["bbox2"], res_dict["text2"], "/")
            res.append((res_dict["bbox1"], res_dict["text1"], score))
            res.append((res_dict["bbox2"], res_dict["text2"], score))
        else:
            res.append((bbox, text, score))

    return res


def nms_min(S, thres=0.2):
    """ nms_min """
    if len(S) == 0:
        return np.array([])
    order = np.argsort([Polygon(x.reshape((4, 2))).area() for x in S])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection_min(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]
    return S[keep], keep


def intersection_min(g, p):
    """
    Intersection_iog.
    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    inter = g.area() + p.area() - (g + p).area()
    union = min(p.area(), g.area())
    if union == 0:
        print("p_area is very small")
        return 0
    else:
        return inter / union


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def distance(box_1, box_2):
    """ distance """
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for k in range(10):
        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
    return _boxes


def match_result(dt_boxes, pred_bboxes):
    """ match_result """
    matched = {}
    for i, gt_box in enumerate(dt_boxes):
        distances = []
        for j, pred_box in enumerate(pred_bboxes):
            distances.append((distance(gt_box, pred_box),
                1. - compute_iou(gt_box, pred_box)
                ))  # 获取两两cell之间的L1距离和 1- IOU
        sorted_distances = distances.copy()
        # 根据距离和IOU挑选最"近"的cell
        sorted_distances = sorted(
            sorted_distances, key=lambda item: (item[1], item[0]))
        if distances.index(sorted_distances[0]) not in matched.keys():
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)
    return matched


def grouping(word_res, line_res):
    """ grouping """
    # print(line_res)
    num_boxes = len(word_res)
    sorted_boxes = sorted(word_res, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    for k in range(10):
        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 \
                    and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
    word_text = np.array([x[1] for x in _boxes])
    word_bbox = np.array([x[0] for x in _boxes]).reshape((-1, 4, 2))
    word_bbox = [word_bbox[:, :, 0].min(axis=1), word_bbox[:, :, 1].min(axis=1),
                 word_bbox[:, :, 0].max(axis=1), word_bbox[:, :, 1].max(axis=1)]
    word_bbox = np.stack(word_bbox, axis=1)
    #word_bbox = np.array([[int(box[0]), int(box[1]), int(box[4]), int(box[5])] for box in word_bbox])

    det_result = np.array([x[0] for x in line_res])
    class_result = np.array([x[1] for x in line_res])
    pred_bboxes, keep = nms_min(det_result, 0.5)
    line_cls = np.array(class_result)[keep].tolist()
    #line_bbox = [[int(box[0]), int(box[1]), int(box[4]), int(box[5])] for box in pred_bboxes]
    #line_bbox = np.array(line_bbox).reshape(-1, 4)
    line_bbox = pred_bboxes.reshape((-1, 4, 2))
    line_bbox = [line_bbox[:, :, 0].min(axis=1), line_bbox[:, :, 1].min(axis=1),
                 line_bbox[:, :, 0].max(axis=1), line_bbox[:, :, 1].max(axis=1)]
    line_bbox = np.stack(line_bbox, axis=1)

    matched_index = match_result(word_bbox, line_bbox)
    grouping_res = []
    for i, bbox in enumerate(line_bbox):
        if i in matched_index.keys():
            word_list = matched_index[i]
            line_str_list = []
            for word_idx in word_list:
                line_str_list.append(word_text[word_idx])
            line_str = ' '.join(line_str_list)
            poly = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
            grouping_res.append((poly, line_str, line_cls[i]))
    return grouping_res
