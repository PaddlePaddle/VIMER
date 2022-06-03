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


class OCRPostProcess(object):
    """
    The post process for OCR end-to-end
    """
    def __init__(self, output_dir):
        self.output_path = output_dir
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

    def __call__(self, results):
        img = results['org_image'].numpy()[0]
        img_name = results['image_name'][0]
        e2e_preds = results['e2e_preds'][0]
        ratio = results['ratio'].numpy()[0]
        ratio_h, ratio_w = ratio.tolist()

        ocr_res = split_anno_by_symbol(e2e_preds)
        ocr_str = ''
        pred_label = []
        for bbox, word, prob in ocr_res:
            bbox_np = np.array(bbox).astype(np.double)
            bbox_np[::2] /= ratio_w
            bbox_np[1::2] /= ratio_h
            bbox_np = bbox_np.astype(np.int).tolist()
            img = draw_ocr(img, bbox_np, word)
            bbox_str = map(str, bbox_np)
            bbox_str = ','.join(bbox_str)
            tmp_ocr_str = "{}\t{}\t{}\n".format(bbox_str, word, prob)
            ocr_str += tmp_ocr_str
            if len(word) > 0:
                pred_label.append([bbox, word, prob])

        out_filename_img = os.path.join(self.output_path, img_name)
        cv2.imwrite(out_filename_img, img)
        '''
        out_ocr_filename = out_filename_img + '.txt'
        with codecs.open(out_ocr_filename, 'w', 'utf-8') as fp:
            fp.write(ocr_str)
        '''
        return pred_label
