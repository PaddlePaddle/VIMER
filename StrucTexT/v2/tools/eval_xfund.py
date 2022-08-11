"""
Function: evaluation for chinsese dataset
Date: 2022-06-28
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from src.metrics.ocr_metric import *

def create_pred_dict(pred_filenames):
    """
    Create dictionary for prediction.
    """
    dic = {}
    pred_count = 0

    for pred_filename in pred_filenames:
        with open(pred_filename, 'r') as f1:
            polys = []
            for line in f1:
                line_split = line.strip().split('\t')
                text = line_split[0]
                text_quad = line_split[1].split(' ')
                text_quad = [int(x) for x in text_quad]
                line_label = line_split[2].lower()
                polys.append((text_quad, text, line_label))
                pred_count += 1
            dic[os.path.basename(pred_filename).replace('.txt', '')] = polys
    return dic, pred_count


def create_gt_dict(filename):
    """
    Create dictionary for ground-truth.
    Input:
        filenames:
    Output:
        dic: keys are img_names, values are (word_bb, txt_tags, txts)
        gt_count: the number of valid gt
    """
    dic = {}
    id2name = {}
    gt_count = 0
    cls_dict = {'other': 0, 'header': 1, 'question': 2, 'answer': 3}

    with open(filename, 'r') as fp:
        for line in fp:
            polys = []
            line_split = line.strip().split('\t')
            file_name = line_split[0]
            anno = json.loads(line_split[1])

            ocr_info = anno['ocr_info']
            for o_info in ocr_info:
                text = o_info['text']
                bbox = o_info['bbox']
                poly = [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
                gt_count += 1
                label = o_info['label']
                polys.append((poly, label, text))
            dic[file_name] = polys
    return dic, gt_count


def eval(pred_folder, gt_res, iou_thresh=0.5):
    """
    Evaluate the performance of polygon detection.
    """
    # Create dictionary for prediction
    pred_filenames = sorted(glob.glob('{}/*.txt'.format(pred_folder)))
    pred_dic, pred_count = create_pred_dict(pred_filenames)
    gt_dic, gt_count = create_gt_dict(gt_res)

    # Evaluation
    fp_count = 0
    hit_count = 0
    seqerr = 0
    hit_str_count = 0
    ned_count = 0

    for filename, gt in gt_dic.items():
        if not filename in pred_dic.keys():
            print("not found {} in pred result".format(filename))
            continue
        pred = pred_dic[filename]
        eval_result = eval_one_file(pred, gt, iou_thresh)
        hit_count += eval_result["hit_count"]
        hit_str_count += eval_result["hit_str_num"]
        fp_count += eval_result["fp_count"]
        ned_count += eval_result["ned_count"]

    recall = float(hit_count) / (gt_count + 1e-5)
    precision = float(hit_count) / (pred_count + 1e-5)
    f_score = recall * precision * 2 / (recall + precision + 1e-5)
    seqerr = 1 - float(hit_str_count) / (hit_count + 1e-5)
    recall_e2e = float(hit_str_count) / (gt_count + 1e-5)
    precision_e2e = float(hit_str_count) / (pred_count + 1e-5)
    f_score_e2e = recall_e2e * precision_e2e * \
        2 / (1e-5 + recall_e2e + precision_e2e)

    ned_e2e = float(ned_count / (fp_count + gt_count + 1e-5))
    print("gt_count: %d pred_count: %d hit_count(iou): %d hit_count(iou && transcript): %d "
          % (gt_count, pred_count, hit_count, hit_str_count))
    print("1-NED: %.4f" % (ned_e2e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = argparse.ArgumentParser('launch for xfund eval')

    parser.add_argument('--pred_folder', type=str, required=True)
    parser.add_argument('--gt_file', type=str, required=True)

    args = parser.parse_args()

    eval(args.pred_folder, args.gt_file)