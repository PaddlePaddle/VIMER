""" coco_utils """
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import itertools

from .map_utils import draw_pr_curve

def get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    """ get_det_res """
    det_res = []
    k = 0
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(image_id, np.ndarray):
        image_id = image_id.tolist()
    if isinstance(bbox_nums, np.ndarray):
        bbox_nums = bbox_nums.tolist()
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin + bias
            h = ymax - ymin + bias
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_det_poly_res(bboxes, bbox_nums, image_id, label_to_cat_id_map, bias=0):
    """ get_det_poly_res """
    det_res = []
    k = 0
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    if isinstance(image_id, np.ndarray):
        image_id = image_id.tolist()
    if isinstance(bbox_nums, np.ndarray):
        bbox_nums = bbox_nums.tolist()
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, x1, y1, x2, y2, x3, y3, x4, y4 = dt
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': rbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    return infer_res


def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    assert coco_gt is not None or anno_file is not None
    if style == 'keypoints_crowd':
        #please install xtcocotools==1.6
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    else:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    logging.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    elif style == 'keypoints_crowd':
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            logging.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logging.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logging.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


def json_eval_results(metric, json_directory, dataset):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logging.info("{} not exists!".format(v_json))
