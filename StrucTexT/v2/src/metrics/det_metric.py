""" det_metric.py """
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

from collections import namedtuple
from shapely.geometry import Polygon

import os
import json
import numpy as np
import paddle
from pycocotools.coco import COCO

from .map_utils import prune_zero_padding, DetectionMAP
from .coco_utils import get_det_res, get_infer_results, cocoapi_eval

class DetectionIoUEvaluator(object):
    """
    reference from :
    https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
    """
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        """ evaluate_image """
        def get_union(pD, pG):
            """ get_union """
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            """ get_intersection_over_union """
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            """ get_intersection """
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            """ compute_ap """
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            points = [[item[0] + 0.01, item[1] + 0.01] for item in points]
            dontCare = gt[n]['ignore']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
            if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            points = [[item[0] + 0.01, item[1] + 0.01] for item in points]
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
            if len(detDontCarePolsNum) > 0 else "\n")

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }
        return perSampleMetrics

    def combine_results(self, results):
        """ combine_results """
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        methodMetrics = {
            'precision': float('%.4f' % (methodPrecision * 100)),
            'recall': float('%.4f' % (methodRecall * 100)),
            'hmean': float('%.4f' % (methodHmean * 100))
        }

        return methodMetrics


class DetMetric(paddle.metric.Metric):
    """ DetMetric """
    def __init__(self, label_names=[], main_indicator='hmean'):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.label_names = label_names
        self.reset()

    def name(self):
        """ name """
        return self.__class__.__name__

    def update(self, preds, labels, masks=None):
        """
        preds: a `N` list of dict produced by post process
            points: np.ndarray of shape (K, 4, 2), the polygons of objective regions.
            classes (optional): np.ndarray of shape (K, 4, 2), the polygons of objective regions.
        labels: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
        masks: np.ndarray  of shape (N, K), the ignore_tags indicates whether a region is ignorable or not.
        """
        gt_polyons_batch = labels.tolist()
        if masks is None:
            ignore_tags_batch = np.zeros(labels.shape[:2])
        else:
            ignore_tags_batch = masks
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def accumulate(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metircs = self.evaluator.combine_results(self.results)
        self.reset()
        return metircs

    def reset(self):
        """ reset """
        self.results = []  # clear results


class COCOMetric(paddle.metric.Metric):
    """ COCOMetric """
    def __init__(self,
                 anno_file,
                 output_eval='/dev/shm/',
                 overlap_thresh=0.5,
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False,
                 bias=False,
                 main_indicator="mAP"):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.output_eval = output_eval
        self.classwise = classwise
        self.bias = bias

        coco = COCO(anno_file)
        cats = coco.loadCats(coco.getCatIds())
        self.clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
        self.catid2name = {cat['id']: cat['name'] for cat in cats}

        self.iou_type = 'bbox'
        self.reset()

    def name(self):
        """ name """
        return self.__class__.__name__

    def reset(self):
        """ reset """
        # only bbox and mask evaluation support currently
        self.results = {'bbox': []}
        self.eval_results = {}

    def update(self, inputs):
        """ update """
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in inputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        #im_id = inputs['im_id']
        #outs['im_id'] = im_id.numpy() if isinstance(im_id,
        #                                            paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []

    def accumulate(self):
        """ accumulate """
        if len(self.results['bbox']) > 0:
            '''
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
            '''
            bbox_stats = cocoapi_eval(
                self.results['bbox'],
                'bbox',
                anno_file=self.anno_file,
                classwise=self.classwise)
            self.eval_results['bbox'] = bbox_stats

    def get_results(self):
        """ get_results """
        return self.eval_results
