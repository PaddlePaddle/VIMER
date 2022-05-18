# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn.functional as F
from sklearn import metrics

from utils import comm
from utils.compute_dist import build_dist
from evaluation.evaluator import DatasetEvaluator
from evaluation.query_expansion import aqe
from evaluation.rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluatorSingleTask(DatasetEvaluator):
    """ReidEvaluatorSingleTask
    """
    def __init__(self, cfg, num_query, num_valid_samples, output_dir=None, **kwargs):
        self.cfg = cfg
        self._num_query = num_query
        self._num_valid_samples = num_valid_samples
        self._output_dir = output_dir

        self._cpu_device = paddle.CPUPlace()

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        """reset
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """process
        """
        # remove task dict
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'

        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        prediction = {
            'feats': outputs, #paddle.to_tensor(outputs, place=self._cpu_device, dtype=paddle.float32),
            'pids': inputs['targets'], #paddle.to_tensor(inputs['targets'], place=self._cpu_device, dtype=inputs['targets']),
            'camids': inputs['camids'], #paddle.to_tensor(inputs['camids'], place=self._cpu_device, dtype=inputs['camids']),
        }
        self._predictions.append(prediction)

    def evaluate(self):
        """evaluate
        """
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])

        features = paddle.concat(features, axis=0)
        pids = paddle.concat(pids, axis=0).numpy()
        camids = paddle.concat(camids, axis=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query: self._num_valid_samples]
        gallery_pids = pids[self._num_query: self._num_valid_samples]
        gallery_camids = camids[self._num_query: self._num_valid_samples]

        self._results = OrderedDict()

        if self.cfg.AQE.enabled:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.AQE.QE_time
            qe_k = self.cfg.AQE.QE_k
            alpha = self.cfg.AQE.alpha
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.metric)

        if self.cfg.rerank.enabled:
            logger.info("Test with rerank setting")
            k1 = self.cfg.rerank.k1
            k2 = self.cfg.rerank.k2
            lambda_value = self.cfg.rerank.lambda_

            if self.cfg.metric == "cosine":
                query_features = F.normalize(query_features, axis=1)
                gallery_features = F.normalize(gallery_features, axis=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        from evaluation.rank import evaluate_rank
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.ROC.enabled:
            from evaluation.roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        # comm.synchronize()


# class ReidSaveFeatureTask(DatasetEvaluator):
#     def __init__(self, cfg, num_query, output_dir=None, **kwargs):
#         self.cfg = cfg
#         self._num_query = num_query
#         self._output_dir = output_dir

#         self._cpu_device = paddle.CPUPlace()

#         self._predictions = []
#         self._compile_dependencies()

#     def reset(self):
#         self._predictions = []

#     def process(self, inputs, outputs):
#         # remove task dict
#         assert len(inputs) == 1, 'support only single task evaluation'
#         assert len(outputs) == 1, 'support only single task evaluation'

#         inputs = list(inputs.values())[0]
#         outputs = list(outputs.values())[0]

#         features = paddle.to_tensor(outputs, place=self._cpu_device, dtype=paddle.float32)
#         features = F.normalize(features, p=2, axis=1) # normalize
#         prediction = {
#             'feats': features,
#         }
#         self._predictions.append(prediction)

#     def evaluate(self):
#         if comm.get_world_size() > 1:
#             comm.synchronize()
#             predictions = comm.gather(self._predictions, dst=0)
#             predictions = list(itertools.chain(*predictions))

#             if not comm.is_main_process():
#                 return {}

#         else:
#             predictions = self._predictions

#         def incremental_write_to_hdf5(f, data):
#             for key in data.keys():
#                 value = data[key]
#                 dataset = f.get(key)
#                 if not dataset:
#                     maxshape = (None,) + value.shape[1:]
#                     f.create_dataset(key, data=value, compression="gzip", chunks=True, maxshape=maxshape)
#                 else:
#                     dataset.resize((dataset.shape[0] + value.shape[0]), axis=0)
#                     dataset[-value.shape[0]:] = value

#         import h5py
#         fout = h5py.File('feature_out.hdf5', 'a')
#         for prediction in predictions:
#             incremental_write_to_hdf5(fout, {'feature': prediction['feats'].numpy()})
#         fout.close()

#         return {}

#     def _compile_dependencies(self):
#         # Since we only evaluate results in rank(0), so we just need to compile
#         # cython evaluation tool on rank(0)
#         if comm.is_main_process():
#             try:
#                 from .rank_cylib.rank_cy import evaluate_cy
#             except ImportError:
#                 start_time = time.time()
#                 logger.info("> compiling reid evaluation cython tool")

#                 compile_helper()

#                 logger.info(
#                     ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
#                     "seconds".format(time.time() - start_time))
#         comm.synchronize()
