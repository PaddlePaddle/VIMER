# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
from collections import OrderedDict
from typing import List, Optional, Dict

import faiss
import numpy as np
import paddle
import paddle.nn.functional as F

from evaluation.evaluator import DatasetEvaluator
from utils import comm

logger = logging.getLogger("ufo.retri_evaluator")


@paddle.no_grad()
def recall_at_ks(query_features, query_labels, ks, gallery_features=None, gallery_labels=None, cosine=False):
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.
    Parameters
    ----------
    query_features : paddle.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : paddle.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : paddle.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : paddle.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.
    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.
    """
    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, axis=1)
        gallery_features = F.normalize(gallery_features, p=2, axis=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    query_features, gallery_features = map(to_cpu_numpy, [query_features, gallery_features])

    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 0

    max_k = max(ks)
    # index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
    index_function = faiss.IndexFlatIP if cosine else faiss.IndexFlatL2

    # index = index_function(res, gallery_features.shape[1], flat_config)
    index = index_function(gallery_features.shape[1])

    index.add(gallery_features)
    closest_indices = index.search(query_features, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (query_labels[:, None] == gallery_labels[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}


class RetriEvaluatorSingleTask(DatasetEvaluator):
    """RetriEvaluatorSingleTask
    """
    def __init__(self, cfg, num_query, output_dir=None, **kwargs):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.recalls = cfg.recalls

        self.features = []
        self.labels = []

    def reset(self):
        """reset
        """
        self.features = []
        self.labels = []

    def process(self, inputs, outputs):
        """process
        """
        # remove task dict
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'

        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        self.features.append(outputs.cpu())
        self.labels.extend(inputs["targets"])

    def evaluate(self):
        """evaluate
        """
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            labels = comm.gather(self.labels)
            labels = sum(labels, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            labels = self.labels

        features = paddle.concat(features, axis=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_labels = np.asarray(labels[:self._num_query])



        self._results = OrderedDict()

        if self._num_query == len(features):
            cmc = recall_at_ks(query_features, query_labels, self.recalls, cosine=True)
        else:
            # gallery features, person ids and camera ids
            gallery_features = features[self._num_query:]
            gallery_pids = np.asarray(labels[self._num_query:])
            cmc = recall_at_ks(query_features, query_labels, self.recalls,
                               gallery_features, gallery_pids,
                               cosine=True)

        for r in self.recalls:
            self._results['Recall@{}'.format(r)] = cmc[r]
        self._results["metric"] = cmc[self.recalls[0]]

        return copy.deepcopy(self._results)
