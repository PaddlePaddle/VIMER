# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# Modified from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/compute_dist.py


import faiss
import numpy as np
import paddle
import paddle.nn.functional as F

__all__ = [
    "build_dist",
    "compute_jaccard_distance",
    "compute_euclidean_distance",
    "compute_cosine_distance",
]


@paddle.no_grad()
def build_dist(feat_1, feat_2, metric="euclidean", **kwargs):
    r"""Compute distance between two feature embeddings.

    Args:
        feat_1 (paddle.Tensor): 2-D feature with batch dimension.
        feat_2 (paddle.Tensor): 2-D feature with batch dimension.
        metric:

    Returns:
        numpy.ndarray: distance matrix.
    """
    assert metric in ["cosine", "euclidean", "jaccard"], "Expected metrics are cosine, euclidean and jaccard, " \
                                                         "but got {}".format(metric)

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)

    # elif metric == "jaccard":
    #     feat = paddle.concat((feat_1, feat_2), axis=0)
    #     dist = compute_jaccard_distance(feat, k1=kwargs["k1"], k2=kwargs["k2"], search_option=0)
    #     return dist[:feat_1.size(0), feat_1.size(0):]


def k_reciprocal_neigh(initial_rank, i, k1):
    """k_reciprocal_neigh
    """
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


@paddle.no_grad()
def compute_jaccard_distance(features, k1=20, k2=6, search_option=0, fp16=False):
    """compute_jaccard_distance
    """
    pass


@paddle.no_grad()
def compute_euclidean_distance(features, others):
    """compute_euclidean_distance
    """
    m, n = features.size(0), others.size(0)
    dist_m = (
            paddle.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + paddle.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t()).numpy()

    return dist_m


@paddle.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (paddle.Tensor): 2-D feature matrix.
        others (paddle.Tensor): 2-D feature matrix.
    Returns:
        paddle.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, axis=1)
    others = F.normalize(others, p=2, axis=1)
    dist_m = 1 - paddle.mm(features, others.t()).numpy()
    return dist_m
