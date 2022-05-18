# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import paddle
import paddle.nn.functional as F

from .utils import euclidean_dist, cosine_dist


def softmax_weights(dist, mask):
    """
    dist
    mask
    """
    max_v = paddle.max(dist * mask, axis=1, keepdim=True)[0]
    diff = dist - max_v
    Z = paddle.sum(paddle.exp(diff) * mask, axis=1, keepdim=True) + 1e-6  # avoid division by zero
    W = paddle.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: a tensor, distance(anchor, positive); shape [N]
      dist_an: a tensor, distance(anchor, negative); shape [N]
      p_inds: a tensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: a tensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.shape) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap = paddle.max(dist_mat * is_pos, axis=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an = paddle.min(dist_mat * is_neg + is_pos * 1e9, axis=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: a tensor, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: a tensor, distance(anchor, positive); shape [N]
      dist_an: a tensor, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = paddle.sum(dist_ap * weights_ap, axis=1)
    dist_an = paddle.sum(dist_an * weights_an, axis=1)

    return dist_ap, dist_an


def soft_margin_loss(x, y):
    """
      Args:
        x: shape [N]
        y: shape [N]
    """
    return paddle.sum(paddle.log(1 + paddle.exp(-1 * x * y))) /  x.numel()


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    #TODO For distributed training, gather all features from different process. 
    # if comm.get_world_size() > 1:
    #     pass
    # else:
    #     pass

    N = dist_mat.shape[0]
    is_pos = paddle.cast(
      targets.reshape((N, 1)).expand((N, N)).equal(targets.reshape((N, 1)).expand((N, N)).t()), 
      embedding.dtype)
    is_neg = paddle.cast(
      targets.reshape((N, 1)).expand((N, N)).not_equal(targets.reshape((N, 1)).expand((N, N)).t()),
      embedding.dtype)

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    # y = dist_an.new().resize_as_(dist_an).fill_(1)
    y = paddle.ones_like(dist_an)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss
