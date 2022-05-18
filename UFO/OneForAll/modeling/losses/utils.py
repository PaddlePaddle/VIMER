# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import paddle
import paddle.nn.functional as F

def euclidean_dist(x, y):
    """euclidean_dist
    """
    m, n = x.shape[0], y.shape[0]
    xx = paddle.pow(x, 2).sum(1, keepdim=True).expand((m, n))
    yy = paddle.pow(y, 2).sum(1, keepdim=True).expand((n, m)).t()
    dist = xx + yy - 2 * paddle.matmul(x, y.t())
    dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    """cosine_dist
    """
    x = F.normalize(x, axis=1)
    y = F.normalize(y, axis=1)
    dist = 2 - 2 * paddle.matmul(x, y.t())
    return dist
