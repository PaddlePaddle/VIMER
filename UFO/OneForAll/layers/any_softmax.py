# encoding: utf-8
"""paddle authors
"""

import paddle
import paddle.nn as nn

__all__ = [
    "Linear",
    "CosSoftmax",
]


class Linear(nn.Layer):
    """Linear
    """
    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        """forward
        """
        return logits * self.s

    def extra_repr(self):
        """extra_repr
        """
        return "num_classes={}, scale={}, margin={}".format(self.num_classes, self.s, self.m)


class CosSoftmax(Linear):
    r"""Implement of large margin cosine distance:
    """

    def forward(self, logits, targets):
        """forward
        """
        m_zero = paddle.zeros_like(logits, dtype=logits.dtype)
        m_valid = paddle.ones_like(logits, dtype=logits.dtype) * self.m
        
        o = targets.unsqueeze(-1).tile((1, logits.shape[1]))
        k = paddle.arange(logits.shape[1]).unsqueeze(0).expand_as(o)
        m_hot = paddle.where(o == k, m_valid, m_zero)
        logits -= m_hot

        logits = logits * self.s
        return logits