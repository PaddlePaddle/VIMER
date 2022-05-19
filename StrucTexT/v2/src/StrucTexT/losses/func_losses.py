""" func_losses"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle as P
import paddle.nn as nn
from paddle.nn import functional as F

__all__=['dice_loss', 'sigmoid_focal_loss']


def dice_loss(inputs, targets, normalizer=None, epsilon=1e-5):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        normalizer (float|Tensor, optional): The number normalizes the dice loss. It has to be
            a 1-D Tensor whose shape is `[1, ]`. The data type is float32, float64.
            If set to None, the dice loss will not be normalized. Default is None.
    Returns:
        Loss tensor
    """
    inputs = F.sigmoid(inputs)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + epsilon) / (denominator + epsilon)
    loss = loss.mean() if normalizer is None else (loss.sum() / normalizer)
    return loss

def sigmoid_focal_loss(inputs, targets, normalizer=None, alpha=0.25, gamma=2.0):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        normalizer (Tensor, optional): The number normalizes the focal loss. It has to be
            a 1-D Tensor whose shape is `[1, ]`. The data type is float32, float64.
            For object detection task, it is the the number of positive samples.
            If set to None, the focal loss will not be normalized. Default is None.
        alpha(int|float, optional): Hyper-parameter to balance the positive and negative example,
            it should be between 0 and 1.  Default value is set to 0.25.
        gamma(int|float, optional): Hyper-parameter to modulate the easy and hard examples.
            Default value is set to 2.0.
    Returns:
        Loss tensor
    """
    prob = F.sigmoid(inputs)
    targets = P.cast(targets, 'float32')
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss.mean(1)
    loss = loss.mean() if normalizer is None else (loss.sum() / normalizer)
    return loss

