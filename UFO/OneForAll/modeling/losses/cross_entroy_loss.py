# encoding: utf-8
import paddle
import paddle.nn.functional as F

from detectron2.utils.events import EventStorage, get_event_storage
from utils import comm


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.shape[0]
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.equal(gt_classes.reshape((1, -1)).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = paddle.cast(correct[:k].reshape((-1,)), 'float32').sum(axis=0, keepdim=True)
        ret.append(correct_k * (1. / bsz))

    if comm.is_main_process():
        storage = get_event_storage()
        storage.put_scalar("cls_accuracy", ret[0])
    return ret[0]


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    """
    pred_class_outputs
    gt_classes
    eps
    """
    num_classes = pred_class_outputs.shape[1]

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, axis=1)
        smooth_param = alpha * soft_label[paddle.arange(soft_label.shape[0]), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_outputs, axis=1)
    if len(gt_classes.shape) == 1:
        with paddle.no_grad():
            origin_value = paddle.ones_like(log_probs, dtype='float32')
            origin_value *= smooth_param / (num_classes - 1)
            new_value = paddle.ones_like(log_probs, dtype='float32')
            new_value *= (1 - smooth_param)
            o = gt_classes.unsqueeze(-1).tile((1, pred_class_outputs.shape[1]))
            k = paddle.arange(pred_class_outputs.shape[1]).unsqueeze(0).expand_as(o)
            targets = paddle.where(o == k, new_value, origin_value)
    else:
        targets = gt_classes
    # targets = 
    targets = paddle.cast(targets, log_probs.dtype)
    loss = (-1 * targets * log_probs).sum(axis=1)

    with paddle.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).shape[0], 1)

    loss = loss.sum() / non_zero_cnt

    return loss
