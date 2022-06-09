import paddle 
import paddle.nn as nn
import paddle.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Layer):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, axis=-1)
        nll_loss = -logprobs.gather(axis=-1, index=target.unsqueeze([1]))
        nll_loss = nll_loss.squeeze([1])
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

    def __str__(self,):
        return type(self).__name__

class SoftTargetCrossEntropy(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        return loss.mean()
    
    def __str__(self,):
        return type(self).__name__