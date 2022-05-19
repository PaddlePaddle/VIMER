""" none_op """
from paddle import nn

class NoneOp(nn.Layer):
    """ NoneOp """
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x, **kwargs):
        """ forward """
        return x
