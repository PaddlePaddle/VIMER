import warnings
import paddle 
import paddle.nn.functional as F
from paddle.nn.layer.norm import _BatchNormBase


class LP_BatchNorm(_BatchNormBase):
    """ A variant used in linear probing.
    To freeze parameters (normalization operator specifically), model set to eval mode during linear probing.
    According to paper, an extra BN is used on the top of encoder to calibrate the feature magnitudes.
    In addition to self.training, we set another flag in this implement to control BN's behavior to train in eval mode.
    """

    def __init__(self, num_features, epsilon=1e-5, momentum=0.1, weight_attr=False, bias_attr=False):
        super(LP_BatchNorm, self).__init__(num_features, momentum, epsilon, 
          weight_attr=weight_attr, bias_attr=bias_attr, use_global_stats=None)

    def _check_data_format(self, input):
        if input == 'NCHW' or input == 'NC' or input == 'NCL':
            self._data_format = 'NCHW'
        elif input == "NHWC" or input == 'NLC':
            self._data_format = "NHWC"
        else:
            raise ValueError(
                'expected NC , NCL, NLC or None for data_format input')

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, is_train):
        """
        We use is_train instead of self.training.
        """
        self._check_data_format(self._data_format)
 
        self._check_input_dim(input)
        if is_train:
            warnings.warn(
                "When training, we now always track global mean and variance.")

        assert self._mean is None or isinstance(self._mean, paddle.Tensor)
        assert self._variance is None or isinstance(self._variance, paddle.Tensor)
        return F.batch_norm(
            input,
            self._mean,
            self._variance,
            weight=self.weight,
            bias=self.bias,
            training=is_train,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format,
            use_global_stats=self._use_global_stats)