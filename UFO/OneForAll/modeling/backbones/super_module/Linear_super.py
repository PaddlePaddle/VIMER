"""super_module/Linear_super.py
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


xavier_uniform_ =  paddle.nn.initializer.XavierUniform()
constant_ = paddle.nn.initializer.Constant()

class LinearSuper(nn.Linear):
    """LinearSuper
    """
    def __init__(self, super_in_dim, super_out_dim,
            bias_attr=None, uniform_=None, non_linear='linear', scale=False, weight_attr=None):
        super().__init__(super_in_dim, super_out_dim, weight_attr=weight_attr, bias_attr=bias_attr)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear) #TODO add initialization for weights
        self.profiling = False

    def _reset_parameters(self, bias, uniform_, non_linear):
        xavier_uniform_(self.weight) if uniform_ is None else uniform_(self.weight) #TODO add non_linear
        if bias:
            constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        """set_sample_config
        """
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        """_sample_parameters
        """
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        """forward
        """
        self._sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        """"
        """

        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    
    def get_complexity(self, sequence_length):
        """get_complexity
        """
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim):
    """sample_weight
    """
    sample_weight = weight[:sample_in_dim, :]
    sample_weight = sample_weight[:, :sample_out_dim]

    return sample_weight


def sample_bias(bias, sample_out_dim):
    """sample_bias
    """
    sample_bias = bias[:sample_out_dim]

    return sample_bias
