"""  layers """
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

import math
import six
import cv2
import numpy as np
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle import to_tensor
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay
from paddle.vision.ops import DeformConv2D

import utils.op as ops
from utils.initializer import xavier_uniform_, constant_
from utils.bbox_utils import delta2bbox


def _to_list(l):
    if isinstance(l, (list, tuple)):
        return list(l)
    return [l]


class DeformableConvV2(nn.Layer):
    """ DeformableConvV2 """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 lr_scale=1,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=L2Decay(0.),
                 dcn_bias_lr_scale=2.):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size ** 2
        self.mask_channel = kernel_size ** 2

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.),
                learning_rate=lr_scale,
                regularizer=regularizer)
        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size ** 2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)
        if skip_quant:
            self.conv_offset.skip_quant = True

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                initializer=Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

    def forward(self, x):
        """ forward """
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


class ConvBNLayer(nn.Layer):
    """ ConvBNLayer """
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 lr_mult=1.0,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                name=name + "_weights", learning_rate=lr_mult),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = nn.BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=lr_mult),
            bias_attr=ParamAttr(
                bn_name + '_offset', learning_rate=lr_mult),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        """ forward """
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class ConvNormLayer(nn.Layer):
    """ ConvNormLayer """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 norm_groups=32,
                 use_dcn=False,
                 bias_on=False,
                 lr_scale=1.,
                 freeze_norm=False,
                 initializer=Normal(
                     mean=0., std=0.01),
                 skip_quant=False,
                 dcn_lr_scale=2.,
                 dcn_regularizer=L2Decay(0.)):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn']

        if bias_on:
            bias_attr = ParamAttr(
                initializer=Constant(value=0.), learning_rate=lr_scale)
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=bias_attr)
            if skip_quant:
                self.conv.skip_quant = True
        else:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(
                    initializer=initializer, learning_rate=1.),
                bias_attr=True,
                lr_scale=dcn_lr_scale,
                regularizer=dcn_regularizer,
                dcn_bias_regularizer=dcn_regularizer,
                dcn_bias_lr_scale=dcn_lr_scale,
                skip_quant=skip_quant)

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2D(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(
                ch_out, weight_attr=param_attr, bias_attr=bias_attr)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=norm_groups,
                num_channels=ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr)

    def forward(self, inputs):
        """ forward """
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class LiteConv(nn.Layer):
    """ LiteConv """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 with_act=True,
                 norm_type='sync_bn',
                 name=None):
        super(LiteConv, self).__init__()
        self.lite_conv = nn.Sequential()
        conv1 = ConvNormLayer(
            in_channels,
            in_channels,
            filter_size=5,
            stride=stride,
            groups=in_channels,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv2 = ConvNormLayer(
            in_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv3 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=1,
            stride=stride,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv4 = ConvNormLayer(
            out_channels,
            out_channels,
            filter_size=5,
            stride=stride,
            groups=out_channels,
            norm_type=norm_type,
            initializer=XavierUniform())
        conv_list = [conv1, conv2, conv3, conv4]
        self.lite_conv.add_sublayer('conv1', conv1)
        self.lite_conv.add_sublayer('relu6_1', nn.ReLU6())
        self.lite_conv.add_sublayer('conv2', conv2)
        if with_act:
            self.lite_conv.add_sublayer('relu6_2', nn.ReLU6())
        self.lite_conv.add_sublayer('conv3', conv3)
        self.lite_conv.add_sublayer('relu6_3', nn.ReLU6())
        self.lite_conv.add_sublayer('conv4', conv4)
        if with_act:
            self.lite_conv.add_sublayer('relu6_4', nn.ReLU6())

    def forward(self, inputs):
        """ forward """
        out = self.lite_conv(inputs)
        return out


class DropBlock(nn.Layer):
    """ DropBlock """
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        """ forward """
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size ** 2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class MultiClassNMS(object):
    """ MultiClassNMS """
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape 
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,] 
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1. 
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        return ops.multiclass_nms(bboxes, score, **kwargs)


class MatrixNMS(object):
    """ MatrixNMS """
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 post_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 use_gaussian=False,
                 gaussian_sigma=2.,
                 normalized=False,
                 background_label=0):
        super(MatrixNMS, self).__init__()
        self.score_threshold = score_threshold
        self.post_threshold = post_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.background_label = background_label

    def __call__(self, bbox, score, *args):
        return ops.matrix_nms(
            bboxes=bbox,
            scores=score,
            score_threshold=self.score_threshold,
            post_threshold=self.post_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k,
            use_gaussian=self.use_gaussian,
            gaussian_sigma=self.gaussian_sigma,
            background_label=self.background_label,
            normalized=self.normalized)


class MaskMatrixNMS(object):
    """
    Matrix NMS for multi-class masks.
    Args:
        update_threshold (float): Updated threshold of categroy score in second time.
        pre_nms_top_n (int): Number of total instance to be kept per image before NMS
        post_nms_top_n (int): Number of total instance to be kept per image after NMS.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
    Input:
        seg_preds (Variable): shape (n, h, w), segmentation feature maps
        seg_masks (Variable): shape (n, h, w), segmentation feature maps
        cate_labels (Variable): shape (n), mask labels in descending order
        cate_scores (Variable): shape (n), mask scores in descending order
        sum_masks (Variable): a float tensor of the sum of seg_masks
    Returns:
        Variable: cate_scores, tensors of shape (n)
    """

    def __init__(self,
                 update_threshold=0.05,
                 pre_nms_top_n=500,
                 post_nms_top_n=100,
                 kernel='gaussian',
                 sigma=2.0):
        super(MaskMatrixNMS, self).__init__()
        self.update_threshold = update_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.kernel = kernel
        self.sigma = sigma

    def _sort_score(self, scores, top_num):
        if paddle.shape(scores)[0] > top_num:
            return paddle.topk(scores, top_num)[1]
        else:
            return paddle.argsort(scores, descending=True)

    def __call__(self,
                 seg_preds,
                 seg_masks,
                 cate_labels,
                 cate_scores,
                 sum_masks=None):
        # sort and keep top nms_pre
        sort_inds = self._sort_score(cate_scores, self.pre_nms_top_n)
        seg_masks = paddle.gather(seg_masks, index=sort_inds)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        sum_masks = paddle.gather(sum_masks, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)

        seg_masks = paddle.flatten(seg_masks, start_axis=1, stop_axis=-1)
        # inter.
        inter_matrix = paddle.mm(seg_masks, paddle.transpose(seg_masks, [1, 0]))
        n_samples = paddle.shape(cate_labels)
        # union.
        sum_masks_x = paddle.expand(sum_masks, shape=[n_samples, n_samples])
        # iou.
        iou_matrix = (inter_matrix / (
            sum_masks_x + paddle.transpose(sum_masks_x, [1, 0]) - inter_matrix))
        iou_matrix = paddle.triu(iou_matrix, diagonal=1)
        # label_specific matrix.
        cate_labels_x = paddle.expand(cate_labels, shape=[n_samples, n_samples])
        label_matrix = paddle.cast(
            (cate_labels_x == paddle.transpose(cate_labels_x, [1, 0])),
            'float32')
        label_matrix = paddle.triu(label_matrix, diagonal=1)

        # IoU compensation
        compensate_iou = paddle.max((iou_matrix * label_matrix), axis=0)
        compensate_iou = paddle.expand(
            compensate_iou, shape=[n_samples, n_samples])
        compensate_iou = paddle.transpose(compensate_iou, [1, 0])

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = paddle.exp(-1 * self.sigma * (decay_iou ** 2))
            compensate_matrix = paddle.exp(-1 * self.sigma *
                                           (compensate_iou ** 2))
            decay_coefficient = paddle.min(decay_matrix / compensate_matrix,
                                           axis=0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient = paddle.min(decay_matrix, axis=0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores = cate_scores * decay_coefficient
        y = paddle.zeros(shape=paddle.shape(cate_scores), dtype='float32')
        keep = paddle.where(cate_scores >= self.update_threshold, cate_scores,
                            y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        keep = paddle.concat(
            [keep, paddle.cast(paddle.shape(cate_scores)[0] - 1, 'int64')])

        seg_preds = paddle.gather(seg_preds, index=keep)
        cate_scores = paddle.gather(cate_scores, index=keep)
        cate_labels = paddle.gather(cate_labels, index=keep)

        # sort and keep top_k
        sort_inds = self._sort_score(cate_scores, self.post_nms_top_n)
        seg_preds = paddle.gather(seg_preds, index=sort_inds)
        cate_scores = paddle.gather(cate_scores, index=sort_inds)
        cate_labels = paddle.gather(cate_labels, index=sort_inds)
        return seg_preds, cate_scores, cate_labels


def Conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           bias=True,
           weight_init=Normal(std=0.001),
           bias_init=Constant(0.)):
    """ Conv2d """
    weight_attr = paddle.framework.ParamAttr(initializer=weight_init)
    if bias:
        bias_attr = paddle.framework.ParamAttr(initializer=bias_init)
    else:
        bias_attr = False
    conv = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        weight_attr=weight_attr,
        bias_attr=bias_attr)
    return conv


def ConvTranspose2d(in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    output_padding=0,
                    groups=1,
                    bias=True,
                    dilation=1,
                    weight_init=Normal(std=0.001),
                    bias_init=Constant(0.)):
    """ ConvTranspose2d """
    weight_attr = paddle.framework.ParamAttr(initializer=weight_init)
    if bias:
        bias_attr = paddle.framework.ParamAttr(initializer=bias_init)
    else:
        bias_attr = False
    conv = nn.Conv2DTranspose(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        weight_attr=weight_attr,
        bias_attr=bias_attr)
    return conv


def BatchNorm2d(num_features, eps=1e-05, momentum=0.9, affine=True):
    """ BatchNorm2d """
    if not affine:
        weight_attr = False
        bias_attr = False
    else:
        weight_attr = None
        bias_attr = None
    batchnorm = nn.BatchNorm2D(
        num_features,
        momentum,
        eps,
        weight_attr=weight_attr,
        bias_attr=bias_attr)
    return batchnorm


def ReLU():
    """ ReLU """
    return nn.ReLU()


def Upsample(scale_factor=None, mode='nearest', align_corners=False):
    """ Upsample """
    return nn.Upsample(None, scale_factor, mode, align_corners)


def MaxPool(kernel_size, stride, padding, ceil_mode=False):
    """ MaxPool """
    return nn.MaxPool2D(kernel_size, stride, padding, ceil_mode=ceil_mode)


class Concat(nn.Layer):
    """ Concat """
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        """ forward """
        return paddle.concat(inputs, axis=self.dim)

    def extra_repr(self):
        """ extra_repr """
        return 'dim={}'.format(self.dim)


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.
    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.
    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    return nn.layer.transformer._convert_attention_mask(attn_mask, dtype)


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=False)
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
                attr=None,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._type_list = ('q_proj', 'k_proj', 'v_proj')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                constant_(p)

    def compute_qkv(self, tensor, index):
        """ compute_qkv """
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                x=tensor,
                weight=self.in_proj_weight[:, index * self.embed_dim:(index + 1)
                                           * self.embed_dim],
                bias=self.in_proj_bias[index * self.embed_dim:(index + 1) *
                                       self.embed_dim]
                if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.reshape(
            [0, 0, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = (self.compute_qkv(t, i)
                   for i, t in enumerate([query, key, value]))

        # scale dot product attention
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        vis_mask = None
        if attn_mask is not None:
            if attn_mask.shape[1] == 2:
                attn_mask, vis_mask = attn_mask.unsqueeze(1).unbind(2)
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        '''
        if paddle.shape(weights)[-1] == 4096:
            for k, x in enumerate(weights.unbind(0)):
                x = x.abs()
                x = (x - paddle.min(x)) * 255 / (paddle.max(x) - paddle.min(x))
                x = x[:, 0] #(hs, q, wh)
                wh = int(math.sqrt(x.shape[-1]))
                x = x.reshape((-1, wh, wh)) # (hs, w, h)
                for i, z in enumerate(x.unbind(0)):
                    y = paddle.cast(z, 'int32').numpy().astype('uint8')
                    y = cv2.resize(y, (512, 512))
                    cv2.imwrite('tmp2/b%d_q0_h%d_pred_atten.jpg' % (k, i), y)
            if vis_mask is not None:
                a = vis_mask[:, 0, 0] #(bs, wh)
                a = (a - paddle.min(a)) * 255 / (1e-7 + paddle.max(a) - paddle.min(a))
                wh = int(math.sqrt(a.shape[-1]))
                a = a.reshape((-1, wh, wh))
                for i, c in enumerate(a.unbind(0)):
                    b = paddle.cast(c[0], 'int32').numpy().astype('uint8')
                    b = cv2.resize(b, (512, 512))
                    cv2.imwrite('tmp2/b%d_q0_h0_gt_atten.jpg' % (i), b)
        if vis_mask is not None:
            weights = vis_mask
        '''
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


class ConvMixer(nn.Layer):
    """ ConvMixer """
    def __init__(
            self,
            dim,
            depth,
            kernel_size=3, ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size

        self.mixer = self.conv_mixer(dim, depth, kernel_size)

    def forward(self, x):
        """ forward """
        return self.mixer(x)

    @staticmethod
    def conv_mixer(
            dim,
            depth,
            kernel_size, ):
        """ conv_mixer """
        Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2D(dim))
        Residual = type('Residual', (Seq, ),
                        {'forward': lambda self, x: self[0](x) + x})
        return Seq(*[
            Seq(Residual(
                ActBn(
                    nn.Conv2D(
                        dim, dim, kernel_size, groups=dim, padding="same"))),
                ActBn(nn.Conv2D(dim, dim, 1))) for i in range(depth)
        ])
