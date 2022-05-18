"""Build Head
"""
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import logging
from functools import partial
from collections import OrderedDict
import os

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from paddle import ParamAttr

from layers import any_softmax

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
normal_ = Normal
BIAS_LR_FACTOR=2.0

logger = logging.getLogger(__name__)

class SyncBatchNorm(nn.SyncBatchNorm):
    """
    SyncBatchNorm
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.9, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        """Init
        """
        # paddle 不支持对weight 和 bias中的某一个参数单独设置不更新梯度stop_gradient=True，采用调整学习率比例的方式来freeze某个参数，
        # 即将学习率比例设置为0.000000001；
        if weight_freeze:
            weight_attr = ParamAttr(learning_rate=0.000000001)
        else:
            weight_attr = ParamAttr(learning_rate=1.)
        if bias_freeze:
            bias_attr = ParamAttr(learning_rate=0.000000001)
        else:
            bias_attr = ParamAttr(learning_rate=1.) 
        super().__init__(num_features, epsilon=eps, momentum=momentum, weight_attr=weight_attr, bias_attr=bias_attr)
        zeros_ = Constant(value=bias_init)
        ones_ = Constant(value=weight_init)
        zeros_(self.weight)
        ones_(self.bias)


class BatchNorm(nn.BatchNorm2D):
    """
    BatchNorm
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.9, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        """Init
        """
        # paddle 不支持对weight 和 bias中的某一个参数单独设置不更新梯度stop_gradient=True，采用调整学习率比例的方式来freeze某个参数，
        # 即将学习率比例设置为0.000000001；
        if weight_freeze:
            weight_attr = ParamAttr(learning_rate=0.000000001)
        else:
            weight_attr = ParamAttr(learning_rate=1.)
        if bias_freeze:
            bias_attr = ParamAttr(learning_rate=0.000000001)
        else:
            bias_attr = ParamAttr(learning_rate=1.) 
        super().__init__(num_features, epsilon=eps, momentum=momentum, weight_attr=weight_attr, bias_attr=bias_attr)
        zeros_ = Constant(value=bias_init)
        ones_ = Constant(value=weight_init)
        zeros_(self.weight)
        ones_(self.bias)


class GlobalAvgPool(nn.AdaptiveAvgPool2D):
    """
    GlobalAvgPool
    """
    def __init__(self, output_size=1, *args, **kwargs):
        """Init
        """
        super().__init__(output_size)


class EmbeddingHead(nn.Layer):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    def __init__(
            self,
            feat_dim=2048,
            embedding_dim=0,
            num_classes=0,
            neck_feat="before",
            pool_type="GlobalAvgPool",
            cls_type="Linear",
            scale=1,
            margin=0,
            with_bnneck=False,
            norm_type='BN',
            dropout=False,
            share_last=True,
            pretrain_path='',
            load_head=False,
            depth='base',
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        self.share_last = share_last
        self.pool_layer = GlobalAvgPool()
        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2D(feat_dim, embedding_dim, 1, 1, bias_attr=False))
            feat_dim = embedding_dim

        if with_bnneck:
            if norm_type == 'BN':
                neck.append(BatchNorm(feat_dim, bias_freeze=True))
            elif norm_type == 'SyncBN':
                neck.append(SyncBatchNorm(feat_dim, bias_freeze=True))
            else:
                assert norm_type in ['BN', 'SyncBN'], "No norm_type of {} is found".format(norm_type)
        self.bottleneck = nn.Sequential(*neck)

        if dropout:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Identity()

        # Classification head
        self.cls_type = cls_type        
        
        if self.cls_type == 'classification':
            assert num_classes > 0
            self.linear = paddle.nn.Linear(feat_dim, num_classes, bias_attr=ParamAttr(learning_rate=0.1*BIAS_LR_FACTOR), weight_attr=ParamAttr(learning_rate=0.1))
            if self.training:
                self.cls_layer = getattr(any_softmax, "Linear")(num_classes, scale, margin)
        else:
            if self.training and num_classes > 0:
                assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                                    "but got {}".format(any_softmax.__all__, cls_type)
                self.weight = paddle.create_parameter(shape=(feat_dim, num_classes), dtype='float32', 
                                                    default_initializer=Normal(std=0.001)
                                                    )
                self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        self.reset_parameters()
        if self.cls_type == 'classification' and load_head:
            # pretrain_path
            state_dict = paddle.load(pretrain_path)
            logger.info("Loading Head from {}".format(pretrain_path))
            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                if 'head' in k:
                    k_new = k[5:]
                    state_dict_new[k_new] = state_dict[k]
            self.linear.set_state_dict(state_dict_new)
        
        self.feature_dict = None

    def reset_parameters(self):
        # self.weight 的初始化方式在创建过程中被指定，
        # 例如self.weight = paddle.create_parameter(shape=(num_classes, feat_dim), dtype='float32', 
        #                                        default_initializer=TruncatedNormal(std=0.01)
        #                                       )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, BatchNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            #TODO add initilization for conv2d
            pass
    
    def forward(self, features, targets=None):
        """
        NOTE: this forward function only supports `self.training is False`
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = self.dropout(neck_feat)
        neck_feat = neck_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training:
            if self.cls_type == 'classification':
                return self.linear(neck_feat)
            else:
                return neck_feat
        # fmt: on

        # Training
        if self.cls_type == 'classification':
            logits = self.linear(neck_feat)
        else:
            if self.cls_layer.__class__.__name__ == 'Linear':
                logits = F.linear(neck_feat, self.weight.transpose((1, 0)))
            else:
                logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight, axis=0))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError("{} is invalid for MODEL.HEADS.NECK_FEAT".format(self.neck_feat))
        # fmt: on
        self.feature_dict = {"neck_feat":neck_feat, "logits":logits, "cls_outputs": cls_outputs, "pred_class_logits": logits * self.cls_layer.s,  "features": feat,}
        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits * self.cls_layer.s,
            "features": feat,
        }