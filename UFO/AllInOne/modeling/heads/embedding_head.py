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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
logger = logging.getLogger(__name__)

class BatchNorm(nn.BatchNorm2D):
    """
    BatchNorm
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        """Init
        """
        super().__init__(num_features, epsilon=eps, momentum=momentum)
        # if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        # if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        zeros_ = Constant(value=bias_init)
        ones_ = Constant(value=weight_init)
        zeros_(self.weight)
        ones_(self.bias)
        self.weight.stop_gradient = (not weight_freeze)
        self.bias.stop_gradient =(not bias_freeze)

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
            neck.append(BatchNorm(feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        if dropout:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Identity()

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
        assert not self.training
        if not self.training:
            return neck_feat
        # fmt: on