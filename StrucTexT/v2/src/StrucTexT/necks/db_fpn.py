# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

class EAM(nn.Layer):
    """ external-attention module """
    def __init__(self, in_channels, hidden_channels):
        super(EAM, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.m_k = nn.Conv2D(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='eam_mk', initializer=weight_attr),
            bias_attr=False)

        self.m_v = nn.Conv2D(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                name='eam_mv', initializer=weight_attr),
            bias_attr=False)

    def forward(self, x):
        """ forward """
        n, c, h, w = x.shape
        attn = self.m_k(x)  ## n*in*h*w -> n*hidden*h*w
        attn = paddle.reshape(attn, [0, 0, -1])    ## n*hidden*h*w -> n*hidden*(h*w)
        attn = nn.functional.softmax(attn, axis=2)
        ## l1_normalize along axis 1
        attn = attn / (paddle.sum(attn, axis=1, keepdim=True) + 1e-5)
        attn = paddle.reshape(attn, [0, 0, h, w])
        out = self.m_v(attn)
        return out

class FPN(nn.Layer):
    """ FPN """
    def __init__(self, in_channels, out_channels, with_eam=False,
            eam_hidden_size=64, **kwargs):
        super(FPN, self).__init__()
        self.mode = 'nearest'
        self.with_eam = with_eam
        self.out_channels = out_channels
        if 'mode' in kwargs:
            self.mode = kwargs.get(mode, 'nearest')
        weight_attr = paddle.nn.initializer.KaimingUniform()

        # build in blocks
        self.in_convs = nn.LayerList([
            nn.Conv2D(
                in_channels=in_channel,
                out_channels=out_channels,
                kernel_size=1,
                weight_attr=ParamAttr(
                    name='in{}_conv.w_0'.format(i), initializer=weight_attr),
                bias_attr=False)
            for i, in_channel in enumerate(in_channels, 2)])

        # build out blocks
        self.p_convs = nn.LayerList([
            nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels // len(in_channels),
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(
                    name='p{}_conv.w_0'.format(i + 2), initializer=weight_attr),
                bias_attr=False)
            for i in range(len(in_channels))])

        if with_eam:
            self.eam = EAM(out_channels, eam_hidden_size)

    def forward(self, x):
        """ forward """
        len_s = len(x)
        assert(len_s == len(self.in_convs)) # c2, c3, c4, c5 = x
        ins = [self.in_convs[i](c) for i, c in enumerate(x)] # in2, in3, in4, in5 = g
        outs = [ins[-1]] # in5, out4, out3, out2
        for o in ins[:-1][::-1]:
            out = o + F.upsample(outs[-1], scale_factor=2, mode=self.mode, align_mode=1)
            outs.append(out)
        out = [self.p_convs[i](c) for i, c in enumerate(outs[::-1])] # p2, p3, p4, p5
        pu = [F.upsample(c, scale_factor=2 ** i, mode=self.mode, align_mode=1) for i, c in enumerate(out)]
        fuse = paddle.concat(pu[::-1], axis=1)

        if self.with_eam:
            fuse = self.eam(fuse)
        return {'out': fuse, 'additional_info': {'all_feats': out, 'image_feat': x}}
