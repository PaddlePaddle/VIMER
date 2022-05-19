""" V2Net """
import math
import json
import random
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pathlib import Path
from ppdet.core.workspace import register, serializable
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Uniform
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.vision.ops import DeformConv2D

from .name_adapter import NameAdapter
from ..shape_spec import ShapeSpec
from .modeling_ernie import ErnieEncoderStack

__all__ = ['V2Net']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

class ConvNormLayer(nn.Layer):
    """ ConvNormLayer """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=False)
        else:
            self.offset_channel = 2 * filter_size**2
            self.mask_channel = filter_size**2

            self.conv_offset = nn.Conv2D(
                in_channels=ch_in,
                out_channels=3 * filter_size**2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                weight_attr=ParamAttr(initializer=Constant(0.)),
                bias_attr=ParamAttr(initializer=Constant(0.)))
            self.conv = DeformConv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                weight_attr=ParamAttr(learning_rate=lr),
                bias_attr=False)

        norm_lr = 0. if freeze_norm else lr
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),
            trainable=False if freeze_norm else True)

        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            self.norm = nn.BatchNorm2D(
                ch_out,
                weight_attr=param_attr,
                bias_attr=bias_attr,
                use_global_stats=global_stats)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.stop_gradient = True

    def forward(self, inputs):
        """ forward """
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset, mask = paddle.split(
                offset_mask,
                num_or_sections=[self.offset_channel, self.mask_channel],
                axis=1)
            mask = F.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


class SELayer(nn.Layer):
    """ SELayer """
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(
            ch,
            c_,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(
            c_,
            ch,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

    def forward(self, inputs):
        """ forward """
        out = self.pool(inputs)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        scale = out * inputs
        return scale


class ConvBNLayer(nn.Layer):
    """ ConvBNLayer """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act)

    def forward(self, inputs):
        """ forward """
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    """ BottleneckBlock """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        """ forward """
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    """ BasicBlock """
    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BasicBlock, self).__init__()
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_sublayer(
                    'pool',
                    nn.AvgPool2D(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_sublayer(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        """ forward """
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)
        out = F.relu(out)
        return out


class PositionalEncoding(nn.Layer):
    """ PositionalEncoding """
    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = paddle.zeros((max_len, d_model))
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
        div_term = paddle.exp(paddle.arange(0, d_model, 2).astype('float32') * -math.log(10000.0) / d_model)
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, feat, **kwargs):
        """ forward """
        feat = feat + self.pe[:, :feat.shape[1]] # pe 1*5000*512
        return self.dropout(feat)


class ResNet(nn.Layer):
    """ ResNet """
    def __init__(self, **kwargs):
        in_channels = 3
        layers = kwargs.get('layer', 50)
        self.shrink_ratio = 0.125
        super(ResNet, self).__init__()

        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.stages = []
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    block_list.append(bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    shortcut = True
                    block_list.append(basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))


    def forward(self, inputs, data=None):
        """ forward """
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            out.append(y)
        return {'out': y, 'all_feats': out}


@register
@serializable
class V2Net(nn.Layer):
    """ V2Net """
    def __init__(self, out_channels=128, **kwargs):
        super(V2Net, self).__init__()
        self.cnn = ResNet(**kwargs)
        self.in_channels = self.cnn.out_channels
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.input_proj = nn.Conv2D(
                2048, out_channels, kernel_size=1)
        cfg = {
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": self.out_channels,
                "num_attention_heads": 12,
                "num_hidden_layers": 8
        }
        self.transformer = ErnieEncoderStack(cfg)
        self.positional_encoding = PositionalEncoding(
                d_model=self.out_channels)

        self.fuse_conv2 = nn.Conv2D(
            in_channels=self.out_channels // 4 * 5,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.fuse_conv3 = nn.Conv2D(
            in_channels=self.out_channels // 4 * 5,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.fuse_conv4 = nn.Conv2D(
            in_channels=self.out_channels // 4 * 5,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.fuse_conv5 = nn.Conv2D(
            in_channels=self.out_channels // 4 * 5,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        na = NameAdapter(self)


    def forward(self, inputs):
        """ forward """
        ### CNN
        x = self.cnn(inputs['image'])
        c2, c3, c4, c5 = x['all_feats']

        ### Transformer
        # use last level feature map
        src_proj = self.input_proj(c5)
        bs, c, h, w = src_proj.shape
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten = src_proj.flatten(2).transpose([0, 2, 1])
        src_flatten = self.positional_encoding(src_flatten)
        context_embed, _, _ = self.transformer(src_flatten)
        context_embed = context_embed.transpose([0, 2, 1]).reshape([bs, c, h, w])

        ### FPN
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        p5 = paddle.concat([p5, context_embed], axis=1)
        p4 = paddle.concat([p4, F.upsample(context_embed, scale_factor=2, mode="nearest", align_mode=1)], axis=1)
        p3 = paddle.concat([p3, F.upsample(context_embed, scale_factor=4, mode="nearest", align_mode=1)], axis=1)
        p2 = paddle.concat([p2, F.upsample(context_embed, scale_factor=8, mode="nearest", align_mode=1)], axis=1)
        fuse = [self.fuse_conv2(p2), self.fuse_conv3(p3), self.fuse_conv4(p4), self.fuse_conv5(p5)]

        return fuse

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channels, stride=s) for s in [4,8,16,32]
        ]
