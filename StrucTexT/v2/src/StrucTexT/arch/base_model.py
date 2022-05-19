""" Encoder """
import os
import cv2
import json
import copy
import numpy as np
import importlib
import paddle
import paddle as P
import paddle.nn as nn
import paddle.nn.functional as F
from ..necks import build_neck
from ..backbones import build_backbone
from ..ernie import ErnieModel, build_linear


class ImageEncoder(nn.Layer):
    """ ImageEncoder """
    def __init__(self, config, name=''):
        """ __init__ """
        super(ImageEncoder, self).__init__()

        module = config.pop('name')
        neck_config = config.pop('neck') if 'neck' in config else None
        self.net = build_backbone(module, config)
        in_channels = self.net.out_channels
        if neck_config:
            neck_config['in_channels'] = in_channels
            neck_name = neck_config.pop('name')
            self.neck = build_neck(
                    neck_name,
                    neck_config)
            in_channels = self.neck.out_channels
        else:
            self.neck = None
        config['out_channels'] = in_channels

    def forward(self, x, **kwargs):
        """ forward """
        x = self.net(x, **kwargs)
        if self.neck:
            x = self.neck(x['additional_info']['all_feats'], **kwargs)
        return x


class TextEncoder(ErnieModel):
    """ TextEncoder """
    def __init__(self, config, name=''):
        super(TextEncoder, self).__init__(config)
        config['out_channels'] = config['hidden_size']

    def forward(self, x, **kwargs):
        """ forward """
        sent_ids = kwargs.get('sent_ids')
        type_ids = kwargs.get('type_ids')
        pooled, encoded = super(TextEncoder, self).forward(
                src_ids=x,
                sent_ids=sent_ids,
                type_ids=type_ids)
        return encoded


class FusionEncoder(nn.Layer):
    """ FusionEncoder """
    def __init__(self, config, name=''):
        super(FusionEncoder, self).__init__()
        config = copy.deepcopy(config)

        module = config.pop('name')
        self.net = build_backbone(module, config)

    def forward(self, x, **kwargs):
        """ forward """
        x = self.net(x, **kwargs)
        return x


class Encoder(nn.Layer):
    """ Encoder """
    def __init__(self, config, name=''):
        """ __init__ """
        super(Encoder, self).__init__()
        config = copy.deepcopy(config)

        self.out_channels = None

        image_config = config.get('image_module')
        if image_config is not None:
            self.visual_embedding = ImageEncoder(image_config)
            self.out_channels = image_config['out_channels']
        else:
            self.visual_embedding = None

        text_config = config.get('text_module')
        if text_config is not None:
            if isinstance(text_config, str):
                text_config = json.loads(open(text_config).read())
            self.text_embedding = TextEncoder(text_config)
            self.out_channels = text_config['out_channels']
        else:
            self.text_embedding = None

        if self.out_channels is None:
            raise RuntimeError('vision tower and text tower cannot both be empty')

        fusion_config = config.get('fusion_module')
        if fusion_config is not None:
            self.fusion_embedding = FusionEncoder(fusion_config)
            self.out_channels = fusion_config['d_model']
        else:
            self.fusion_embedding = None


    def encode_image(self, x, **kwargs):
        """ encode_image """
        if self.visual_embedding:
            x = self.visual_embedding(x, **kwargs)
        return x

    def encode_text(self, x, **kwargs):
        """ encode_text """
        if self.text_embedding:
            x = self.text_embedding(x, **kwargs)
        return x

    def encode_fusion(self, x, **kwargs):
        """ encode_fusion """
        if self.fusion_embedding:
            x = self.fusion_embedding(x, **kwargs)
        return x

    def forward(self, x, **kwargs):
        """ forward """
        image, text = x
        if image is not None:
            image_features = self.encode_image(image, **kwargs)
        else:
            image_features = None

        if text is not None:
            text_features = self.encode_text(text, **kwargs)
        else:
            text_features = None

        if self.fusion_embedding:
            input = [image_features['out'], text_features]
            enc_output = self.encode_fusion(input, **kwargs)
        else:
            enc_output = None

        return {'out': enc_output,
                'additional_info': {
                    'image_feat': image_features,
                    'text_feat': text_features}
               }
