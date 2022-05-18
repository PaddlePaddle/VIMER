"""Build Vision Transformer
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
import math
from functools import partial
import copy
from collections import OrderedDict
import os

import paddle
import paddle.nn as nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.initializer import TruncatedNormal
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Normal
import numpy as np

from .super_module.Linear_super import LinearSuper
from .super_module.multihead_super import AttentionSuper

logger = logging.getLogger(__name__)


def to_2tuple(x):
    """Convert x to [x, x]
    """
    return tuple([x] * 2)

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

BIAS_LR_FACTOR=2.0
WEIGHT_DECAY_NORM=0.0001 #TODO add additional weight decay for norm_type module

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        """Init
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    """MLP Layer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        params:
        return:
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearSuper(in_features, hidden_features, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        self.act = act_layer()
        self.fc2 = LinearSuper(hidden_features, out_features, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def set_sample_config(self, sample_in_dim=None, sample_ffn_dim=None, sample_out_dim=None):
        """set_sample_config
        """
        self.fc1.set_sample_config(sample_in_dim=sample_in_dim, sample_out_dim=sample_ffn_dim)
        self.fc2.set_sample_config(sample_in_dim=sample_ffn_dim, sample_out_dim=sample_out_dim)

class Block(nn.Layer):
    """Block Layer
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, change_qkv=True, predefined_head_dim=64,
                 use_checkpointing=False):
        """Init
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, change_qkv=change_qkv, predefined_head_dim=predefined_head_dim
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.feature = None
        self.super_mlp_ratio = mlp_ratio
        self.predefined_head_dim = predefined_head_dim
        self.use_checkpointing=use_checkpointing

    def forward(self, x):
        """Forward
        """
        if self.is_identity_layer == True:
            return x
        if self.use_checkpointing:
            x = x + self.drop_path(recompute(self.attn, self.norm1(x)))
            x = x + self.drop_path(recompute(self.mlp, self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        self.feature = x
        return x
    
    def set_sample_config(self, is_identity_layer, 
        sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, 
        sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):
        """set_sample_config
        """
        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer * self.predefined_head_dim, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)
        self.mlp.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_ffn_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """Init
        """        
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Forward
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({}*{}) doesn't match model ({}*{}).".format(H, W, self.img_size[0], self.img_size[1])

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class HybridEmbed(nn.Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        """Init
        """
        super().__init__()
        assert isinstance(backbone, nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(paddle.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2D(feature_dim, embed_dim, 1)

    def forward(self, x):
        """Forward
        """
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class PatchEmbedOverlap(nn.Layer):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        """Init
        """
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size, 
            bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))
        n = embed_dim * patch_size[0] * patch_size[1]
        TruncatedNormal(self.proj.weight, math.sqrt(2. / n))

    def forward(self, x):
        """Forward
        """
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({}*{}) doesn't match model ({}*{}).".format(H, W, self.img_size[0], self.img_size[1])
        x = self.proj(x)

        x = x.flatten(2).transpose((0, 2, 1))  # [64, 8, 768]
        return x

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    """calc_dropout
    """
    return dropout * 1.0 * sample_embed_dim / super_embed_dim
    
class VisionTransformer(nn.Layer):
    """ Vision Transformer
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6), sie_xishu=1.0,
                 predefined_head_dim=64,
                 use_checkpointing=False, share_last=True):
        """Init
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbedOverlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.super_embed_dim = embed_dim
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = self.create_parameter(
                shape=(camera, 1, embed_dim), default_initializer=zeros_)
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.share_last = share_last
        if not self.share_last:
            depth -= 1
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_checkpointing=False,
                predefined_head_dim=predefined_head_dim)
            for i in range(depth)])

        if self.share_last:
            self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token)
        trunc_normal_(self.pos_embed)

        self.apply(self._init_weights)

        self.use_checkpointing = use_checkpointing

    def set_sample_config(self, config):
        """set_sample_config
        """
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        # self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < (len(self.blocks) if (self.training and self.use_checkpointing) else self.sample_layer_num):
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, 
                                                    self.sample_embed_dim[i], 
                                                    self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        """Parameters of using no weight decay
        """
        return {'pos_embed', 'cls_token'}

    def forward(self, x, camera_id=None):
        """Foward
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if idx < self.sample_layer_num:
                if self.use_checkpointing:
                    x = recompute(blk, x)
                else:
                    x = blk(x)
            else:
                if self.training:
                # In training mode, the checkpointing technique requires calculating the gradients in last unused blocks.
                # Thus, we make forward calculation of last unused blocks, 
                # but set a ZERO mask to the results of last unused blocks to eliminate theirs effect on final features
                # Note, the checkpointing technique is activated in DEFAULT. 
                    if self.use_checkpointing:
                        x_pesudo = recompute(blk, x)
                        x = 0 * x_pesudo + x
                    else:
                        x_pesudo = blk(x)
                        x = 0 * x_pesudo + x
                else:
                    pass

        if self.share_last:
            x = self.norm(x)
            return x[:, 0].reshape((x.shape[0], -1, 1, 1))
        else:
            return x


def resize_pos_embed(posemb, posemb_new, hight, width):
    """
    Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape((1, gs_old, gs_old, -1)).transpose((0, 3, 1, 2))
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.transpose((0, 2, 3, 1)).reshape((1, hight * width, -1))
    posemb = paddle.concat([posemb_token, posemb_grid], axis=1)
    return posemb


def build_vit_backbone_lazy(pretrain=False, pretrain_path='', pretrain_npz=False,
                            input_size=[256, 128], depth='base', sie_xishu=3.0, stride_size=[16, 16],
                            patch_size=14, drop_ratio=0.0, drop_path_ratio=0.1, attn_drop_rate=0.0,
                            predefined_head_dim=64,
                            use_checkpointing=False, share_last=True):
    """
    Create a Vision Transformer instance from config.
    Returns:
        Vision Transformer: a :class:`VisionTransformer` instance.
    """

    embed_dim = {
        'small': 768,
        'base': 768,
        'large': 1024,
        'huge': 1280,
    }[depth]

    num_depth = {
        'small': 8,
        'base': 12,
        'large': 24,
        'huge': 32,
    }[depth]

    num_heads = {
        'small': 8,
        'base': 12,
        'large': 16,
        'huge': 16,
    }[depth]

    mlp_ratio = {
        'small': 3.,
        'base': 4.,
        'large': 4.,
        'huge': 4.,
    }[depth]

    qkv_bias = {
        'small': False,
        'base': True,
        'large': True,
        'huge': True,
    }[depth]

    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
        'large': None,
        'huge': None,
    }[depth]

    model = VisionTransformer(img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size, depth=num_depth,
                             patch_size=patch_size,
                              num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop_path_rate=drop_path_ratio, drop_rate=drop_ratio, attn_drop_rate=attn_drop_rate,
                              predefined_head_dim=predefined_head_dim,
                              embed_dim=embed_dim, use_checkpointing=use_checkpointing, share_last=share_last)

    if pretrain:
        if not os.path.exists(pretrain_path):
            logger.info('{} is not found! Please check this path.'.format(pretrain_path))
            assert os.path.exists(pretrain_path)

        state_dict = paddle.load(pretrain_path)
        logger.info("Loading pretrained model from {}".format(pretrain_path))

        if 'model' in state_dict:
            state_dict = state_dict.pop('model')
        if 'state_dict' in state_dict:
            state_dict = state_dict.pop('state_dict')
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in pretrain_path:
                    logger.info("distill need to choose right cls token in the pth.")
                    v = paddle.concat([v[:, 0:1], v[:, 2:]], axis=1)
                v = resize_pos_embed(v, model.pos_embed, model.patch_embed.num_y, model.patch_embed.num_x)
            state_dict_new[k] = v
        model.set_state_dict(state_dict_new)
    return model
