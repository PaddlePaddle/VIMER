# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""vision_transformer
"""
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
import os
from functools import partial
from paddle import ParamAttr
from paddle.nn.initializer import Assign
__all__=['vit_deit_base_patch16_224', 'vit_deit_base_patch16_384', 'deit_base_patch16_224', 'deit_base_patch16_384']

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def _cfg(url=""):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    }


def to_2tuple(x):
    """to_2tuple
    """
    return tuple([x] * 2)


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
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """forward
        """
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    """Identity
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        """forward
        """
        return input


class Mlp(nn.Layer):
    """Mlp
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """forward
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def masked_fill(x, mask, value):
    """masked_fill
    """
    y = paddle.full(paddle.shape(x), value, x.dtype)
    return paddle.where(mask, y, x)


class Attention(nn.Layer):
    """Attention
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """forward
        """
        N = paddle.shape(x)[1]
        C = paddle.shape(x)[2]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        if mask is not None:
            mask = mask.astype('bool')
            attn = masked_fill(attn, ~mask[:, None, None, :], float('-inf'))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Layer):
    """Block
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 epsilon=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, mask=None):
        """forward
        """
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, 
                img_size=224,
                patch_size=16,
                in_chans=3,
                embed_dim=768,
                norm_layer=None,
                epsilon=1e-6,
                flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim, eposilon=epsilon) if norm_layer else Identity()


    def forward(self, x):
        """forward
        """
        #B, C, H, W = x.shape
        B = paddle.shape(x)[0]
        C = paddle.shape(x)[1]
        H = paddle.shape(x)[2]
        W = paddle.shape(x)[3]
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({}*{}) doesn't match model ({}*{}).".format(H, W, self.img_size[0], self.img_size[1])

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose((0, 2, 1))
        x = self.norm(x)
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        class_num=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        add_norm_before_transformer=False,
        no_patch_embed_bias=False,
        config=None,
        ocr_max_len=15,
        epsilon=1e-6,
        **kwargs
    ):
        super().__init__()
        drop_rate=drop_rate if config is None else config['drop_rate']

        self.class_num = class_num

        self.num_features = self.embed_dim = embed_dim
        self.add_norm_before_transformer = add_norm_before_transformer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.token_type_embeddings = nn.Embedding(2, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim, epsilon=epsilon)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim, epsilon=epsilon)
        
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        """forward_features
        """
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((paddle.shape(x)[0], -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        token_type_embeddings_image = self.token_type_embeddings(
                paddle.zeros(shape=[paddle.shape(x)[0], paddle.shape(x)[1]], dtype='int64'))
        x = x + token_type_embeddings_image + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        """forward
        """
        x = self.forward_features(x)
        return x[:, 0]


class _VisionTransformer(nn.Layer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or nn.LayerNorm

        if hybrid_backbone is not None:
            pass
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(
            attr=ParamAttr(
                name="cls_token",
                initializer=Assign(paddle.zeros(shape=(1, 1, embed_dim))),
            ),
            shape=(1, 1, embed_dim),
        )
        self.pos_embed = self.create_parameter(
            attr=ParamAttr(
                name="pos_embed",
                initializer=Assign(paddle.zeros(shape=(1, num_patches + 1, embed_dim))),
            ),
            shape=(1, num_patches + 1, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x for x in np.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else Identity()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        B = paddle.shape(x)[0]
        C = paddle.shape(x)[2]

        cls_tokens = paddle.fluid.layers.expand(self.cls_token, (B, 1, 1))
        x = paddle.concat((cls_tokens, x), axis=1)  # [B, 50, 768]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = nn.functional.normalize(x, p=2, axis=1)
        return x


def vit_deit_base_patch16_224( **kwargs):
    """vit_deit_base_patch16_224
    """
    model = VisionTransformer(
        img_size=224,
        num_classes=1000,
        representation_size=None,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def vit_deit_base_patch16_384(**kwargs):
    """vit_deit_base_patch16_384
    """
    model = VisionTransformer(
        img_size=384,
        num_classes=1000,
        representation_size=None,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def deit_base_patch16_224():
    backbone = _VisionTransformer(
        img_size=224,
        drop_path_rate=0.1,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=-1,
    )
    backbone.default_cfg = _cfg()

    return backbone

def deit_base_patch16_384():
    backbone = _VisionTransformer(
        img_size=384,
        drop_path_rate=0.1,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        num_classes=-1,
    )
    backbone.default_cfg = _cfg()

    return backbone

