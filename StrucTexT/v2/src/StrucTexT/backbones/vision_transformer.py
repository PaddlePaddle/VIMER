""" vision_transformer """
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

from collections import Callable
import os
import sys
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

def load_dygraph_pretrain(model, path=None):
    """load_dygraph_pretrain"""
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    if os.path.isdir(path):
        param_state_dict = paddle.load(path)
    if os.path.exists(path + '.pdparams'):
        param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return

MODEL_URLS = {
    "ViT_small_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "ViT_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "ViT_base_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
    "ViT_base_patch32_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
    "ViT_large_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
    "ViT_large_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams",
    "ViT_large_patch32_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams",
    "ViT_huge_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_huge_patch16_224_pretrained.pdparams",
    "ViT_huge_patch32_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_huge_patch32_384_pretrained.pdparams"
}

__all__ = ['VisionTransformer']

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def to_2tuple(x):
    """to_2tuple"""
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
        """forward"""
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    """Identity"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        """forward"""
        return input


class Mlp(nn.Layer):
    """Mlp"""
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer="nn.GELU",
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = eval(act_layer)()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """forward"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    """Attention"""
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

    def forward(self, x):
        """forward"""
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    """Block"""
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer='nn.GELU',
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        """forward"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """forward"""
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size (%d*%d) doesn't match model (%d*%d)." % (H, W, self.img_size[0], self.img_size[1])

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__()
        self.class_num = class_num

        self.num_features = self.embed_dim = embed_dim

        self.out_channels = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), \
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=drop_rate)

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

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              class_num) if class_num > 0 else Identity()

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        def gather(x, axis, index):
            """ gather """
            """
            bs = paddle.shape(x)[0]
            sample_num = paddle.shape(index)[-1]
            bs_index = paddle.arange(bs).unsqueeze(1).tile((1, sample_num))
            index = paddle.stack([bs_index, index], axis=-1).reshape((-1, 2))
            x = paddle.gather_nd(x, index).reshape((bs, sample_num, -1))
            """
            x_slices = []
            for bi in range(paddle.shape(x)[0]):
                index_slice = paddle.slice(index, axes=[0], starts=[bi], ends=[bi + 1])
                x_slice = paddle.slice(x, axes=[0], starts=[bi], ends=[bi + 1])
                x_slices.append(paddle.gather(x_slice, axis=axis, index=index_slice.squeeze(axis=0)))
            x = paddle.concat(x_slices, axis=0)
            return x

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        #noise = paddle.rand(shape=[N, L])  # noise in [0, 1]
        # sort noise for each sample
        #ids_shuffle = paddle.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        #ids_restore = paddle.argsort(ids_shuffle, axis=1)

        noise = np.random.rand(N, L) # noise in [0, 1]
        ids_shuffle = np.argsort(noise, axis=1)
        ids_restore = np.argsort(ids_shuffle, axis=1)

        ids_shuffle = paddle.to_tensor(ids_shuffle, dtype='int64')
        ids_restore = paddle.to_tensor(ids_restore, dtype='int64')

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = gather(x, 1, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = paddle.ones(shape=[N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = gather(mask, 1, ids_restore)


        return x_masked, mask, ids_restore

    def forward(self, x, **kwargs):
        """forward"""
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        # mask branch
        mask_ratio = kwargs.get('mask_ratio', 0.0)
        if mask_ratio > 0.0:
            x = x + self.pos_embed[:, 1:, :]
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = self.cls_token.expand((B, -1, -1))
            x = paddle.concat((cls_tokens, x), axis=1)
            x = self.pos_drop(x)
            out = []
            for blk in self.blocks:
                x = blk(x)
                out.append(x)
            x = self.norm(x)
            cls, _ = x.split([1, -1], axis=1)
            return {'out': x,
                    'additional_info': {
                        'all_feats': out,
                        'cls_feats': cls,
                        'mask': mask,
                        'ids_restore': ids_restore}
                   }
        else:
            cls_tokens = self.cls_token.expand((B, -1, -1))
            x = paddle.concat((cls_tokens, x), axis=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            out = []
            for blk in self.blocks:
                x = blk(x)
                out.append(x)
            x = self.norm(x)
            cls, _ = x.split([1, -1], axis=1)
            return {'out': x,
                    'additional_info': {
                        'cls_feats': cls,
                        'all_feats': out}
                   }


def _load_pretrained(pretrained, model, model_url=None, use_ssld=False):
    pass


def ViT_small_patch16_224(pretrained=False,
                          use_ssld=False,
                          **kwargs):
    """ViT_small_patch16_224"""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768 ** -0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_small_patch16_224"],
        use_ssld=use_ssld)
    return model


def ViT_base_patch16_224(pretrained=False,
                         use_ssld=False,
                         **kwargs):
    """ViT_base_patch16_224"""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def ViT_base_patch16_384(pretrained=False,
                         use_ssld=False,
                         **kwargs):
    """ViT_base_patch16_384"""
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_base_patch16_384"],
        use_ssld=use_ssld)
    return model


def ViT_base_patch32_384(pretrained=False,
                         use_ssld=False,
                         **kwargs):
    """ViT_base_patch32_384"""
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_base_patch32_384"],
        use_ssld=use_ssld)
    return model


def ViT_large_patch16_224(pretrained=False,
                          use_ssld=False,
                          **kwargs):
    """ViT_large_patch16_224"""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_large_patch16_224"],
        use_ssld=use_ssld)
    return model


def ViT_large_patch16_384(pretrained=False,
                          use_ssld=False,
                          **kwargs):
    """ViT_large_patch16_384"""
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_large_patch16_384"],
        use_ssld=use_ssld)
    return model


def ViT_large_patch32_384(pretrained=False,
                          use_ssld=False,
                          **kwargs):
    """ViT_large_patch32_384"""
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_large_patch32_384"],
        use_ssld=use_ssld)
    return model


def ViT_huge_patch16_224(pretrained=False,
                         use_ssld=False,
                         **kwargs):
    """ViT_huge_patch16_224"""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_huge_patch16_224"],
        use_ssld=use_ssld)
    return model


def ViT_huge_patch32_384(pretrained=False,
                         use_ssld=False,
                         **kwargs):
    """ViT_huge_patch32_384"""
    model = VisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["ViT_huge_patch32_384"],
        use_ssld=use_ssld)
    return model
