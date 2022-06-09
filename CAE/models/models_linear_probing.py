import math
import numpy as np
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.modeling_finetune import to_2tuple, trunc_normal_, zeros_, ones_
from models.modeling_cae_modules import DecoderBlock, CrossAttention


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        **kwargs
    }


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter([all_head_dim],
                                                default_initializer=zeros_)
            self.v_bias = self.create_parameter([all_head_dim],
                                                default_initializer=zeros_)
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] -
                                          1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                [self.num_relative_distance, num_heads],
                default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h,
                                                   coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:,
                                                                    None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose([1, 2,
                                                         0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :,
                            0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                paddle.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(
                -1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                                 relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, bias_attr=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            qkv_bias = paddle.concat((self.q_bias, k_bias, self.v_bias))
        # qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads,
                           -1]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose([0, 1, 3, 2]))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              window_size=window_size,
                              attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if init_values > 0:
            self.gamma_1 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_1.stop_gradient = False
            self.gamma_2.stop_gradient = False
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0],
                            img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias_attr=True)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


class RelativePositionBias(nn.Layer):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads],
            default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h,
                                               coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2,
                                                     0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:,
                                1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                             relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww


def get_sinusoid_encoding_table(n_position, d_hid, token=False):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if token:
        sinusoid_table = np.concatenate(
            [sinusoid_table, np.zeros([1, d_hid])], dim=0)

    return paddle.to_tensor(sinusoid_table).unsqueeze(0)


'''
Decoder block with bool_masked_pos argument
'''


class DecoderBlockSimple(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None,
                 no_parameter=False):
        super().__init__()

        # NOTE: cross attention
        if no_parameter:
            self.norm1_q_cross = norm_layer(dim,
                                            weight_attr=False,
                                            bias_attr=False)
            self.norm1_k_cross = norm_layer(dim,
                                            weight_attr=False,
                                            bias_attr=False)
            self.norm1_v_cross = norm_layer(dim,
                                            weight_attr=False,
                                            bias_attr=False)
            self.norm2_cross = norm_layer(dim,
                                          weight_attr=False,
                                          bias_attr=False)
            self.cross_attn = CrossAttentionSimple(dim,
                                                   num_heads=num_heads,
                                                   qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   attn_drop=attn_drop,
                                                   proj_drop=drop,
                                                   window_size=window_size,
                                                   attn_head_dim=attn_head_dim)
        else:
            self.norm1_q_cross = norm_layer(dim)
            self.norm1_k_cross = norm_layer(dim)
            self.norm1_v_cross = norm_layer(dim)
            self.norm2_cross = norm_layer(dim)
            self.cross_attn = CrossAttention(dim,
                                             num_heads=num_heads,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             attn_drop=attn_drop,
                                             proj_drop=drop,
                                             window_size=window_size,
                                             attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q_cross(x_q + pos_q)
        x_k = self.norm1_k_cross(x_kv + pos_k)
        x_v = self.norm1_v_cross(x_kv)

        x = self.cross_attn(x_q,
                            bool_masked_pos,
                            rel_pos_bias=rel_pos_bias,
                            k=x_k,
                            v=x_v)

        return x


'''
Simple cross-attention. no parameter.
'''


class CrossAttentionSimple(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self,
                x,
                bool_masked_pos=None,
                rel_pos_bias=None,
                k=None,
                v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        q = x

        q = q.reshape([B, N, 1, self.num_heads, -1
                       ]).transpose([2, 0, 3, 1,
                                     4]).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape([B, N_k, 1, self.num_heads, -1
                       ]).transpose([2, 0, 3, 1,
                                     4]).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape([B, N_v, 1, self.num_heads, -1
                       ]).transpose([2, 0, 3, 1,
                                     4]).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose([-2, -1]))  # (B, N_head, N, N)
        attn = attn.softmax(axis=-1)
        x = (attn @ v).transpose([1, 2]).reshape([B, N, -1])

        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_mean_pooling=True,
                 init_scale=0.001,
                 lin_probe=False,
                 args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter([1, 1, embed_dim],
                                               default_initializer=zeros_)
        # self.mask_token = self.create_parameter([1, 1, embed_dim], default_initializer=zeros_)
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim], default_initializer=zeros_)
        elif args.sin_pos_emb:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim], default_initializer=zeros_)
            self.pos_embed.set_value(
                self.build_2d_sincos_position_embedding(embed_dim))
            self.pos_embed.stop_gradient = True  # fixed sin-cos embedding
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.LayerList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  init_values=init_values,
                  window_size=self.patch_embed.patch_shape
                  if use_rel_pos_bias else None) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)

        if self.use_mean_pooling:
            self.fc_norm = norm_layer(embed_dim,
                                      weight_attr=True,
                                      bias_attr=True)

        self.lin_probe = lin_probe
        # NOTE: batch norm
        self.args = args
        self.linear_type = args.linear_type
        if lin_probe:
            if args.linear_type != 'standard':
                if args.linear_type == 'attentive_no_parameter':
                    no_parameter = True
                else:
                    no_parameter = False

                self.linear_blocks = nn.LayerList([
                    DecoderBlockSimple(dim=embed_dim,
                                       num_heads=num_heads,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       drop_path=0,
                                       norm_layer=norm_layer,
                                       init_values=0,
                                       no_parameter=no_parameter)
                    for i in range(args.linear_depth)
                ])

                self.query_token = self.create_parameter(
                    [1, 1, embed_dim], default_initializer=zeros_)
                trunc_normal_(self.query_token, std=.02)

        self.head = nn.Linear(
            embed_dim, num_classes,
            bias_attr=True) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None and use_abs_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        zeros_(self.head.bias)
        self.apply(self._init_weights)

        # self.fix_init_weight()

    def build_2d_sincos_position_embedding(self,
                                           embed_dim=768,
                                           temperature=10000.):
        h, w = self.patch_embed.patch_shape
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = paddle.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = paddle.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = paddle.concat([
            paddle.sin(out_w),
            paddle.cos(out_w),
            paddle.sin(out_h),
            paddle.cos(out_h)
        ],
                                axis=1)[None, :, :]

        pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
        _, num_patches, _ = pos_emb.shape
        pos_embed = self.create_parameter(
            shape=[1, 1 + num_patches, embed_dim],
            default_initializer=nn.initializer.Assign(
                paddle.concat([pe_token, pos_emb], axis=1)))
        pos_embed.stop_gradient = True
        return pos_embed

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.scale(1 / (math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            try:
                zeros_(m.bias)
                ones_(m.weight)
            except:
                pass

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand(
            [batch_size, -1,
             -1])  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand([batch_size, -1, -1]).astype(
                    x.dtype).clone().detach()
            else:
                x = x + self.pos_embed.expand([batch_size, -1, -1]).astype(
                    x.dtype).clone().detach()

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        if self.linear_type == 'standard':
            if self.use_mean_pooling:
                x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
                outcome = self.fc_norm(x)
                return outcome
            else:
                return x[:, 0]
        else:
            query_tokens = self.query_token.expand([batch_size, -1, -1])
            key_value_pos_embed = self.pos_embed.expand(
                [batch_size, -1, -1]).astype(x.dtype).clone().detach()

            x = x + key_value_pos_embed
            for blk in self.linear_blocks:
                query_tokens = blk(query_tokens,
                                   x,
                                   0,
                                   0,
                                   bool_masked_pos=None,
                                   rel_pos_bias=None)

            return query_tokens[:, 0, :]

    def forward(self, x, is_train=True):
        x = self.forward_features(x, is_train)
        x = self.head(x)
        return x


def cae_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def cae_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def cae_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=384,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def cae_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def cae_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=384,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model


def cae_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=512,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm,
                                                 epsilon=1e-6,
                                                 weight_attr=True,
                                                 bias_attr=True),
                              **kwargs)
    model.default_cfg = _cfg()
    return model
