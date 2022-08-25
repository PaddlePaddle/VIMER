"""fusion_transformer_base
"""
from collections import OrderedDict
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from paddlenlp.transformers import BertModel
import pdb

# import os
# import torch
# from paddlenlp.transformers import BertModel
__all__ = ["fusion_base_patch16_224", "fusion_base_patch16_384"]

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


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
    """to_2tuple"""
    return tuple([x] * 2)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

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

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
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


def masked_fill(x, mask, value):
    """masked_fill"""
    y = paddle.full(paddle.shape(x), value, x.dtype)
    return paddle.where(mask, y, x)


class Attention(nn.Layer):
    """Attention"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """forward"""
        N = paddle.shape(x)[1]
        C = paddle.shape(x)[2]
        qkv = (
            self.qkv(x)
            .reshape((-1, N, 3, self.num_heads, C // self.num_heads))
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        if mask is not None:
            mask = mask.astype("bool")
            attn = masked_fill(attn, ~mask[:, None, None, :], float("-inf"))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Layer):
    """Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        """forward"""
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        epsilon=1e-6,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flatten = flatten
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = (
            norm_layer(embed_dim, eposilon=epsilon) if norm_layer else Identity()
        )

    def forward(self, x):
        """forward"""
        # B, C, H, W = x.shape
        B = paddle.shape(x)[0]
        C = paddle.shape(x)[1]
        H = paddle.shape(x)[2]
        W = paddle.shape(x)[3]
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), "Input image size ({}*{}) doesn't match model ({}*{}).".format(
            H, W, self.img_size[0], self.img_size[1]
        )

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose((0, 2, 1))
        x = self.norm(x)
        return x


class VisionStFusionTransformer(nn.Layer):
    """Vision Transformer with support for patch input"""

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
        config=None,
        scene_text_max_len=15,
        scene_text_depth=12,
        fusion_depth=4,
        scene_text_feature_size=768,
        scene_text_lacation_size=5,
        epsilon=1e-6,
        scene_text_model_dir="./bert-base-uncased/",
        **kwargs
    ):
        super().__init__()
        drop_rate = drop_rate if config is None else config["drop_rate"]

        self.class_num = class_num

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_
        )
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_
        )
        self.add_parameter("cls_token", self.cls_token)
        self.token_type_embeddings = nn.Embedding(3, embed_dim)
        self.fusion_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_
        )
        self.add_parameter("fusion_token", self.fusion_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)

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
                    epsilon=epsilon,
                )
                for i in range(depth)
            ]
        )

        self.scene_text_fusion_blocks = nn.LayerList(
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
                    epsilon=epsilon,
                )
                for i in range(fusion_depth)
            ]
        )

        self.scene_text_model = BertModel.from_pretrained(scene_text_model_dir)

        self.fusion_depth = fusion_depth

        self.norm = norm_layer(embed_dim, epsilon=epsilon)

        self.fusion_pos_embed = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_
        )
        self.add_parameter("fusion_pos_embed", self.fusion_pos_embed)
        self.ocr_pos_projection = nn.Linear(4, embed_dim)
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.fusion_pos_embed)
        trunc_normal_(self.fusion_token)
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

    def pre_encoder_vision(self, x):
        """pre_encoder_vision"""
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((paddle.shape(x)[0], -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        token_type_embeddings_image = self.token_type_embeddings(
            paddle.zeros(shape=[paddle.shape(x)[0], paddle.shape(x)[1]], dtype="int64")
        )
        x = x + token_type_embeddings_image + self.pos_embed
        img_features = self.pos_drop(x)
        for blk in self.blocks[: -1 * self.fusion_depth]:
            img_features = blk(img_features, mask=None)

        return img_features, x

    def pre_encoder_scene_text(
        self,
        input_ids=None,  # scene_text_ids
        attention_mask=None,  # scene_text_attention_mask
        token_type_ids=None,  # scene_text_token_type_ids
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """pre_encoder_scene_text"""

        extended_attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
            paddle.get_default_dtype()
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4

        scene_text_features = self.scene_text_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        for layer_module in self.scene_text_model.encoder.layers[
            : -1 * self.fusion_depth
        ]:  # BERT first several layers
            layer_outputs = layer_module(
                scene_text_features,
                src_mask=extended_attention_mask,
            )
            scene_text_features = layer_outputs

        return scene_text_features  # [bs, seq_length, scene_text_feature_size]

    def forward_features(
        self,
        x,
        scene_text_ids=None,
        scene_text_attention_mask=None,
        scene_text_token_type_ids=None,
        scene_text_pos=None,
    ):
        """forward_features"""
        img_features, x = self.pre_encoder_vision(x)
        scene_text_features = self.pre_encoder_scene_text(
            input_ids=scene_text_ids,
            attention_mask=scene_text_attention_mask,
            token_type_ids=scene_text_token_type_ids,
        )

        if scene_text_pos is not None:
            scene_text_pos = self.ocr_pos_projection(scene_text_pos)
            scene_text_features = scene_text_features + scene_text_pos

        fusion_token = self.fusion_token.expand((paddle.shape(x)[0], -1, -1))
        fusion_token_token_type_embed = self.token_type_embeddings(
            paddle.full(
                shape=[paddle.shape(fusion_token)[0], paddle.shape(fusion_token)[1]],
                fill_value=2,
                dtype="int64",
            )
        )
        fusion_token = (
            fusion_token + fusion_token_token_type_embed + self.fusion_pos_embed
        )
        image_fusion_attention_mask = paddle.ones(
            shape=[paddle.shape(x)[0], paddle.shape(x)[1] + 1], dtype="int64"
        )
        fusion_attention_mask = paddle.ones(shape=[paddle.shape(x)[0], 1])
        scene_text_fusion_attention_mask = paddle.concat(
            (fusion_attention_mask, scene_text_attention_mask), axis=1
        )
        extended_scenet_text_fusion_attention_mask = (
            scene_text_fusion_attention_mask.unsqueeze(axis=[1, 2]).astype(
                paddle.get_default_dtype()
            )
        )
        extended_scenet_text_fusion_attention_mask = (
            1.0 - extended_scenet_text_fusion_attention_mask
        ) * -1e4

        for i in range(self.fusion_depth):
            image_fusion_x = paddle.concat((img_features, fusion_token), axis=1)
            image_fusion_x = self.blocks[i - self.fusion_depth](
                image_fusion_x, mask=image_fusion_attention_mask
            )
            scene_text_fusion_x = paddle.concat(
                (fusion_token, scene_text_features), axis=1
            )

            layer_outputs = self.scene_text_model.encoder.layers[i - self.fusion_depth](
                scene_text_fusion_x,
                src_mask=extended_scenet_text_fusion_attention_mask,
            )

            scene_text_fusion_x = layer_outputs
            fusion_token = image_fusion_x[:, -1, :] + scene_text_fusion_x[:, 0, :]
            fusion_token = fusion_token.unsqueeze(1)
            img_features = image_fusion_x[:, :-1, :]
            scene_text_features = scene_text_fusion_x[:, 1:, :]

        image_fusion_x = self.norm(image_fusion_x)
        scene_text_fusion_x = self.norm(scene_text_fusion_x)
        fusion_token = image_fusion_x[:, -1, :] + scene_text_fusion_x[:, 0, :]
        img_features = image_fusion_x[:, :-1, :]
        scene_text_features = scene_text_fusion_x[:, 1:, :]
        image_cls = img_features[:, 0, :]
        scene_text_cls = scene_text_features[:, 0, :]
        return image_cls, scene_text_cls, fusion_token

    def forward(
        self,
        x,
        scene_text_ids=None,
        scene_text_attention_mask=None,
        scene_text_token_type_ids=None,
        scene_text_pos=None,
    ):
        """forward"""
        x = self.forward_features(
            x,
            scene_text_ids=scene_text_ids,
            scene_text_attention_mask=scene_text_attention_mask,
            scene_text_token_type_ids=scene_text_token_type_ids,
            scene_text_pos=scene_text_pos,
        )
        return x


def fusion_base_patch16_224(fusion_depth=4, **kwargs):
    """fusion_base_patch16_224"""
    model = VisionStFusionTransformer(
        img_size=224,
        num_classes=1000,
        representation_size=None,
        fusion_depth=fusion_depth,
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


def fusion_base_patch16_384(fusion_depth=4, **kwargs):
    """fusion_base_patch16_384"""
    model = VisionStFusionTransformer(
        img_size=384,
        num_classes=1000,
        representation_size=None,
        fusion_depth=fusion_depth,
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
