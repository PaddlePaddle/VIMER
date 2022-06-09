import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from functools import partial
from models.modeling_finetune import PatchEmbed, DropPath, Mlp

zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)

def trunc_normal_(tensor, mean=0., std=1.):
    nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)  # TODO a=-std, b=std


'''
##########################
transformer encoder
##########################
'''
class VisionTransformerEncoder(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = self.create_parameter([1, 1, embed_dim], default_initializer=zeros_)
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter([1, num_patches + 1, embed_dim], default_initializer=zeros_)
        elif args.sincos_pos_emb:
            self.pos_embed = self.create_parameter([1, num_patches + 1, embed_dim], default_initializer=zeros_)
            self.pos_embed.set_value(self.build_2d_sincos_position_embedding(embed_dim, use_cls_token=True))
            self.pos_embed.stop_gradient = True  # fixed sin-cos embedding
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        # init for learnable absolute position embedding
        if self.pos_embed is not None and use_abs_pos_emb:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def build_2d_sincos_position_embedding(self, embed_dim=768, temperature=10000., use_cls_token=False):
        h, w = self.patch_embed.patch_shape
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = paddle.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = paddle.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = paddle.concat([paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h), paddle.cos(out_h)], axis=1)[None, :, :]

        if use_cls_token:
            pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
            pos_emb = paddle.concat([pe_token, pos_emb], axis=1)
        return pos_emb

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.set_value(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, dim = x.shape

        cls_tokens = self.cls_token.expand([batch_size, -1, -1])  # stole cls_tokens impl from Phil Wang, thanks

        # NOTE: unmasked embeddings
        x_unmasked = x[~bool_masked_pos].reshape([batch_size, -1, dim])  # [bs, _, c]
        x_unmasked = paddle.concat((cls_tokens, x_unmasked), axis=1)

        # NOTE: unmasked position embeddings
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand([batch_size, self.num_patches + 1, dim])
            pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape([batch_size, -1, dim])
            pos_embed_unmasked = paddle.concat((pos_embed[:, :1], pos_embed_unmasked), axis=1)
            x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.pos_drop(x_unmasked)

        for blk in self.blocks:
            x_unmasked = blk(x_unmasked, bool_masked_pos)

        x_unmasked = self.norm(x_unmasked)

        return x_unmasked

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        return x


'''
##########################
transformer regressor and decoder
##########################
'''
class VisionTransformerRegressorDecoder(nn.Layer):
    def __init__(self, patch_size=16, num_classes=8192, embed_dim=768, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, num_patches=196, init_std=0.02, args=None, patch_shape=(14, 14)):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.args = args

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.LayerList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, args.num_decoder_self_attention)]  # stochastic depth decay rule
        self.self_att_blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(args.num_decoder_self_attention)])

        self.norm = norm_layer(embed_dim)
        if args.num_decoder_self_attention > 0:
            self.norm2 = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        self.init_std = init_std

        trunc_normal_(self.head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.set_value(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.cross_attn.proj.weight, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight, layer_id + 1)

        for layer_id, layer in enumerate(self.self_att_blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                zeros_(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos):
        N_unmask_patch = x_unmasked.shape[1]
        N_mask_patch = x_masked.shape[1]
        '''
        latent contextual regressor
        '''
        for blk in self.blocks:
            x_masked = blk(x_masked, paddle.concat([x_unmasked, x_masked], axis=1), pos_embed_masked, paddle.concat([pos_embed_unmasked, pos_embed_masked], axis=1), bool_masked_pos)
        x_masked = self.norm(x_masked)
        latent_pred = x_masked
        '''
        decoder block
        '''
        if len(self.self_att_blocks) > 0:
            x_masked = x_masked + pos_embed_masked  # add pos embed
            for blk in self.self_att_blocks:
                x_masked = blk(x_masked)
            x_masked = self.norm2(x_masked)

        logits = self.head(x_masked)

        return logits, latent_pred

'''
Cross-attention block with bool_masked_pos argument
'''
class DecoderBlock(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()

        # NOTE: cross attention
        self.norm1_q_cross = norm_layer(dim)
        self.norm1_k_cross = norm_layer(dim)
        self.norm1_v_cross = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1_cross = self.create_parameter([dim], default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_2_cross = self.create_parameter([dim], default_initializer=nn.initializer.Constant(value=init_values))
        else:
            self.gamma_1_cross = self.create_parameter([dim], default_initializer=ones_)
            self.gamma_2_cross = self.create_parameter([dim], default_initializer=ones_)
            self.gamma_1_cross.stop_gradient = True
            self.gamma_2_cross.stop_gradient = True

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q_cross(x_q + pos_q),
         bool_masked_pos, k=self.norm1_k_cross(x_kv + pos_k), v=self.norm1_v_cross(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x

'''
Self-attention block with bool_masked_pos argument
'''
class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = self.create_parameter([dim], default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_2 = self.create_parameter([dim], default_initializer=nn.initializer.Constant(value=init_values))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, bool_masked_pos=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), bool_masked_pos))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), bool_masked_pos))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

'''
Attention with bool_masked_pos argument.
'''
class CrossAttention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias_attr=False)
        self.k = nn.Linear(dim, all_head_dim, bias_attr=False)
        self.v = nn.Linear(dim, all_head_dim, bias_attr=False)

        if qkv_bias:
            self.q_bias = self.create_parameter([all_head_dim], default_initializer=zeros_)
            # self.k_bias = self.create_parameter([all_head_dim], default_initializer=zeros_)
            self.v_bias = self.create_parameter([all_head_dim], default_initializer=zeros_)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, k=None, v=None):
        B, N, C = x.shape

        if k is None:
            k = x
            v = x
            N_k = N
            N_v = N
        else:
            N_k = k.shape[1]
            N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            v_bias = self.v_bias

        q = F.linear(x=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(x=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(x=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape([B, N, 1, self.num_heads, -1]).transpose([2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape([B, N_k, 1, self.num_heads, -1]).transpose([2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape([B, N_v, 1, self.num_heads, -1]).transpose([2, 0, 3, 1, 4]).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose([0, 1, 3, 2]))  # (B, N_head, N, N)

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter([all_head_dim], default_initializer=zeros_)
            self.v_bias = self.create_parameter([all_head_dim], default_initializer=zeros_)
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None):

        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            qkv_bias = paddle.concat((self.q_bias, k_bias, self.v_bias))

        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose([0, 1, 3, 2]))  # (B, N_head, N, N)

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


if __name__ == "__main__":
    from functools import partial

    import sys
    sys.path.insert(0, '/home/nieyang/cae_paddle')

    from packages.masking_generator import MaskingGenerator

    class ARGS:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = ARGS(
        sincos_pos_emb=True,
        num_decoder_self_attention=4,
    )

    encoder = VisionTransformerEncoder(
        # cae_base_patch16_224_8k_vocab
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        vocab_size=8192,
        drop_path_rate=0.,
        use_abs_pos_emb=False,
        init_values=0.1,
        args=args,
    )
    masked_position_generator = MaskingGenerator(
        (224 // 16, 224 // 16), num_masking_patches=75, min_num_patches=16)

    image = paddle.ones([2, 3, 224, 224])
    bool_masked_pos = paddle.to_tensor(masked_position_generator())
    bool_masked_pos = bool_masked_pos.unsqueeze(0).expand([2, -1, -1]).flatten(1).astype(paddle.bool)
    x_unmasked = encoder(image, bool_masked_pos)

    regressor_and_decoder = VisionTransformerRegressorDecoder(
        patch_size=16,
        num_classes=8192,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        qkv_bias=True,
        drop_path_rate=0.,
        init_values=0.1,
        num_patches=encoder.patch_embed.num_patches,
        args=args,
    )
