import math
import time
import paddle
import paddle.nn as nn
from functools import partial

from models.modeling_cae_modules import *


def trunc_normal_(tensor, mean=0., std=1.):
    nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)  # TODO a=-std, b=std


zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)
xavier_uniform_ = nn.initializer.XavierUniform()


class VisionTransformerForMaskedImageModeling(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, init_std=0.02, args=None, **kwargs):
        super().__init__()

        self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        # Forward the masked patch to the teacher network. The teacher network is the same as the student by default.
        self.teacher_network = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                use_abs_pos_emb=use_abs_pos_emb, init_std=init_std, args=args)

        # detach the teacher model
        for param in self.teacher_network.parameters():
            param.stop_gradient = True  # TODO check param.detach_()

        self.init_std = init_std
        self.args = args
        self.num_patches = self.encoder.patch_embed.num_patches

        self.regressor_and_decoder = VisionTransformerRegressorDecoder(patch_size=patch_size, num_classes=args.decoder_num_classes, embed_dim=args.decoder_embed_dim, depth=args.regressor_depth,
                 num_heads=args.decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=args.decoder_layer_scale_init_value, num_patches=self.num_patches, init_std=init_std, args=args)

        if args.decoder_embed_dim != embed_dim:
            self.encoder_to_decoder = nn.Linear(embed_dim, args.decoder_embed_dim, bias_attr=True)
            self.encoder_to_decoder_norm = norm_layer(args.decoder_embed_dim)
        else:
            self.encoder_to_decoder = None

        self.mask_token = self.create_parameter([1, 1, args.decoder_embed_dim], default_initializer=zeros_)
        trunc_normal_(self.mask_token, std=self.init_std)

        ### init the weight
        self.apply(self._init_weights)

        # copy the params from the student to teacher
        self._init_teacher()

    def _init_teacher(self):
        for t_param, s_param in zip(self.teacher_network.parameters(), self.encoder.parameters()):
            t_param.set_value(s_param.detach())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def _update_ema_variables(self, ema_decay):
        for t_param, s_param in zip(self.teacher_network.parameters(), self.encoder.parameters()):
            bias = s_param.detach() * (1 - ema_decay)
            t_param.scale_(ema_decay)
            t_param.add_(bias)

    def forward(self, x, bool_masked_pos, return_all_tokens=None):
        # x: [bs, 3, 224, 224]
        # bool_masked_pos: [bs, num_patch * num_patch]
        batch_size = x.shape[0]
        '''
        ##########################
        encoder
        ##########################
        '''
        x_unmasked = self.encoder(x, bool_masked_pos=bool_masked_pos)  # [bs, num_visible + 1, C_e]

        if self.encoder_to_decoder is not None:
            x_unmasked = self.encoder_to_decoder(x_unmasked)  # [64, 49, C_d]
            x_unmasked = self.encoder_to_decoder_norm(x_unmasked)
        '''
        Forward the teacher network
        '''
        with paddle.no_grad():
            latent_target = self.teacher_network(x, bool_masked_pos=(~bool_masked_pos))
            latent_target = latent_target[:, 1:, :]  # remove class token
            if self.encoder_to_decoder is not None:
                latent_target = self.encoder_to_decoder_norm(self.encoder_to_decoder(latent_target.detach()))

            self._update_ema_variables(self.args.dual_path_ema)
        '''
        ##########################
        latent contextual regressor
        ##########################
        '''
        b, num_visible_plus1, dim = x_unmasked.shape
        num_masked_patches = self.num_patches - (num_visible_plus1 - 1)  # number of masked patches

        # generate position embeddings.
        try:
            pos_embed = self.encoder.pos_embed.expand([batch_size, self.num_patches + 1, dim])
        except:
            pos_embed = self.encoder.build_2d_sincos_position_embedding(dim, use_cls_token=True).expand([batch_size, self.num_patches + 1, dim])

        # pos embed for class token.
        pos_cls_token = pos_embed[:, :1]
        ''' masked pos embed, no class token '''
        pos_embed_masked = pos_embed[:, 1:][bool_masked_pos].reshape([batch_size, -1, dim])
        ''' unmasked pos embed, class token is optional '''
        pos_embed_unmasked = pos_embed[:, 1:][~bool_masked_pos].reshape([batch_size, -1, dim])
        ''' remove class token '''
        x_unmasked = x_unmasked[:, 1:, :]
        ''' masked embedding '''
        x_masked = self.mask_token.expand([batch_size, num_masked_patches, -1])  # [b, num_masked, C_d]

        logits, latent_pred = self.regressor_and_decoder(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos)
        logits = logits.reshape([logits.shape[0] * logits.shape[1], logits.shape[2]])  # reshape to calculate loss

        return logits, latent_pred, latent_target


def cae_small_patch16_224_8k_vocab(**kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True), vocab_size=8192, **kwargs)
    return model


def cae_base_patch16_224_8k_vocab(**kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True), vocab_size=8192, **kwargs)
    return model


def cae_large_patch16_224_8k_vocab(**kwargs):
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6, weight_attr=True, bias_attr=True), vocab_size=8192, **kwargs)
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/nieyang/cae_paddle')

    from packages.masking_generator import MaskingGenerator

    class ARGS:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = ARGS(
        sincos_pos_emb=True, decoder_num_classes=8192, decoder_num_heads=12,
        regressor_depth=4, num_decoder_self_attention=4, decoder_layer_scale_init_value=0.1,
        decoder_embed_dim=768, dual_path_ema=0,)

    model = cae_base_patch16_224_8k_vocab(
        drop_path_rate=0., use_abs_pos_emb=False, init_values=0.1, args=args,)
    masked_position_generator = MaskingGenerator(
        (224 // 16, 224 // 16), num_masking_patches=75, min_num_patches=16)

    image = paddle.ones([2, 3, 224, 224])
    bool_masked_pos = paddle.to_tensor(masked_position_generator())
    bool_masked_pos = bool_masked_pos.unsqueeze(0).expand([2, -1, -1]).flatten(1).astype(paddle.bool)
    output = model(image, bool_masked_pos)
