"""super_module/multihead_super.py
"""
import paddle
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from .Linear_super import LinearSuper
from .qkv_super import QkvSuper

BIAS_LR_FACTOR=2.0

class AttentionSuper(nn.Layer):
    """支持super_embed_dim == 64 * num_heads的vit结构
    """
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., normalization = False, relative_position = False, predefined_head_dim=64, 
                 num_patches = None, max_relative_position=14, scale=False, change_qkv = False, learning_scale=None):
        super().__init__()
        self.num_heads = num_heads
        if num_heads * predefined_head_dim > super_embed_dim:
            qkv_super_embed_dim = num_heads * predefined_head_dim
        else:
            qkv_super_embed_dim = super_embed_dim
        # head_dim = super_embed_dim // num_heads
        head_dim = qkv_super_embed_dim // num_heads
        assert head_dim == predefined_head_dim
        self.predefined_head_dim = predefined_head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = qkv_super_embed_dim

        self.fc_scale = scale
        self.change_qkv = change_qkv
        lr_factor = 1.0 if learning_scale is None else float(learning_scale)
        if change_qkv:
            self.qkv = QkvSuper(super_embed_dim, 3 * qkv_super_embed_dim, 
            bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR * lr_factor) if qkv_bias else False, 
            weight_attr=ParamAttr(learning_rate=lr_factor),
            )
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * qkv_super_embed_dim, 
            bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR * lr_factor) if qkv_bias else False, 
            weight_attr=ParamAttr(learning_rate=lr_factor),
            )
            pass

        self.relative_position = relative_position
        if self.relative_position:
            # self.rel_pos_embed_k = RelativePosition2D_super(super_embed_dim //num_heads, max_relative_position)
            # self.rel_pos_embed_v = RelativePosition2D_super(super_embed_dim //num_heads, max_relative_position)
            pass

        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None

        self.proj = LinearSuper(qkv_super_embed_dim, super_embed_dim, bias_attr=ParamAttr(learning_rate=BIAS_LR_FACTOR))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            # self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5
            self.sample_scale = self.scale

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5
            assert self.sample_qk_embed_dim // self.sample_num_heads == self.predefined_head_dim

        # print('self.change_qkv is {}'.format(self.change_qkv))
        # print('self.sample_qk_embed_dim and self.sample_scale are {} and {}'.format(self.sample_qk_embed_dim, self.sample_scale))

        self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3 * self.sample_qk_embed_dim)
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            # self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
            # self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim // sample_num_heads)
            pass

    def calc_sampled_param_num(self):
        return 0

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += self.max_relative_position * sequence_length * \
                sequence_length + sequence_length * sequence_length / 2.0
            total_flops += self.max_relative_position * sequence_length * \
                sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0
        return total_flops

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, N, 3, self.sample_num_heads, -1)).transpose((2, 0, 3, 1, 4))
        # print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]   #  (cannot use tensor as tuple)

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.sample_scale

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, -1))

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
