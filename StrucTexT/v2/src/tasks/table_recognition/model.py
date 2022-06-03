""" table structext recognization """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import json
import copy
import paddle as P
import numpy as np
from paddle import nn
from paddle import ParamAttr
from paddle.nn import functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from src.StrucTexT.arch.base_model import Encoder
from src.StrucTexT.ernie import append_name, build_linear, build_ln
from src.StrucTexT.backbones.resnet_vd import ConvBNLayer
from paddle.vision.ops import roi_align


def get_roi_features(input,  # N, C, H, W
                     rois,  # num_rois, 4, num_rois = batch_size * max_token_len
                     proposal_h=8,
                     proposal_w=64,
                     spatial_scale=0.0625,  # downsampling scale
                     rois_num=None):
    """get_roi_features"""
    if input.ndim == 3:
        input = input.unsqueeze(0)
    if rois.ndim == 2:
        rois = rois.unsqueeze(0)
    if rois_num is None:
        rois_num = P.ones((rois.shape[0],), dtype='int32') * rois.shape[1]
        rois = P.flatten(rois, stop_axis=1)
    rois = P.cast(rois, 'float32')
    out = roi_align(
        input,
        rois,
        output_size=(proposal_h, proposal_w),
        spatial_scale=spatial_scale,
        rois_num=rois_num)
    return out


class Model(Encoder):
    def __init__(self, config, name=''):
        super(Model, self).__init__(config, name=name)
        self.config = config['task_module']

        in_channels = config['in_channels']
        self.input_size = in_channels[-1] if isinstance(in_channels, list) else in_channels
        self.hidden_size = config['hidden_size']
        self.max_elem_length = config['max_elem_length']
        self.elem_num = 2
        self.encoder = PositionalEncoding(d_model=self.input_size, dropout=0.2)
        decoder_params = dict(self_attn=dict(
            headers=8,
            d_model=self.input_size,
            dropout=0.),
            src_attn=dict(
                headers=8,
                d_model=self.input_size,
                dropout=0.),
            feed_forward=dict(
                d_model=self.input_size,
                d_ff=2024,
                dropout=0.),
            size=self.input_size,
            dropout=0.)
        self.row_decoder = RowtDecoder(N=3, decoder=decoder_params,
                d_model=self.input_size)
        self.col_decoder = ColtDecoder(N=3, decoder=decoder_params,
                d_model=self.input_size)
        self.link_decoder = LinktDecoder(N=3, decoder=decoder_params,
                d_model=self.input_size)

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        image = input_data['image']
        targets = input_data.get('targets')
        eval_mode = input_data.get('eval_mode', False)
        enc_out = super(Model, self).forward([image, None])
        enc_final = enc_out['additional_info']['image_feat']
        fea = enc_final['out']
        batch_size = fea.shape[0]

        if len(fea.shape) == 3:
            pass
        else:
            last_shape = int(np.prod(fea.shape[2:]))  # gry added
            fea = paddle.reshape(fea, [fea.shape[0], fea.shape[1], last_shape])
            fea = fea.transpose([0, 2, 1])  # (NTC)(batch, width, channels)

        out_enc = self.encoder(fea)
        batch_size = out_enc.shape[0]
        row_index = paddle.reshape(paddle.to_tensor(list(range(500))), (1, -1))
        row_index = paddle.tile(row_index, (batch_size, 1))
        col_index = paddle.reshape(paddle.to_tensor(list(range(250))), (1, -1))
        col_index = paddle.tile(col_index, (batch_size, 1))

        row_probs, row_m_probs, row_features = self.row_decoder(inputs, out_enc, row_index)
        col_probs, col_m_probs, col_features = self.col_decoder(inputs, out_enc, col_index)
        link_probs = self.link_decoder(inputs, row_features, col_features)
        link_up = link_probs[:, 0:1, :, :]
        link_down = link_probs[:, 1:2, :, :]
        link_left = link_probs[:, 2:3, :, :]
        link_right = link_probs[:, 3:4, :, :]

        return {'row_probs': row_probs,
                'col_probs': col_probs,
                'row_m_probs': row_m_probs,
                'col_m_probs': col_m_probs,
                'link_up': link_up,
                'link_down': link_down,
                'link_left': link_left,
                'link_right': link_right}


class PositionalEncoding(nn.Layer):
    def __init__(self, d_model, dropout=0., max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = paddle.zeros((max_len, d_model))
        position = paddle.arange(0, max_len).unsqueeze(1).astype('float32')
        div_term = paddle.exp(paddle.arange(0, d_model, 2).astype('float32') * -math.log(10000.0) / d_model)
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        # if len(feat.shape) > 3:
        #     b, c, h, w = feat.shape
        #     feat = feat.view(b, c, h*w) # flatten 2D feature map
        #     feat = feat.permute((0, 2, 1))
        feat = feat + self.pe[:, :feat.shape[1]]  # pe 1*5000*512
        return self.dropout(feat)


class PositionalEncoding2D(nn.Layer):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, d_model, 2).astype('float32') *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose([1, 0, 2])
        self.register_buffer('pe', pe)

        self.avg_pool_1 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1.weight.data.fill_(1.)
        self.avg_pool_2 = nn.AdaptiveAvgPool2D((1, 1))
        self.linear2 = nn.Linear(d_model, d_model)
        # self.linear2.weight.data.fill_(1.)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # print(x.shape)
        w_pe = self.pe[:x.shape[-1], :]
        w1 = self.linear1(self.avg_pool_1(x).squeeze()).unsqueeze(0)
        w_pe = w_pe * w1
        w_pe = w_pe.transpose([1, 2, 0])
        w_pe = w_pe.unsqueeze(2)

        h_pe = self.pe[:x.shape[-2], :]
        w2 = self.linear2(self.avg_pool_2(x).squeeze()).unsqueeze(0)
        h_pe = h_pe * w2
        h_pe = h_pe.transpose([1, 2, 0])
        h_pe = h_pe.unsqueeze(3)

        x = x + w_pe + h_pe
        # print(x.shape)
        x = x.reshape(
            [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).transpose(
            [0, 2, 1])

        # print(x.shape)
        return self.dropout(x)


class Embeddings(nn.Layer):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N):
    """ Produce N identical layers """
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


class SubLayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # tmp = self.norm(x)
        # tmp = sublayer(tmp)
        tmp = sublayer(self.norm(x))
        if isinstance(tmp, tuple):
            return x + self.dropout(tmp[0]), tmp[1]
        return x + self.dropout(tmp), None


class FeedForward(nn.Layer):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def self_attention(query, key, value, mask=None, dropout=None, pred_layer=False):
    """
    Compute 'Scale Dot Product Attention'
    """

    d_k = value.shape[-1]
    score = paddle.matmul(query, key.transpose([0, 1, 3, 2]) / math.sqrt(d_k))
    if mask is not None:
        # pass
        # score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        # score = score.masked_fill(mask == 0, -6.55e4) # for fp16
        # score = score[mask == 0] = -6.55e4
        tmp = paddle.ones(shape=score.shape, dtype='float32') * -1e9
        mask = paddle.tile(mask, repeat_times=[1, score.shape[1], 1, 1])
        score = paddle.where(mask == 0, tmp, score)
    if not pred_layer:
        p_attn = F.softmax(score, axis=-1)
    else:
        p_attn = score
    if dropout is not None:
        p_attn = dropout(p_attn)
    return paddle.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Layer):

    def __init__(self, headers, d_model, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, pred_layer=False):
        nbatches = query.shape[0]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).reshape([nbatches, -1, self.headers, self.d_k]).transpose([0, 2, 1, 3])
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout, pred_layer=pred_layer)
        x = x.transpose([0, 2, 1, 3]).reshape([nbatches, -1, self.headers * self.d_k])
        return self.linears[-1](x), self.attn


class DecoderLayer(nn.Layer):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x, _ = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x, attn = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)[0], attn


class RowtDecoder(nn.Layer):
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 ):
        super(RowtDecoder, self).__init__()
        self.max_row = 500
        self.embedding = Embeddings(d_model=d_model, vocab=self.max_row)
        self.layers = clones(DecoderLayer(**decoder), N - 1)
        self.add_layers = clones(DecoderLayer(**decoder), 1)
        self.row_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid())
        self.row_m = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(decoder['size'])

    def decode(self, image_fea, feature, cell_index, src_mask, tgt_mask):
        image_fea = F.upsample(image_fea, size=[500,250], mode="bilinear", align_mode=1)
        image_fea = paddle.transpose(image_fea, [0, 2, 3, 1])
        row_fea = paddle.mean(image_fea, axis = 2)
        x = self.embedding(cell_index)
        x = x + row_fea
        for i, layer in enumerate(self.layers):
            x, atten = layer(x, feature, src_mask, tgt_mask)

        for layer in self.add_layers:
            x, _ = layer(x, feature, src_mask, tgt_mask)
        x = self.norm(x)
        rows = self.row_fc(x)
        rows_m = self.row_m(x)
        return rows, rows_m, x

    def forward(self, image_fea, out_enc, cell_index):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        src_mask = None
        tgt_mask = None
        return self.decode(image_fea, out_enc, cell_index, src_mask, tgt_mask)


class ColtDecoder(nn.Layer):
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 ):
        super(ColtDecoder, self).__init__()
        self.max_col = 250
        self.layers = clones(DecoderLayer(**decoder), N - 1)
        self.embedding = Embeddings(d_model=d_model, vocab=self.max_col)
        self.add_layers = clones(DecoderLayer(**decoder), 1)
        self.col_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid())
        self.col_m = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(decoder['size'])

    def decode(self, image_fea, feature, cell_index, src_mask, tgt_mask):

        image_fea = F.upsample(image_fea, size=[500,250], mode="bilinear", align_mode=1)
        image_fea = paddle.transpose(image_fea, [0, 2, 3, 1])
        col_fea = paddle.mean(image_fea, axis = 1)
        x = self.embedding(cell_index)
        for i, layer in enumerate(self.layers):
            x, atten = layer(x, feature, src_mask, tgt_mask)

        for layer in self.add_layers:
            x, _ = layer(x, feature, src_mask, tgt_mask)
        x = self.norm(x)
        cols = self.col_fc(x)
        cols_m = self.col_m(x)
        return cols, cols_m, x

    def forward(self, image_fea, out_enc, cell_index):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        src_mask = None
        tgt_mask = None
        return self.decode(image_fea, out_enc, cell_index, src_mask, tgt_mask)


class LinktDecoder(nn.Layer):
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 ):
        super(LinktDecoder, self).__init__()
        self.max_row = 500
        self.row_encoder = clones(DecoderLayer(**decoder), N)
        self.col_encoder = clones(DecoderLayer(**decoder), N)
        self.corner_encoder = MultiHeadAttention(headers = 1, d_model = d_model, dropout = 0.2)   #(self, headers, d_model, dropout):

        self.corner_fc = nn.Sequential(
            nn.Conv2D(d_model * 2, d_model, 3,  padding=1),
            nn.ReLU(),
            nn.Conv2D(d_model, d_model // 4, 3,  padding=1),
            nn.ReLU(),
            nn.Conv2D(d_model // 4, 4, 1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm(decoder['size'])

    def decode(self, img_features, row_features, col_features, row_mask, col_mask):
        input_row_features = row_features
        input_col_features = col_features
        for i, layer in enumerate(self.row_encoder):
            row_features_1, _ = layer(row_features, input_col_features, row_mask, col_mask)
            row_features_2, _ = layer(row_features, input_row_features, row_mask, col_mask)
            row_features = row_features_1 + row_features_2

        for i, layer in enumerate(self.col_encoder):
            col_features_1, _ = layer(col_features, input_row_features, row_mask, col_mask)
            col_features_2, _ = layer(col_features, input_col_features, row_mask, col_mask)
            col_features = col_features_1 + col_features_2

        row_features = paddle.transpose(paddle.tile(paddle.unsqueeze(row_features, 2), [1, 1, 250, 1]), [0, 3, 1, 2])
        col_features = paddle.transpose(paddle.tile(paddle.unsqueeze(col_features, 1), [1, 500, 1, 1]), [0, 3, 1, 2])
        corner_features = self.norm(row_features + col_features)
        img_features = F.upsample(img_features, size=[500,250], mode="bilinear", align_mode=1)
        return self.corner_fc(paddle.concat([corner_features, img_features], axis = 1))

    def forward(self, img_features, row_features, col_features):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        row_mask = None
        col_mask = None
        return self.decode(img_features, row_features, col_features, row_mask, col_mask)
