"""
ocr recg head
"""
import paddle as P
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
import math
from utils.initializer import linear_init_, conv_init_, xavier_uniform_, normal_

from StrucTexT.backbones.base.transformer import TransformerEncoder, TransformerEncoderLayer
from StrucTexT.backbones.base.transformer import TransformerDecoder, TransformerDecoderLayer


class PositionEmbeddingLearned1D(nn.Layer):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, dim=-1, max_length=100):
        super().__init__()
        self.col_embed = nn.Embedding(max_length, num_pos_feats)
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters
        """
        nn.initializer.XavierUniform(self.col_embed.weight)

    def forward(self, x):
        """
            x: (..., w, ...)
            pos: (w, 256)
        """
        w = x.shape[self.dim]
        i = paddle.arange(w)
        x_emb = self.col_embed(i)
        pos = x_emb
        return pos


class TransformerRecg(nn.Layer):
    """transformer recg
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.return_intermediate_dec = return_intermediate_dec
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.initializer.XavierUniform(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """forward
        """
 
        bs, c, h, w = src.shape
        src = src.flatten(2).transpose([0, 2, 1])
        query_num, query_dim = query_embed.shape
        # FIXME repeat api
        # batch_size, tgt_len, d_model
        query_embed = query_embed.reshape([-1]).expand([bs, -1]).reshape([bs, query_num, query_dim])
        # [seq_len, embed]
  
        if mask is not None:
            mask = mask.flatten(1)

        tgt = paddle.zeros_like(query_embed)
        memory = self.encoder(src, src_mask=mask, pos_embed=pos_embed)
        # tgt[50,45,128] query, bs, dim
        # memroy [48,45, 128]  [h*w bs, dim]
        hs = self.decoder(
            tgt, memory, memory_mask=mask, pos_embed=pos_embed, query_pos_embed=query_embed
        )
        # memory.view(k, l, bs, queries, embed).permute(1, 2, 3, 0, 4)

        if self.return_intermediate_dec:
            # # [l, bs, queries, layer, seq_len, embed]'

            # return hs.permute(2, 0, 1, 3).view(bs, queries, hs.shape[0], 50, embed)
            return hs.squeeze(0)
        else:
            # [bs, queries, seq_len, embed]
            # hs [1, bs, queries, embed ]
            # return hs.transpose(1, 2).squeeze(0).view(bs, queries, seq_len, embed)
            return hs.squeeze(0)
           

class TransformerEncoderOnly(nn.Layer):
    """transfomer encoder only
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.initializer.XavierUniform(p)

    def forward(self, src, mask, pos_embed):
        """forward
        """

        # flatten [l, bs, queries, k, embed] to [k, l * bs * queries, embed]
        #   bs, c, h, w = src.shape
  
        bs, c, h, w = src.shape  # [num, embed_dim, h, w]

        src = src.flatten(2).transpose([0, 2, 1])  # [num, h*w, embed_dim]

        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_mask=mask, pos_embed=pos_embed) # [num, h*w, embed_dim]

        return memory


class RecgHead(nn.Layer):
    """ for recog, bs * query * h * w --> bs * query * seq_len * class_num 
    """
    def __init__(self, 
        method="decoder", 
        hidden_channels=256, 
        seq_len=32, 
        recg_class_num=97,
        decoder_layers=2,
        return_intermediate_dec=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.recg_class_num = recg_class_num
        self.recg_proj = nn.Linear(hidden_channels, recg_class_num)
        self.decoder_layers = decoder_layers
        self.return_intermediate_dec = return_intermediate_dec
        # positional encoding
        self.pe_layer = PositionEmbeddingLearned1D(self.hidden_channels, dim=-1, max_length=1000)
        # self.pe_layer = PositionalEmbeddingSine1D(self.hidden_channels)
        self.method = method
        if method == "encoder":
            self.transformer = TransformerEncoderOnly(
                d_model=hidden_channels,
                dropout=0.1,
                nhead=8,
                dim_feedforward=2048,
                num_encoder_layers=2,
                normalize_before=False,
            )
        elif method == "decoder":
            self.transformer = TransformerRecg(
                d_model=hidden_channels,
                nhead=8,
                num_encoder_layers=0,
                num_decoder_layers=self.decoder_layers,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=self.return_intermediate_dec,
            )
            self.query_embed = nn.Embedding(self.seq_len, hidden_channels)
        else:
            raise ValueError("Recg method not implemented!")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # conv_init_(self.input_proj)
        # normal_(self.query_pos_embed.weight)

    def forward(self, x):
        """forward
            input: x[num_rois, hidden_dim, h, w]
        """
        # [bs, queries, k, embed] -> [bs, queries, k, recg_class_num]
        # x: [b,c,h,w]
        # num_rois, dim, h, w = x.shape
        # mask = paddle.zeros([num_rois, h, w], dtype='bool')
        # not_mask = ~mask
        # pos = self.pe_layer(not_mask)
        # mask = None
        
        # [50, 256]
        num_rois, dim, h, w = x.shape
        # [1,50,256]
        pos = self.pe_layer(x).unsqueeze(0)
        # [37, 50, 256]
        pos = paddle.expand(pos, shape=[num_rois, w, dim])
        # batch_size, src_len, d_model]
        mask = None
        if self.method == "encoder":
            x = self.transformer(x, mask, pos)
        else:
            x = self.transformer(x, mask, self.query_embed.weight, pos)

        out = self.recg_proj(x)

        return out




