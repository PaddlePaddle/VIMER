""" Encoder """
import os
import cv2
import json
import numpy as np
import importlib
import paddle
import paddle as P
import paddle.nn as nn
import paddle.nn.functional as F
from utils.op import roi_align
from model.backbones.east_fpn import FPN
from model.ernie.modeling_ernie import ErnieModel, append_name, _build_linear, _get_rel_pos_bias

def normalize_bboxes(input, width, height, scale=None):
    """ normalize_bboxes """
    x1, y1, x2, y2 = P.split(input, num_or_sections=4, axis=-1)

    x1 = P.clip(x1, min=0., max=width - 1)
    x2 = P.clip(x2, min=0., max=width - 1)
    y1 = P.clip(y1, min=0., max=height - 1)
    y2 = P.clip(y2, min=0., max=height - 1)

    xx1 = P.where(x2 > x1, x1, x2)
    xx2 = P.where(x2 > x1, x2, x1)
    yy1 = P.where(y2 > y1, y1, y2)
    yy2 = P.where(y2 > y1, y2, y1)

    if scale is None:
        scale = max(width, height)
    nx1 = xx1.cast('float32') / width * scale
    nx2 = xx2.cast('float32') / width * scale
    ny1 = yy1.cast('float32') / height * scale
    ny2 = yy2.cast('float32') / height * scale

    input = P.concat([xx1, yy1, xx2, yy2], axis=-1)
    n_input = P.concat([nx1, ny1, nx2, ny2], axis=-1).cast('int64')
    input.stop_gradient = True
    n_input.stop_gradient = True
    return input, n_input


def get_roi_features(input, #N, C, H, W
                     rois, #num_rois, 4, num_rois = batch_size * max_token_len
                     proposal_h=8,
                     proposal_w=64,
                     spatial_scale=0.25, # downsampling scale
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


def calculatesum_mask(input_mask, tgt_mask=None):
    """
     mask: (batch_size, max_seqlen)
    """
    input_mask = input_mask.cast('float32').unsqueeze(-1)
    if tgt_mask is not None:
        tgt_mask = tgt_mask.cast('float32').unsqueeze(-1)
    else:
        tgt_mask = input_mask
    attn_bias = input_mask.matmul(tgt_mask, transpose_y=True)
    attn_bias = P.scale(x=attn_bias, scale=10000.0,
                bias=-1.0, bias_after_scale=False)
    attn_bias.stop_gradient = True
    return attn_bias


class Encoder(ErnieModel):
    """ Encoder """
    def __init__(self, config, name=''):
        """ __init__ """
        self.config = config
        emb_config = config['embedding']
        cnn_config = config['visual_backbone']
        ernie_config = config['ernie']
        super(Encoder, self).__init__(ernie_config, name=name)

        max_seqlen = emb_config['max_seqlen']
        max_position = ernie_config['max_position_embeddings']
        assert (max_position >= max_seqlen), 'the max_seqlen \
                must be samll than position_size of ernie'

        self.d_spa_pos = emb_config.get('spa_pos_size', None)
        self.d_rel_pos = emb_config.get('rel_pos_size', None)

        self.proposal_w = emb_config['roi_width']
        self.proposal_h = emb_config['roi_height']
        self.d2_pos = emb_config['max_2d_position_embedding']
        fpn_dim = cnn_config['fpn_dim']
        d_v_input = self.proposal_w * self.proposal_h * fpn_dim

        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=self.initializer))
        if self.d_spa_pos:
            self.spa_pos_bias_emb = nn.Embedding(
                self.d_spa_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'spa_pos_embedding'),
                    initializer=self.initializer))

        self.x_position_embeddings = nn.Embedding(
            self.d2_pos, self.d_model,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'x_pos_embedding'),
                initializer=self.initializer))
        self.y_position_embeddings = nn.Embedding(
            self.d2_pos, self.d_model,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'y_pos_embedding'),
                initializer=self.initializer))
        self.h_position_embeddings = nn.Embedding(
            self.d2_pos, self.d_model,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'h_position_embedding'),
                initializer=self.initializer))
        self.w_position_embeddings = nn.Embedding(
            self.d2_pos, self.d_model,
             weight_attr=P.ParamAttr(
                name=append_name(name, 'w_position_embedding'),
                initializer=self.initializer))

        ''' cnn '''
        backbones = importlib.import_module(cnn_config['module'])
        self.backbone = getattr(backbones, cnn_config['class'])(**cnn_config['params'])
        self.neck = FPN(self.backbone.out_channels, fpn_dim)

        self.proj = _build_linear(d_v_input, self.d_model, 'proj', self.initializer)
        self.bn = nn.BatchNorm(self.d_model)

    def _cal_1d_pos_emb(self, pos_ids):
        seq_len = pos_ids.shape[-1]
        rel_pos_mat = pos_ids.unsqueeze(1) - pos_ids.unsqueeze(2)

        rel_pos = _get_rel_pos_bias(
            rel_pos=rel_pos_mat,
            num_buckets=self.d_rel_pos
        )
        rel_pos = self.rel_pos_bias_emb(rel_pos)
        rel_pos = rel_pos.transpose([0, 3, 1, 2])
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        seq_len = bbox.shape[-2]
        max_rel_pos = self.d2_pos // 2

        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(1) - position_coord_x.unsqueeze(2)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(1) - position_coord_y.unsqueeze(2)
        rel_pos_x = _get_rel_pos_bias(
            rel_pos=rel_pos_x_2d_mat,
            max_len=max_rel_pos,
            num_buckets=self.d_spa_pos
        )
        rel_pos_y = _get_rel_pos_bias(
            rel_pos=rel_pos_y_2d_mat,
            max_len=max_rel_pos,
            num_buckets=self.d_spa_pos
        )
        rel_pos_x = self.spa_pos_bias_emb(rel_pos_x)
        rel_pos_y = self.spa_pos_bias_emb(rel_pos_y)
        rel_pos = rel_pos_x + rel_pos_y
        rel_pos = rel_pos.transpose([0, 3, 1, 2])
        return rel_pos

    def forward(self, *args, **kwargs):
        """ forward """
        images = kwargs.get('images')
        seq_token = kwargs.get('sentence')
        seq_ids = kwargs.get('sentence_ids')
        seq_mask = kwargs.get('sentence_mask')
        seq_bboxes = kwargs.get('sentence_bboxes')
        past_cache = kwargs.get('past_cache', None)

        batch_size = seq_mask.shape[0]
        max_seqlen = seq_mask.shape[1]

        token_embeded = self.word_emb(seq_token)
        token_embeded = token_embeded.unbind(axis=0)

        h, w = images.shape[2], images.shape[3]
        seq_bboxes, norm_bboxes = normalize_bboxes(seq_bboxes, w, h, self.d2_pos)

        cnn = self.neck(self.backbone(images))
        for idx in range(batch_size):
            token_index = P.cast(seq_mask[idx] == 0, 'int32')
            token_num = P.sum(token_index).numpy().tolist()[0]
            line_index = P.cast(seq_mask[idx] == 1, 'int32')
            line_num = P.sum(line_index).numpy().tolist()[0]
            pad_index = P.cast(seq_mask[idx] > 1, 'int32')
            pad_num = P.sum(pad_index).numpy().tolist()[0]
            lang_emb, line_emd, pad_emb = token_embeded[idx].split([token_num, line_num, pad_num], axis=0)
            line_bboxes = seq_bboxes[idx, token_num:token_num + line_num]
            roi_feats = get_roi_features(cnn[idx], line_bboxes,
                                         proposal_h=self.proposal_h,
                                         proposal_w=self.proposal_w)
            roi_feats = P.flatten(roi_feats, start_axis=1) # (B * N, D * H * W)
            image_segments = self.bn(self.proj(roi_feats)) # (B * N, d_model)
            token_embeded[idx] = P.concat([lang_emb, image_segments, pad_emb]).unsqueeze((0))
        token_embeded = P.concat(token_embeded)

        left_position_embeddings = self.x_position_embeddings(norm_bboxes[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(norm_bboxes[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(norm_bboxes[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(norm_bboxes[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(norm_bboxes[:, :, 3] - norm_bboxes[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(norm_bboxes[:, :, 2] - norm_bboxes[:, :, 0])
        layout_embeded = left_position_embeddings + upper_position_embeddings + right_position_embeddings \
                + lower_position_embeddings + h_position_embeddings + w_position_embeddings

        pos_ids = P.arange(0, max_seqlen, 1, dtype='int64').reshape([1, -1])
        pos_embeded = self.pos_emb(pos_ids)
        line_embeded = self.pos_emb(seq_ids)
        sent_embeded = self.sent_emb(seq_mask)

        embedded = token_embeded + pos_embeded + sent_embeded + line_embeded + layout_embeded

        nan_count = int(embedded.isnan().cast('int32').sum().numpy()[0])
        if nan_count > 0:
            embedded = P.where(embedded.is_nan(), P.zeros_like(embedded, 'float32'), embedded)
            logging.error('there are nan in embedded input', nan_count)

        attn_bias = calculatesum_mask(seq_mask < 2)
        attn_bias = P.stack(x=[attn_bias] * self.n_head, axis=1)
        if self.d_rel_pos:
            attn_bias += self._cal_1d_pos_emb(seq_ids)
        if self.d_spa_pos:
            attn_bias += self._cal_2d_pos_emb(norm_bboxes)

        pooled, encoded = super(Encoder, self).forward(emb_out=embedded, attn_bias=attn_bias)

        return encoded, token_embeded

    def eval(self):
        """ eval """
        if P.in_dynamic_mode():
            super(Encoder, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        """ train """
        if P.in_dynamic_mode():
            super(Encoder, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self
