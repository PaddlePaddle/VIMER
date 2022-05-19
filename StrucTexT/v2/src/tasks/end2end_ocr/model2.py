""" sequence labeling at segment-level"""
import os
import sys
import math
import json
import paddle as P
import numpy as np
import pycocotools.mask as mask_util
from paddle import nn
from paddle import ParamAttr
from paddle.nn import functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from src.StrucTexT.arch.base_model import Encoder
from src.StrucTexT.ernie import append_name, build_linear, build_ln
from src.StrucTexT.backbones.resnet_vd import ConvBNLayer
from paddle.vision.ops import roi_align

def get_gt_mask_from_polygons(gt_poly, width, height, pad_mask,
        labels, is_instance=False):
    """ get_gt_mask_from_polygons """
    assert(labels.ndim == 2)
    bs, seq = labels.shape
    if is_instance:
        out_gt_mask = P.zeros([bs, seq, height, width], 'int32') #(bs, query_num, h, w)
        out_pos_mask = P.zeros([bs, seq, height, width], 'int32') #(bs, query_num, h, w)
    else:
        out_gt_mask = P.zeros([bs, height, width], 'int32')
        out_pos_mask = P.zeros([bs, height, width], 'int32')
    for i, (polygons, padding, label) in enumerate(zip(gt_poly, pad_mask, labels)):
        for j, (obj_poly, pad, cls) in enumerate(zip(polygons, padding, label)):
            if pad.numpy().tolist()[0] == 0:
                break
            obj_poly = obj_poly.numpy().astype('float64').tolist()
            obj_poly = list(map(lambda i: obj_poly[i], [0, 1, 2, 1, 2, 3, 0, 3]))
            rles = mask_util.frPyObjects([obj_poly], height, width)
            rle = mask_util.merge(rles)
            mask = mask_util.decode(rle)
            mask = P.to_tensor(mask, 'int32')
            if is_instance:
                out_pos_mask[i, j] = mask
                out_gt_mask[i, j] = mask * cls
            else:
                out_pos_mask[i] += mask
                out_gt_mask[i] = P.maximum(out_gt_mask[i], mask * cls)
    out_pos_mask = P.cast(out_pos_mask > 0, 'int32')
    out_pos_mask.stop_gradient = True
    out_gt_mask.stop_gradient = True
    return out_gt_mask, out_pos_mask


def normalize_bboxes(input, width, height, scale=1):
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

    nx1 = xx1.cast('float32') / width * scale
    nx2 = xx2.cast('float32') / width * scale
    ny1 = yy1.cast('float32') / height * scale
    ny2 = yy2.cast('float32') / height * scale

    input = P.concat([xx1, yy1, xx2, yy2], axis=-1)
    n_input = P.concat([nx1, ny1, nx2, ny2], axis=-1)
    input.stop_gradient = True
    n_input.stop_gradient = True
    return input, n_input


def get_roi_features(input, #(N, C, H, W)
                     rois, #(num_rois, 4), num_rois = batch_size * max_token_len
                     proposal_h,
                     proposal_w,
                     spatial_scale, # downsampling scale
                     rois_num=None):
        """get_roi_features"""
        if input.ndim == 3:
            input = input.unsqueeze(0)
        if rois.ndim == 3 and rois_num is None:
            rois_num = P.ones((rois.shape[0],)) * rois.shape[1]
            rois = P.flatten(rois, stop_axis=1)
        # paddle vision
        out = roi_align(
                 x=input,
                 boxes=rois.cast('float32'),
                 boxes_num=rois_num.cast('int32'),
                 output_size=(proposal_h, proposal_w),
                 spatial_scale=spatial_scale)
        return out


class Model(Encoder):
    """ task for entity labeling"""
    def __init__(self, config, name=''):
        """ __init__ """
        super(Model, self).__init__(config, name=name)
        self.config = config['task_module']

        num_labels = self.config['num_labels']

        self.proposal_w = self.config['proposal_w']
        self.proposal_h = self.config['proposal_h']
        d_v_input = self.proposal_h * self.proposal_w * self.out_channels

        self.input_proj = ConvBNLayer(
                num_channels=self.out_channels,
                num_filters=self.out_channels - 2,
                filter_size=1,
                stride=1,
                act='relu',
                name='proj_conv')

        self.ss_conv1 = ConvBNLayer(
                num_channels=self.out_channels,
                num_filters=self.out_channels // 4,
                filter_size=3,
                stride=1,
                act='relu',
                name='ss_conv1')

        self.ss_conv2 = ConvBNLayer(
                num_channels=self.out_channels // 4,
                num_filters=self.out_channels // 8,
                filter_size=3,
                stride=1,
                act='relu',
                name='ss_conv2')

        self.ss_classifier = nn.Conv2D(
                in_channels=self.out_channels // 8,
                out_channels=num_labels + 1,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=P.ParamAttr(
                    name='labeling_ss.w_0',
                    initializer=nn.initializer.KaimingNormal()),
                bias_attr=False)

        self.label_classifier = nn.Linear(
                d_v_input,
                num_labels,
                weight_attr=P.ParamAttr(
                    name='labeling_cls.w_0',
                    initializer=nn.initializer.KaimingNormal()),
                bias_attr=False)

        self.mlm = nn.Linear(
                d_v_input,
                768,
                weight_attr=P.ParamAttr(
                    name='token_trans.w_0',
                    initializer=nn.initializer.KaimingNormal()),
                bias_attr=True)
        self.word_emb = nn.Embedding(
                30522,
                768,
                weight_attr=P.ParamAttr(
                    name='word_embedding',
                    initializer=nn.initializer.KaimingNormal()))

    def loss(self, logit, label, mask=None,
            ss_logit=None, ss_label=None, ss_mask=None,
            label_smooth=-1):
        """ loss """
        label = label.cast('int64')
        mask = mask.cast('bool')
        num_classes = logit.shape[-1]
        if label_smooth > 0:
            label = F.one_hot(label, num_classes)
            label = F.label_smooth(label, epsilon=label_smooth)
        weight = P.ones([num_classes], dtype='float32')
        if num_classes == 4:
            weight = P.to_tensor([2, 3, 1, 1], dtype='float32')
        if num_classes == 5:
            weight = P.to_tensor([5, 3, 3, 3, 1], dtype='float32')
        loss = F.cross_entropy(logit, label,
                weight=weight,
                soft_label=label_smooth > 0,
                reduction='none')
        loss = P.masked_select(loss, mask).mean()

        if ss_logit is not None and \
           ss_label is not None and \
           ss_mask is not None:
            ss_label = P.cast(ss_label + ss_mask, 'int64')
            se_loss = F.cross_entropy(ss_logit, ss_label)
            loss += 0.1 * se_loss

        return loss

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        image = input_data['image']
        label = input_data.pop('label')
        label_mask = input_data.pop('label_mask')
        line_bboxes = input_data.pop('line_bboxes')

        enc_out = super(Model, self).forward([image, None])
        enc_final = enc_out['additional_info']['image_feat']
        enc_final = enc_final['out']

        if enc_final.ndim == 3:
            enc_final = P.transpose(enc_final, [0, 2, 1]) #(B, C, P)
            img_shape = enc_final.shape
            wh = int(math.sqrt(img_shape[-1]))
            if img_shape[-1] == wh ** 2 + 1:
                cls, enc_final = enc_final.split([1, -1], axis=-1)
            enc_final = enc_final.reshape([img_shape[0], -1, wh, wh])
        enc_final = self.input_proj(enc_final)

        x_range = P.linspace(-1, 1, P.shape(enc_final)[-1], dtype='float32')
        y_range = P.linspace(-1, 1, P.shape(enc_final)[-2], dtype='float32')
        y, x = P.meshgrid([y_range, x_range])
        x = P.unsqueeze(x, [0, 1])
        y = P.unsqueeze(y, [0, 1])
        y = P.expand(y, shape=[P.shape(enc_final)[0], 1, -1, -1])
        x = P.expand(x, shape=[P.shape(enc_final)[0], 1, -1, -1])
        coord_feat = P.concat([x, y], axis=1)
        enc_final = P.concat([enc_final, coord_feat], axis=1)

        bs, c, h, w = image.shape
        out_sz = enc_final.shape[-1]
        bbox, norm_bbox = normalize_bboxes(line_bboxes, w, h, out_sz)
        gt_mask, pos_mask = get_gt_mask_from_polygons(norm_bbox, out_sz,
                out_sz, label_mask, label)

        roi_feats = get_roi_features(enc_final, norm_bbox,
                                     proposal_h=self.proposal_h,
                                     proposal_w=self.proposal_w,
                                     spatial_scale=1.0)
        roi_feats = roi_feats.reshape(label_mask.shape + [-1])

        logit = self.label_classifier(roi_feats)
        mask = label_mask.cast('bool')
        ss_map = self.ss_conv2(self.ss_conv1(enc_final))
        ss_map = self.ss_classifier(ss_map).transpose((0, 2, 3, 1))

        loss = self.loss(logit, label, mask=mask,
                ss_logit=ss_map, ss_label=gt_mask, ss_mask=pos_mask,
                label_smooth=0.1)
        logit = P.argmax(logit, axis=-1)

        selected_logit = P.masked_select(logit, mask)
        selected_label = P.masked_select(label, mask)
        logit = logit.reshape(label_mask.shape)

        sentence = input_data['sentence']
        sentence_mask = input_data['sentence_mask']
        token_bboxes = input_data['token_bboxes']
        bbox, norm_bbox = normalize_bboxes(token_bboxes, w, h, out_sz)
        roi_feats = get_roi_features(enc_final, norm_bbox,
                                     proposal_h=self.proposal_h,
                                     proposal_w=self.proposal_w,
                                     spatial_scale=1.0)
        roi_feats = roi_feats.reshape(token_bboxes.shape[:-1] + [-1])
        roi_feats = roi_feats.gather_nd(sentence_mask)
        '''
        #modify
        #token_feats = self.mlm(roi_feats)
        #token_logit = token_feats.matmul(self.word_emb.weight, transpose_y=True)
        '''
        token_loss = F.cross_entropy(token_logit, sentence.cast('int64'))
        loss += token_loss * 0.1
        return {'loss': loss, 'logit': selected_logit, 'label': selected_label}
