""" DB """
import os
import sys
import math
import json
import copy
import paddle as P
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle import ParamAttr

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from StrucTexT.arch.base_model import Encoder
from paddle.vision.ops import roi_align
from .dataset import LabelConverter
from .recg_head import RecgHead
from postprocess.db_postprocess import DBPostProcess

class ConvBNLayer(nn.Layer):
    """ConvBNLayer"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)

        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    def forward(self, x):
        """forward"""
        x = self.conv(x)
        x = self.bn(x)
        return x


def get_bias_attr(k):
    """get_bias_attr
    """
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class DBHead(nn.Layer):
    """DB Head
    """
    def __init__(self, in_channels, name_list):
        super(DBHead, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )

    def forward(self, x):
        """forward
        """
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


class Model(Encoder):
    """ task for e2e text spotting """
    def __init__(self, config, name=''):
        super(Model, self).__init__(config, name=name)
        self.det_config = copy.deepcopy(config['det_module'])
        self.recg_config = copy.deepcopy(config['recg_module'])
        self.task = config.get('task', 'e2e')
        self.postprocess_cfg = copy.deepcopy(config['postprocess'])

        in_channels = 128
        self.k = 50
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'thresh'
        ]
        self.binarize = DBHead(in_channels, binarize_name_list)
        self.thresh = DBHead(in_channels, thresh_name_list)

        self.neck_conv = ConvBNLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=[5, 1],
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="neck_conv")

        # recg_head
        self.method = self.recg_config.get("method")
        self.recg_class_num = self.recg_config.get('num_classes')
        self.recg_seq_len = self.recg_config.get('max_seq_len')
        self.decoder_layers = self.recg_config.get('decoder_layers')
        self.return_intermediate_dec = self.recg_config.get('return_intermediate_dec')
        self.recg_loss = self.recg_config['recg_loss']

        self.ocr_recg = RecgHead(
            method=self.method,
            hidden_channels=256,
            seq_len=self.recg_seq_len,
            recg_class_num=self.recg_class_num + 2,
            decoder_layers=self.decoder_layers,
            return_intermediate_dec=self.return_intermediate_dec)

        self.label_converter = LabelConverter(
            seq_len=self.recg_seq_len,
            recg_loss=self.recg_loss)

        # postprocess config
        self.post_process_thresh = self.postprocess_cfg['thresh']
        self.box_thresh = self.postprocess_cfg['box_thresh']
        self.max_candithresh = self.postprocess_cfg['max_candidates']
        self.unclip_ratio = self.postprocess_cfg['unclip_ratio']
        self.score_mode = self.postprocess_cfg['score_mode']
        self.postprocess = DBPostProcess(
            thresh=self.post_process_thresh,
            box_thresh=self.box_thresh,
            max_candidates=self.max_candithresh,
            unclip_ratio= self.unclip_ratio,
            score_mode=self.score_mode)

    def step_function(self, x, y):
        """step_func
        """
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def distort_bboxes(self, bboxes, ori_h, ori_w, pad_scale=1):
        """distort bboxes
        Args:
            bboxes: [num, 4]
            ori_h: the height of the image
            ori_w: the width of the image
        """
        pad = paddle.to_tensor([-1, -1, 1, 1], dtype='float32') * pad_scale
        offset = paddle.to_tensor(np.random.randint(-pad_scale, pad_scale + 1, size=bboxes.shape), dtype='float32')
        pad = pad + offset
        bboxes = bboxes + pad

        bboxes[:, ::2] = bboxes[:, ::2].clip(0, ori_w)
        bboxes[:, 1::2] = bboxes[:, 1::2].clip(0, ori_h)
        return bboxes

    def pad_rois_w(self, rois):
        """padding bbox width to the same width
        Args:
            rois: [num, 4]
        Returns:
            rois_padded: [num, 4]
            rois_masks: [num, 1, 1, w_max]
        """
        rois = rois.cast('int32')
        num = rois.shape[0]
        rois_w = paddle.abs(rois[:, 2] - rois[:, 0])  # [num]
        rois_w_max = paddle.max(rois_w, axis=-1)
        rois[:, 2] = paddle.clip(rois[:, 0] + rois_w_max, min=0, max=959)

        rois_masks = paddle.zeros([num, rois_w_max], dtype='int32')
        for i in range(num):
            if rois_w[i] == 0: # boundary condition
                rois_masks[i, :] = 1
            else:
                rois_masks[i, :rois_w[i]] = 1

        return rois.cast('float32'), rois_masks.unsqueeze(-2).unsqueeze(-2), rois_w_max

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        image = input_data['image']
        bs, _, ori_h, ori_w = image.shape
        shape_list = [(image[i].shape[1], image[i].shape[2], 1, 1) for i in range(image.shape[0])]
        # backbone
        enc_out = super(Model, self).forward([image, None])
        enc_out = enc_out['additional_info']['image_feat']
        x = enc_out['out']  # [bs, 128, h, w]

        # detection
        shrink_maps = self.binarize(x)
        results =  {'maps': shrink_maps}

        # recognition
        rois = []
        rois_num = []
        bbox_out = self.postprocess(results, shape_list)
        for b in range(bs):
            pred_res = bbox_out[b]['points']  # [num, 4, 2] nd_array
            pt1 = pred_res[:, 0, :]
            pt2 = pred_res[:, 2, :]
            bboxes = np.concatenate((pt1, pt2), axis=-1)
            bboxes = paddle.to_tensor(bboxes, dtype='float32')  # [num, 4]
            rois_num.append(bboxes.shape[0])
            rois.append(bboxes)

        rois_num = paddle.to_tensor(rois_num, dtype='int32')
        rois = paddle.concat(rois, axis=0)
        roi_feat = roi_align(
            x,
            rois,
            output_size=(5, 50),
            spatial_scale=0.25,
            boxes_num=rois_num)

        neck_feat = self.neck_conv(roi_feat)
        recg_out = self.ocr_recg(neck_feat)[-1]

        num_idx = 0
        recg_result = []
        for num in rois_num:
            recg_result.append(recg_out[num_idx:(num_idx + num)])
            num_idx += num
        pred_labels = {'det_result': bbox_out, 'recg_result': recg_result}
        results['e2e_preds'] = self.inference(pred_labels)

        # prepare eval labels for eval
        if input_data.__contains__('texts_padded_list'):
            bboxes_padded_list = input_data['bboxes_4pts_padded_list']
            texts_padded_list = input_data['texts_padded_list']
            masks_padded_list = input_data['masks_padded_list']
            classes_padded_list = input_data['classes_padded_list']
            gt_label = []
            for b in range(bs):
                bboxes = bboxes_padded_list[b]  # [512, 4]
                texts = texts_padded_list[b]  # [512, 50]
                text_classes = classes_padded_list[b]
                masks = masks_padded_list[b]
                bool_idxes = paddle.nonzero(masks) # [38,1]

                bboxes = paddle.index_select(bboxes, bool_idxes)
                texts = paddle.index_select(texts, bool_idxes)
                classes = paddle.index_select(text_classes, bool_idxes)
                for text, bbox, cls in zip(texts, bboxes, classes):
                    text = self.label_converter.decode(text.numpy()).upper()
                    bbox = bbox.numpy().astype('int').tolist()
                    cls = cls.numpy().tolist()[0]
                    gt_label.append([bbox, cls, text])
            results['gt_label'] = gt_label
            results.update(input_data)
        return results

    def inference(self, raw_results):
        """
        Output: poly, text, score
        """
        batch_size = len(raw_results['det_result'])
        processed_results = []

        for bs_idx in range(batch_size):
            processed_result = []
            res_num = len(raw_results['det_result'][bs_idx]['points'])
            for idx in range(res_num):
                poly = raw_results['det_result'][bs_idx]['points'][idx]
                if isinstance(poly, paddle.Tensor):
                    poly = poly.tolist()
                else:
                    poly = poly.reshape(-1).tolist()

                if raw_results.__contains__('recg_result'):
                    transcript = raw_results['recg_result'][bs_idx][idx]
                    word, prob = self.decode_transcript(transcript)
                    processed_result.append([poly, word, prob])
                else:
                    processed_result.append(poly)
            processed_results.append(processed_result)
        return processed_results

    def decode_transcript(self, pred_recg):
        """decode_transcript
        """
        _, preds_index = pred_recg.topk(1, axis=-1, largest=True, sorted=True)
        probs = paddle.nn.functional.softmax(pred_recg, axis=-1)
        probs = probs.topk(1, axis=-1, largest=True, sorted=True)[0].reshape([-1])
        preds_index = preds_index.reshape([-1])
        word = self.label_converter.decode(preds_index)
        prob = 0.0 if len(word) == 0 else float(probs[:len(word)].mean())

        return word, prob
