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


class SAConv2d(nn.Layer):
    """SAConv2d
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 name=None):
        super(SAConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation

        self.conv_s = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=ParamAttr(name=name + "_bias"))
        weight_diff = self.create_parameter(
            self.conv_s.weight.shape,
            ParamAttr(initializer=paddle.nn.initializer.Constant(value=0)))
        self.add_parameter("weight_diff", weight_diff)
        self.switch = nn.Conv2D(
            self.in_channels,
            1,
            kernel_size=1,
            stride=stride,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1)))
        self.pre_context = nn.Conv2D(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0)))
        self.post_context = nn.Conv2D(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0)))

    def forward(self, x):
        """forward"""
        # pre-context
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        out_s = self.conv_s(x)
        out_l = F.conv2d(x, self.conv_s.weight + self.weight_diff, padding=self.padding * 3, dilation=self.dilation * 3)
        out = switch * out_s + (1 - switch) * out_l
        # post-context
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out


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
                 name=None,
                 sac=False):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        if sac:
            self.conv = SAConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                name=name)
        else:
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


class DBNeck(nn.Layer):
    """DB Neck
    """
    def __init__(self, in_channels, name,
            kernel_size=3, padding=1, layers=3,
            sac=False, dyrelu=False):
        super(DBNeck, self).__init__()
        # conv
        self.neck = nn.Sequential()
        for i in range(layers):
            conv = ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    if_act=True,
                    act='relu',
                    name=name + 'conv{}'.format(i + 1),
                    sac=sac)
            self.neck.add_sublayer('conv{}'.format(i + 1), conv)
            if dyrelu:
                dy = DyReLU(channels=in_channels)
                self.neck.add_sublayer('dy{}'.format(i + 1), dy)

    def forward(self, x):
        """forward
        """
        return self.neck(x)


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
        self.word_neck = DBNeck(in_channels, "word_neck", layers=6, sac=False, dyrelu=False)

        self.binarize_line = DBHead(in_channels, binarize_name_list)
        self.thresh_line = DBHead(in_channels, thresh_name_list)
        self.line_neck = DBNeck(in_channels, "line_neck", layers=6, sac=True, dyrelu=False)

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
                unclip_ratio=self.unclip_ratio,
                score_mode=self.score_mode)

        self.postprocess_line = DBPostProcess(
                thresh=self.post_process_thresh,
                box_thresh=0.3,
                max_candidates=self.max_candithresh,
                unclip_ratio=self.unclip_ratio,
                score_mode=self.score_mode)

        self.labeling_config = copy.deepcopy(config['labeling_module'])
        self.num_labels = self.labeling_config['num_labels'] + 1
        self.proposal_w = self.labeling_config['proposal_w']
        self.proposal_h = self.labeling_config['proposal_h']

        label_input_dim = self.proposal_h * self.proposal_w * self.out_channels
        self.label_classifier = nn.Linear(
                label_input_dim,
                self.num_labels,
                weight_attr=P.ParamAttr(
                    name='labeling_cls.w_0',
                    initializer=nn.initializer.KaimingNormal()),
                bias_attr=False)
        self.label_neck = DBNeck(in_channels, "label_neck",
                kernel_size=3, padding=1, layers=3,
                sac=True, dyrelu=False)

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

        # word detection
        shrink_maps = self.binarize(self.word_neck(x))
        results =  {'maps': shrink_maps}

        # line detection
        shrink_maps_line = self.binarize_line(self.line_neck(x))
        results_line =  {'maps': shrink_maps_line}

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

        # line_labeling
        rois = []
        rois_num = []
        bbox_out_line = self.postprocess_line(results_line, shape_list)

        for b in range(bs):
            pred_res = bbox_out_line[b]['points']
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
            output_size=(self.proposal_h, self.proposal_w),
            spatial_scale=0.25,
            boxes_num=rois_num)

        labeling_feat = self.label_neck(roi_feat)
        labeling_feat = labeling_feat.reshape(labeling_feat.shape[:1] + [-1]) # [bs*num, 128*4*64]
        labeling_logit = self.label_classifier(labeling_feat) # [bs*num, 5]
        labeling_out = P.argmax(labeling_logit, axis=-1) # [bs*num, 5]

        num_idx = 0
        class_result = []
        for num in rois_num:
            class_result.append(labeling_out[num_idx:(num_idx + num)])
            num_idx += num
        pred_labels_line = {'det_result': bbox_out_line, 'class_result': class_result}
        results['line_preds'] = self.inference(pred_labels_line)

        # prepare eval labels for eval
        if input_data.__contains__('bboxes_padded_list_line'):
            bboxes_4pts_padded_list = input_data['bboxes_4pts_padded_list_line']
            texts_padded_list = input_data['texts_padded_list_line']
            masks_padded_list = input_data['masks_padded_list_line']
            classes_padded_list = input_data['classes_padded_list_line']
            gt_label = []
            for b in range(bs):
                bboxes_4pts = bboxes_4pts_padded_list[b]
                text_classes = classes_padded_list[b]
                masks = masks_padded_list[b]
                bool_idxes = paddle.nonzero(masks)

                bboxes_4pts = paddle.index_select(bboxes_4pts, bool_idxes)
                texts = list(map(lambda x: np.array([e.numpy() for e in x]).flatten(), texts_padded_list))
                classes = paddle.index_select(text_classes, bool_idxes)
                for text, bbox, cls in zip(texts, bboxes_4pts, classes):
                    text = self.label_converter.decode(text).upper()
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
                elif raw_results.__contains__('class_result'):
                    cls = raw_results['class_result'][bs_idx][idx].tolist()[0]
                    if cls == self.num_labels - 1:
                        continue
                    processed_result.append([poly, cls])
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
