""" sequence labeling at segment-level"""
import json
import paddle as P
import numpy as np
from paddle import nn
from model.encoder import Encoder
from paddle.nn import functional as F
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln

class Model(Encoder):
    """ task for entity labeling"""
    def __init__(self, config, name=''):
        """ __init__ """
        ernie_config = config['ernie']
        if isinstance(ernie_config, str):
            ernie_config = json.loads(open(ernie_config).read())
            config['ernie'] = ernie_config
        super(Model, self).__init__(config, name=name)

        cls_config = config['cls_header']
        num_labels = cls_config['num_labels']

        self.label_classifier = _build_linear(
                self.d_model,
                num_labels,
                append_name(name, 'labeling_cls'),
                nn.initializer.KaimingNormal())

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        encoded, token_embeded = super(Model, self).forward(**input_data)
        encoded_2d = encoded.unsqueeze((1)) # [batch_size, 1, max_seqlen, d_model]

        seq_ids = input_data.get('sentence_ids')
        seq_mask = input_data.get('sentence_mask')
        label_mask = input_data.pop('label_mask') # [batch_size, line_num]

        max_line_num = label_mask.shape[1]
        seq_ids = P.stack([(seq_ids == i).cast('float32') for i in range(1, max_line_num + 1)], axis=1) # [batch, line_num, max_seqlen]

        lang_mask = P.cast(seq_mask == 0, 'int32') # [batch_size, max_seqlen]
        lang_ids = seq_ids * lang_mask.unsqueeze(1) # [batch_size, line_num, max_seqlen]
        lang_ids = lang_ids.unsqueeze(-1)
        lang_emb = encoded_2d * lang_ids  # [batch_size, line_num, max_seqlen, d_model]
        lang_logit = lang_emb.sum(axis=2) / lang_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        line_mask = P.cast(seq_mask == 1, 'int32') # [batch_size, max_seqlen]
        line_ids = seq_ids * line_mask.unsqueeze(1) # [batch_size, line_num, max_seqlen]
        line_ids = line_ids.unsqueeze(-1)
        line_emb = encoded_2d * line_ids
        line_logit = line_emb.sum(axis=2) / line_ids.sum(axis=2).clip(min=1.0) # [batch_size, line_num, d_model]

        logit = line_logit * lang_logit
        logit = self.label_classifier(logit) # [batch_size, line_num, num_labels]

        label = input_data.get('label')
        logit = P.argmax(logit, axis=-1)
        mask = label_mask.cast('bool')

        selected_logit = P.masked_select(logit, mask)
        selected_label = P.masked_select(label, mask)
        mask = mask.cast('int32')

        return {'logit': selected_logit, 'label': selected_label,
                'logit_prim': logit, 'label_prim': label, 'mask': mask}

    def eval(self):
        """ eval """
        if P.in_dynamic_mode():
            super(Model, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        """ train """
        if P.in_dynamic_mode():
            super(Model, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self
