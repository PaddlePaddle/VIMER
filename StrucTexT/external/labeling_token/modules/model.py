""" sequence labeling at segment-level"""
import json
import paddle as P
import numpy as np
from paddle import nn
from model.encoder import Encoder
from paddle.nn import functional as F
from model.ernie.modeling_ernie import ACT_DICT, append_name, _build_linear, _build_ln

class Model(Encoder):
    """ task for entity labeling at token-level"""
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
        self.train()

    def loss(self, logit, label,
            mask=None, label_smooth=-1):
        """ loss """
        label = label.cast('int64')
        mask = mask.cast('bool')
        num_classes = logit.shape[-1]
        if label_smooth > 0:
            label = F.one_hot(label, num_classes)
            label = F.label_smooth(label, epsilon=label_smooth)
        loss = F.cross_entropy(logit, label,
                soft_label=label_smooth > 0,
                reduction='none').squeeze(-1)
        loss = P.masked_select(loss, mask).mean()
        return loss

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        encoded, token_embeded = super(Model, self).forward(**input_data)

        label_mask = input_data.pop('label_mask') # [batch_size, line_num]
        logit = self.label_classifier(encoded) # [batch_size, max_seqlen, num_labels]

        label = input_data.get('label')
        loss = self.loss(logit, label, mask=label_mask, label_smooth=0.1)
        logit = P.argmax(logit, axis=-1)
        mask = label_mask.cast('int32')

        return {'logit': logit, 'label': label, 'loss': loss, 'mask': mask}

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
