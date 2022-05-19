""" document classify """
import os
import sys
import paddle as P
from paddle import nn
from paddle.nn import functional as F

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

from src.StrucTexT.arch.base_model import Encoder
from src.StrucTexT.backbones.resnet_vd import ConvBNLayer

class Model(Encoder):
    """ task for entity labeling"""
    def __init__(self, config, name=''):
        """ __init__ """
        super(Model, self).__init__(config, name=name)
        self.config = config['task_module']

        num_labels = self.config['num_labels']

        self.conv1 = ConvBNLayer(
                num_channels=self.out_channels,
                num_filters=self.out_channels,
                filter_size=3,
                stride=2,
                act='relu',
                name='proj_conv1')
        self.conv2 = ConvBNLayer(
                num_channels=self.out_channels,
                num_filters=16,
                filter_size=3,
                stride=2,
                act='relu',
                name='proj_conv2')
        self.label_classifier = nn.Linear(
                16 * 60 * 60,
                num_labels,
                weight_attr=P.ParamAttr(
                    name='classifier.w_0',
                    initializer=nn.initializer.KaimingNormal()),
                bias_attr=False)

    def forward(self, *args, **kwargs):
        """ forword """
        feed_names = kwargs.get('feed_names')
        input_data = dict(zip(feed_names, args))

        image = input_data.pop('image')
        label = input_data.pop('label')

        enc_out = super(Model, self).forward([image, None])
        enc_final = enc_out['additional_info']['image_feat']['out']

        enc_final = self.conv2(self.conv1(enc_final))
        enc_final = enc_final.flatten(1)
        logit = self.label_classifier(enc_final)
        loss = self.loss(logit, label, label_smooth=0.1)
        return {'logit': logit, 'label': label}
