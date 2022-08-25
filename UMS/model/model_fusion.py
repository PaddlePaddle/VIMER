"""model_fusion
"""
import os
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, KaimingUniform
from .visual import vit_deit_base_patch16_224
from .visual import vit_deit_base_patch16_384
from .visual import deit_base_patch16_224
from .visual import deit_base_patch16_384
from collections import OrderedDict
import sys

sys.path.append("../")
from paddlenlp.transformers import BertTokenizer, BertModel

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
kaiming_uniform = KaimingUniform()


def weights_init_kaiming(m):
    """kaiming initialization"""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        kaiming_uniform(m.weight)
    elif classname.find("Conv") != -1:
        kaiming_uniform(m.weight)
        if m.bias is not None:
            zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            ones_(m.weight)
            zeros_(m.bias)


"""=================================================================================================================================="""
### ATTRIBUTE CHANGE HELPER
def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


"""=================================================================================================================================="""
### NETWORK SELECTION FUNCTION
def networkselect(args):
    """
    Selection function for available networks.

    Args:
        opt: argparse.Namespace, contains all training-specific training parameters.
    Returns:
        Network of choice
    """
    if args.arch == "vit-deit-base":
        network = DeitBase(args)
    else:
        raise Exception("No implementation for model {} available!")

    return network


class DeitBase(nn.Layer):
    """
    input: image
    output:featue

    """

    def __init__(self, args, list_style=False, no_norm=False):
        super(DeitBase, self).__init__()

        self.model = deit_base_patch16_224()
        self.pooling = nn.Identity()
        self.standardize = nn.Identity()
        self.fc = nn.Identity()

    def forward(self, x):
        """forward"""
        X = self.model(x)
        X = self.pooling(X)
        X = self.standardize(X)
        X = self.fc(X)
        X = paddle.nn.functional.normalize(X, p=2, axis=1)

        return X
