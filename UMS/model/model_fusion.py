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
sys.path.append('../')
from paddlenlp.transformers import BertTokenizer, BertModel
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
kaiming_uniform = KaimingUniform()

def weights_init_kaiming(m):
    """kaiming initialization
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        kaiming_uniform(m.weight)
    elif classname.find('Conv') != -1:
        kaiming_uniform(m.weight)
        if m.bias is not None:
            zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            ones_(m.weight)
            zeros_(m.bias)

class ITCL(nn.Layer):
    """ITCL module
    """
    def __init__(self, args):
        super().__init__()

        image_model_name = args.image_model_name
        image_model_ckpt = args.image_model_ckpt
        text_model_dir = args.text_model_dir

        self.is_frozen = args.is_frozen
        self.projection_dim = args.projection_dim
        self.text_embed_dim = args.text_embed_dim
        self.image_embed_dim = args.image_embed_dim
        if image_model_name == 'vit_deit_base_patch16_384':
            backbone = vit_deit_base_patch16_384()
        elif image_model_name == 'vit_deit_base_patch16_224':
            backbone = vit_deit_base_patch16_224()
        elif image_model_name == 'vit_base_patch16_224':
            backbone = deit_base_patch16_224()
        elif image_model_name == 'vit_base_patch16_384':
            backbone = deit_base_patch16_384()
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim, bias_attr=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias_attr=False)
        self.image_cls = nn.Sequential(('backbone', backbone))
        if os.path.exists(image_model_ckpt):
            image_ckpt = paddle.load(image_model_ckpt)
            del image_ckpt["model"]['pos_embed']
            keys = self.image_cls.backbone.set_state_dict(image_ckpt["model"])
            print("[missing]: {}".format(keys[0]))
            print("[unexpect]: {}".format(keys[1]))
        else:
            print("not load image backbone: {}".format(image_model_ckpt))

        self.text_model = BertModel.from_pretrained(text_model_dir)

        self._init_weights()

    def _init_weights(self):
        self.image_projection.apply(weights_init_kaiming)
        self.text_projection.apply(weights_init_kaiming)
        return

    def encode_text(
        self,
        input_ids,
        attention_mask,
        token_type_ids
    ):
        """text encoder
        """
        return self.text_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids)

    def encode_image(
        self, 
        image 
    ):  
        """image encoder
        """
        return self.image_cls.backbone(image)

    def forward(
        self, 
        image, 
        input_ids, 
        attention_mask=None,
        token_type_ids=None,
        is_train=False,
    ):
        """ITCL forward
        """
        if self.is_frozen:
            paddle.set_grad_enabled(False)

        image_features_cls = self.encode_image(
            image
        )
        
        text_features = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.is_frozen:
            paddle.set_grad_enabled(True)
        image_embeds = self.image_projection(image_features_cls)
        text_features_cls = text_features[0][:, 0]
        text_embeds = self.text_projection(text_features_cls)
        
        image_embeds = image_embeds / paddle.norm(image_embeds, axis=-1, keepdim=True)
        text_embeds = text_embeds / paddle.norm(text_embeds, axis=-1, keepdim=True)
        output = (text_embeds, image_embeds, None)
        return output


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
        """forward
        """
        X = self.model(x)
        X = self.pooling(X)
        X = self.standardize(X)
        X = self.fc(X)
        X = paddle.nn.functional.normalize(X, p=2, axis=1)

        return X
