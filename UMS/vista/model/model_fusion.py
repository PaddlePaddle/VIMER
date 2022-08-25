"""model_fusion
"""
import os
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, KaimingUniform
from .visual import fusion_small_patch16_224
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


class ViSTA(nn.Layer):
    """ViSTA module"""

    def __init__(self, args):
        super().__init__()

        image_model_name = args.image_model_name
        image_model_ckpt = args.image_model_ckpt
        text_model_dir = args.text_model_dir
        scene_text_model_dir = args.scene_text_model_dir

        self.is_frozen = args.is_frozen
        self.projection_dim = args.projection_dim
        self.text_embed_dim = args.text_embed_dim
        self.image_embed_dim = args.image_embed_dim
        self.scene_text_embed_dim = args.scene_text_embed_dim
        if image_model_name == "fusion_small_patch16_224":
            backbone = fusion_small_patch16_224()
        self.image_projection = nn.Linear(
            self.image_embed_dim, self.projection_dim, bias_attr=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias_attr=False
        )
        self.fusion_projection = nn.Linear(
            self.image_embed_dim, self.projection_dim, bias_attr=False
        )
        self.image_cls = nn.Sequential(("backbone", backbone))
        if os.path.exists(image_model_ckpt):
            image_ckpt = paddle.load(image_model_ckpt)
            del image_ckpt["model"]["pos_embed"]
            keys = self.image_cls.backbone.set_state_dict(image_ckpt["model"])
            print("[missing]: {}".format(keys[0]))
            print("[unexpect]: {}".format(keys[1]))
        else:
            print("not load image backbone: {}".format(image_model_ckpt))

        self.text_model = BertModel.from_pretrained(text_model_dir)
        self.scene_text_model = BertModel.from_pretrained(scene_text_model_dir)

        fusion_embed = self.scene_text_model.config["hidden_size"]
        self.ocr_pos_projection = nn.Linear(4, fusion_embed)

        self._init_weights()

    def _init_weights(self):
        self.image_projection.apply(weights_init_kaiming)
        self.text_projection.apply(weights_init_kaiming)
        self.fusion_projection.apply(weights_init_kaiming)
        self.ocr_pos_projection.apply(weights_init_kaiming)
        return

    def encode_scene_text(
        self,
        scene_text_ids,
        scene_text_attention_mask,
        scene_text_token_type_ids,
        scene_text_pos=None,
    ):
        """scene text encoder"""
        return self.scene_text_model(
            input_ids=scene_text_ids,
            attention_mask=scene_text_attention_mask,
            token_type_ids=scene_text_token_type_ids,
        )

    def encode_text(self, input_ids, attention_mask, token_type_ids):
        """text encoder"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def encode_image(self, image, scene_text_features, scene_text_mask):
        """image encoder"""
        if scene_text_features is not None and scene_text_mask is not None:
            return self.image_cls.backbone(
                image,
                scene_text_features.astype("float32"),
                scene_text_mask.astype("float32"),
            )
        else:
            return self.image_cls.backbone(image)

    def forward(
        self,
        image,
        input_ids,
        scene_text_ids,
        attention_mask=None,
        scene_text_attention_mask=None,
        token_type_ids=None,
        scene_text_token_type_ids=None,
        ocr_feature=None,
        ocr_mask=None,
        is_train=False,
        fusion_mask=None,
        scene_text_pos=None,
    ):
        """ViSTA forward"""
        if self.is_frozen:
            paddle.set_grad_enabled(False)

        scene_text_features = self.encode_scene_text(
            scene_text_ids,
            scene_text_attention_mask,
            scene_text_token_type_ids,
            scene_text_pos=scene_text_pos,
        )
        # scene_text_features = scene_text_features.last_hidden_state
        scene_text_features = scene_text_features[0]
        if scene_text_pos is not None:
            scene_text_pos = self.ocr_pos_projection(scene_text_pos)
            scene_text_features = scene_text_features + scene_text_pos
        image_features_cls, scene_text_cls, fusion_token = self.encode_image(
            image, scene_text_features, scene_text_attention_mask
        )

        text_features = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if self.is_frozen:
            paddle.set_grad_enabled(True)
        image_embeds = self.image_projection(image_features_cls)
        fusion_embeds = self.fusion_projection(fusion_token)

        # text_features_cls = text_features.last_hidden_state[:, 0]
        text_features_cls = text_features[0][:, 0]
        text_embeds = self.text_projection(text_features_cls)

        image_embeds = image_embeds / paddle.norm(image_embeds, axis=-1, keepdim=True)
        text_embeds = text_embeds / paddle.norm(text_embeds, axis=-1, keepdim=True)
        fusion_embeds = fusion_embeds / paddle.norm(
            fusion_embeds, axis=-1, keepdim=True
        )

        output = (text_embeds, image_embeds, fusion_embeds)
        return output
