"""vista dataloader
"""
import os
import numpy as np
from .imgtool import process_sent
from PIL import Image
import glob
import copy
from paddle.io import Dataset
from paddle.vision.transforms import (
    Compose,
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    Normalize,
    CenterCrop,
    ToTensor,
)
from paddlenlp.transformers import BertTokenizer


class LocalSceneTextDataset(Dataset):
    """ViSTA dataloader"""

    def __init__(
        self,
        txt_file,
        image_root,
        token_conf_path,
        scene_text_token_conf_path,
        max_seq_length,
        max_scene_text_length,
        image_size,
        ocr_features_path=None,
        return_ocr_feature=0,
        ocr_max_length=15,
        ocr_feature_dim=300,
        is_train=False,
        pos_path=None,
        return_2D_pos=True,
    ):
        self.datas = []
        self.is_train = is_train
        self.return_2D_pos = return_2D_pos
        self.imageID_to_pos = {}
        if return_ocr_feature == 3:
            pos_filenames = glob.glob("{}/*.txt".format(pos_path))
            for filename in pos_filenames:
                image_id = filename.split("/")[-1].split(".")[0].split("_")[1]
                self.imageID_to_pos[image_id] = []
                with open(filename) as f:
                    for line in f.readlines():
                        conts = line.strip().split(",")
                        self.imageID_to_pos[image_id].append(
                            [conts[0], conts[1], conts[4], conts[5], conts[8]]
                        )
            print(
                "{}, {} have {} ocr text".format(
                    txt_file, pos_path, len(self.imageID_to_pos.keys())
                )
            )

        ocr_feature_idx = -1
        with open(txt_file) as f:
            for line in f:
                conts = line.strip().split("\t")
                if len(conts) == 4:
                    ocr_feature_idx = -1
                    image_id, sent_id, text, img_path = conts
                elif len(conts) >= 5:
                    image_id, sent_id, text, img_path, ocr_feature_idx = conts[:5]
                # elif len(conts) == 6:
                #    image_id, sent_id, text, img_path, ocr_feature_idx, scene_texts = conts[:6]
                else:
                    raise ValueError("text error: ", txt_file)

                if return_ocr_feature == 1:
                    ocr_feature_idx = -1

                if is_train:
                    image_id = img_path.split("/")[-1].split(".")[0]
                    if return_ocr_feature == 3 and image_id not in self.imageID_to_pos:
                        print(
                            "warning: {}, {} not find ocr text".format(
                                txt_file, image_id
                            ),
                            self.imageID_to_pos,
                        )
                    self.datas.append([img_path, text, ocr_feature_idx])
                else:
                    self.datas.append([img_path, text, ocr_feature_idx, conts])

        self.tokenizer = BertTokenizer.from_pretrained(token_conf_path)
        self.scene_text_tokenizer = BertTokenizer.from_pretrained(
            scene_text_token_conf_path
        )

        self.max_seq_length = max_seq_length
        self.max_scene_text_length = max_scene_text_length

        self.image_root = image_root
        self.return_ocr_feature = return_ocr_feature
        self.ocr_feature_max_len = ocr_max_length
        self.ocr_feature_dim = ocr_feature_dim
        self.ocr_features = None

        if return_ocr_feature == 2:
            if os.path.exists(ocr_features_path):
                self.ocr_features = np.load(ocr_features_path)
                print(txt_file, " ocr_features shape: ", self.ocr_features.shape)

        print("before crop size:", int(image_size / 224.0 * 256))

        if is_train:
            self.data_transform = Compose(
                [
                    Resize(size=int(image_size / 224.0 * 256)),
                    RandomCrop(image_size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        else:
            self.data_transform = Compose(
                [
                    Resize(size=image_size),
                    CenterCrop(image_size),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.datas)

    def pre_pro_2D_pos(self, pos_tokens, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        tokens = copy.deepcopy(pos_tokens)
        while True:
            total_length = len(tokens)
            if total_length <= (max_length - 2):
                break
            tokens.pop()
        tokens.insert(0, [0, 0, 1, 1])
        tokens.append([0, 0, 1, 1])
        while len(tokens) < max_length:
            tokens.append([0, 0, 0, 0])
        pos = np.array(tokens, dtype=np.float32)
        return pos

    def pro_ocr_txt(self, image_id, w, h):
        """pro ocr pos"""
        pos_datas = []
        if image_id in self.imageID_to_pos:
            pos_datas = self.imageID_to_pos[image_id]
        else:
            print(image_id, pos_datas)

        scene_texts = []
        pos = []
        for data in pos_datas:

            x_min, y_min, x_max, y_max, scene_text = data
            scene_texts.append(scene_text)
            scene_text_split = self.scene_text_tokenizer.tokenize(scene_text)
            for i in range(len(scene_text_split)):
                pos.append(
                    [
                        int(x_min) * 1.0 / w,
                        int(y_min) * 1.0 / h,
                        int(x_max) * 1.0 / w,
                        int(y_max) * 1.0 / h,
                    ]
                )

        scene_texts = " ".join(scene_texts)
        # print(self.scene_text_tokenizer.tokenize(scene_texts))
        scene_text_ids, scene_text_mask, scene_text_segment_ids = process_sent(
            scene_texts, self.scene_text_tokenizer, self.max_scene_text_length
        )

        # print(scene_text_ids, scene_text_mask, scene_text_segment_ids)

        # Warning: this is self.max_scene_text_length
        scene_text_status = []
        if len(pos) == 0:
            scene_text_status.append(0)
        else:
            scene_text_status.append(1)
        scene_text_status = np.array(scene_text_status, dtype=np.float32)
        pos = self.pre_pro_2D_pos(pos, self.max_scene_text_length)
        return [
            scene_text_ids,
            scene_text_mask,
            scene_text_segment_ids,
            scene_text_status,
            pos,
        ]

    def __getitem__(self, index):
        image_path = self.datas[index][0]
        caption = self.datas[index][1]
        ocr_feature_idx = int(self.datas[index][2])
        pair_info = None
        if not self.is_train:
            pair_info = self.datas[index][3]

        image_path = os.path.join(self.image_root, image_path)
        image_id = image_path.split("/")[-1].split(".")[0]

        try:
            input_ids, input_mask, segment_ids = process_sent(
                caption, self.tokenizer, self.max_seq_length
            )
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            image = self.data_transform(image)
            if self.return_ocr_feature == 3:

                (
                    scene_text_ids,
                    scene_text_mask,
                    scene_text_segment_ids,
                    scene_text_status,
                    pos,
                ) = self.pro_ocr_txt(image_id, w, h)
                res = [
                    image,
                    input_ids,
                    input_mask,
                    segment_ids,
                    scene_text_ids,
                    scene_text_mask,
                    scene_text_segment_ids,
                    scene_text_status,
                ]
                if self.return_2D_pos:
                    res.append(pos)
                if not self.is_train:
                    res.append(pair_info)
                return res

            elif self.return_ocr_feature == 1 or self.return_ocr_feature == 2:
                # print("ocr_feature_idx: ", ocr_feature_idx)
                if ocr_feature_idx >= 0 and self.ocr_features is not None:
                    ocr_feature = self.ocr_features[ocr_feature_idx][
                        : self.ocr_feature_max_len
                    ][:]
                else:
                    ocr_feature = np.zeros(
                        (self.ocr_feature_max_len, self.ocr_feature_dim)
                    )
                ocr_mask = []
                for i in range(ocr_feature.shape[0]):
                    if np.all(ocr_feature[i] == 0):
                        ocr_mask.append(0)
                    else:
                        ocr_mask.append(1)
                ocr_mask = np.array(ocr_mask)
                return [
                    image,
                    input_ids,
                    input_mask,
                    segment_ids,
                    ocr_feature,
                    ocr_mask,
                    segment_ids,
                    segment_ids,
                ]
            else:
                if not self.is_train:
                    return [image, input_ids, input_mask, segment_ids, pair_info]
                else:
                    return [image, input_ids, input_mask, segment_ids]
        except Exception as e:
            print("error: ", e, image_path)
            return None, None
