"""
ctc_dataset.py
"""
import os
import random
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


class CTCDataset(Dataset):
    """
    class CTCDataset
    """

    def __init__(
        self,
        text_file,
        image_root,
        token_conf_path,
        max_seq_length,
        image_size,
        scene_text_token_conf_path,
        max_scene_text_length,
        ocr_path,
        return_ocr_pos=False,
        is_train=False,
    ):
        self.datas = []
        self.is_train = is_train
        
        if not os.path.exists(text_file):
            raise ValueError("text error: ", text_file)
        
        with open(text_file) as f:
            for line in f:
                conts = line.strip().split("\t")
                if len(conts) == 4:
                    image_id, sent_id, text, img_path = conts
                elif len(conts) == 5:
                    image_id, sent_id, text, img_path, _ = conts
                else:
                    raise ValueError("text error: ", text_file)

                if is_train:
                    self.datas.append([img_path, text])
                else:
                    self.datas.append([img_path, text, conts])

        self.tokenizer = BertTokenizer.from_pretrained(token_conf_path)
        self.scene_text_tokenizer = BertTokenizer.from_pretrained(
            scene_text_token_conf_path
        )
        self.max_seq_length = max_seq_length
        self.max_scene_text_length = max_scene_text_length
        self.image_root = image_root
        self.return_ocr_pos = return_ocr_pos

        # Read ocr data
        self.image_id_to_pos = {}
        pos_filenames = glob.glob("{}/*.txt".format(ocr_path))
        for filename in pos_filenames:
            image_id = filename.split("/")[-1].split(".")[0].split("_")[-1]
            self.image_id_to_pos[image_id] = []
            with open(filename) as f:
                for line in f.readlines():
                    conts = line.strip().split(",")
                    self.image_id_to_pos[image_id].append(
                        [conts[0], conts[1], conts[4], conts[5], conts[8]]
                    )
        print(
            "{}, {} have {} ocr text".format(
                text_file, ocr_path, len(self.image_id_to_pos.keys())
            )
        )

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

    def process_ocr_pos(self, pos_tokens, max_length):
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

    def process_ocr_text(self, image_id, w, h):
        """pro ocr pos"""
        pos_datas = []
        if image_id in self.image_id_to_pos:
            pos_datas = self.image_id_to_pos[image_id]
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
        scene_text_ids, scene_text_mask, scene_text_segment_ids = process_sent(
            scene_texts, self.scene_text_tokenizer, self.max_scene_text_length
        )

        pos = self.process_ocr_pos(pos, self.max_scene_text_length)
        return [scene_text_ids, scene_text_mask, scene_text_segment_ids, pos]

    def __getitem__(self, index):
        image_path = self.datas[index][0]
        caption = self.datas[index][1]
        pair_info = None
        if not self.is_train:
            pair_info = self.datas[index][2]

        image_path = os.path.join(self.image_root, image_path)
        image_id = image_path.split("/")[-1].split(".")[0]

        try:
            input_ids, input_mask, segment_ids = process_sent(
                caption, self.tokenizer, self.max_seq_length
            )
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            image = self.data_transform(image)

            (
                scene_text_ids,
                scene_text_mask,
                scene_text_segment_ids,
                pos,
            ) = self.process_ocr_text(image_id, w, h)
            res = [
                image,
                input_ids,
                input_mask,
                segment_ids,
                scene_text_ids,
                scene_text_mask,
                scene_text_segment_ids,
            ]
            if self.return_ocr_pos:
                res.append(pos)

            if not self.is_train:
                res.append(pair_info)
                return res
            else:
                return res
        except Exception as e:
            print("error: ", e, image_path)
            return None, None
