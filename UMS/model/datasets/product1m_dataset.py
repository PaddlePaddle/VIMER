"""
    product1m_dataset.py for constructing reader as LocalSceneTextDataset
"""
import os
import random
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, CenterCrop, ToTensor
import numpy as np
from .imgtool import process_sent
from PIL import Image
from paddlenlp.transformers import BertTokenizer
import glob
import copy
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Product1MDataset(Dataset):
    """
        class LocalSceneTextDataset
    """

    def __init__(
            self,
            txt_file,
            image_root,
            token_conf_path,
            max_seq_length,
            image_size,
            is_train=False,
    ):
        self.datas = []
        self.is_train = is_train

        self.imageID_to_pos = {}

        # tmp_count = 0

        with open(txt_file) as f:
            for line in f:
                conts = line.strip().split("#####")
                if len(conts) == 4:
                    image_id, text, web_address, data_address = conts
                elif len(conts) == 5:
                    image_id, text, web_address, data_address, instance_text = conts
                else:
                    raise ValueError("text error: ", txt_file)

                img_path = image_id + ".jpg"


                if is_train:
                    # image_id = img_path.split('/')[-1].split('.')[0]
                    self.datas.append([img_path, text])
                else:
                    self.datas.append([img_path, text, conts])

        self.tokenizer = BertTokenizer.from_pretrained(token_conf_path)

        self.max_seq_length = max_seq_length

        self.image_root = image_root

        print("before crop size:", int(image_size / 224.0 * 256))

        if is_train:
            self.data_transform = Compose([
                Resize(size=int(image_size / 224.0 * 256)),
                RandomCrop(image_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            self.data_transform = Compose([
                Resize(size=image_size),
                CenterCrop(image_size),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):
        image_path = self.datas[index][0]
        caption = self.datas[index][1]
        pair_info = None
        # print("###### ", image_path, caption)
        if not self.is_train:
            pair_info = self.datas[index][2]

        image_name = image_path
        if self.is_train:
            for char in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
                image_root = self.image_root + char
                image_path = os.path.join(image_root, image_name)
                if os.path.exists(image_path):
                    break
        else:
            image_path = os.path.join(self.image_root, image_path)
        image_id = image_path.split('/')[-1].split('.')[0]

        try:
            input_ids, input_mask, segment_ids = process_sent(caption, \
                                                              self.tokenizer, self.max_seq_length)
            image = Image.open(image_path).convert('RGB')
            w, h = image.size
            image = self.data_transform(image)
            if not self.is_train:
                return [image, input_ids, input_mask, segment_ids, pair_info]
            else:
                return [image, input_ids, input_mask, segment_ids]
        except Exception as e:
            print("error: ", e, image_path)
            return None, None
