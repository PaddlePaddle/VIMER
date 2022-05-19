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
from paddle.vision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, CenterCrop, ToTensor
from paddlenlp.transformers import BertTokenizer


class CTCDataset(Dataset):
    """
        class CTCDataset
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
        
        with open(txt_file) as f:
            for line in f:
                conts = line.strip().split("\t") 
                if len(conts) == 4:
                    image_id, sent_id, text, img_path = conts
                else:
                    raise ValueError("text error: ", txt_file)
                
                if is_train:
                    image_id = img_path.split('/')[-1].split('.')[0]
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
                Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])
        
    
    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):
        image_path = self.datas[index][0]
        caption = self.datas[index][1]
        pair_info = None
        if not self.is_train:
            pair_info = self.datas[index][2]
        
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
    
