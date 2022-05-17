# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["ImageNet1k", "ImageNet1kBD"]


@DATASET_REGISTRY.register()
class ImageNet1k(ImageDataset):
    """This is a demo dataset for smoke test, you can refer to
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    dataset_dir = 'ILSVRC2012'
    dataset_name = "imagenet1k"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")
        required_files = [train_dir]
        self.check_before_run(required_files)
        self.dict_label = {}
        train = self.process_dir_train(train_dir)
        query = self.process_dir_val(val_dir)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir_train(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        all_dirs.sort()
        self.dict_label = {}
        for idx, dir_name in enumerate(all_dirs):
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.JPEG"))
            for img_name in all_imgs:
                data.append([img_name, idx, '0'])
            self.dict_label[dir_name] = idx
        return data

    def process_dir_val(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        all_dirs.sort()
        for dir_name in all_dirs:
            idx = self.dict_label[dir_name]
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.JPEG"))
            for img_name in all_imgs:
                data.append([img_name, idx, '0'])
        return data


@DATASET_REGISTRY.register()
class ImageNet1kBD(ImageDataset):
    """This is a demo dataset for smoke test, you can refer to
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    dataset_dir = 'ILSVRC2012'
    dataset_name = "imagenet1kbd"
    train_label = 'train_list.txt'
    test_label = 'val_list.txt'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")
        required_files = [train_dir]
        self.check_before_run(required_files)
        self.dict_label = self.init_dict_label(f'{self.dataset_dir}/{self.train_label}')
        train = self.process_dir(train_dir, f'{self.dataset_dir}/{self.train_label}')
        query = self.process_dir(val_dir, f'{self.dataset_dir}/{self.test_label}')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split(' ')
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, img_dir, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split(' ')
            img_name = os.path.join(img_dir, path)
            data.append([img_name, self.dict_label[class_id], '0'])
        return data