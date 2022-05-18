# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["Aircraft", "Cifar100", "Daimlerpedcls", "Dtd", "Gtsrb", 
           "Imagenet12", "Omniglot", "Svhn", "Ucf101", "VggFlowers"]


@DATASET_REGISTRY.register()
class Aircraft(ImageDataset):
    dataset_dir = 'Decathlon/aircraft'
    dataset_dir_val = 'Decathlon/aircraft'
    dataset_name = 'aircraft'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dataset_dir_val = os.path.join(self.root, self.dataset_dir_val)
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir_val, "val")
        test_dir = os.path.join(self.dataset_dir_val, "test")
        required_files = [train_dir]
        self.check_before_run(required_files)
        self.dict_label = {}
        self.label2dirname = {}
        train = self.process_dir_train(train_dir)
        query = self.process_dir_val(val_dir)
        self.infer_query = self.process_dir_test(test_dir)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir_train(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        all_dirs.sort()
        self.dict_label = {}
        self.label2dirname = {}
        for idx, dir_name in enumerate(all_dirs):
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, idx, '0'])
            self.dict_label[dir_name] = idx
            self.label2dirname[idx] = dir_name
        return data
    
    def process_dir_test(self, data_dir):
        data = []
        all_imgs = glob.glob(os.path.join(data_dir, "*.jpg"))
        for img_name in all_imgs:
            data.append([img_name, None, None])
        return data

    def process_dir_val(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        all_dirs.sort()
        for dir_name in all_dirs:
            idx = self.dict_label[dir_name]
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, idx, '0'])
        return data


@DATASET_REGISTRY.register()
class Cifar100(Aircraft):
    dataset_dir = "Decathlon/cifar100"
    dataset_dir_val = "Decathlon/cifar100"
    dataset_name = 'cifar100'


@DATASET_REGISTRY.register()
class Daimlerpedcls(Aircraft):
    dataset_dir = "Decathlon/daimlerpedcls"
    dataset_dir_val = "Decathlon/daimlerpedcls"
    dataset_name = 'daimlerpedcls'


@DATASET_REGISTRY.register()
class Dtd(Aircraft):
    dataset_dir = "Decathlon/dtd"
    dataset_dir_val = "Decathlon/dtd"
    dataset_name = 'dtd'


@DATASET_REGISTRY.register()
class Gtsrb(Aircraft):
    dataset_dir = "Decathlon/gtsrb"
    dataset_dir_val = "Decathlon/gtsrb"
    dataset_name = 'gtsrb'


@DATASET_REGISTRY.register()
class Imagenet12(Aircraft):
    dataset_dir = "Decathlon/imagenet12"
    dataset_dir_val = "Decathlon/imagenet12"
    # dataset_dir_val = "Decathlon/data/ILSVRC2012"
    dataset_name = 'imagenet12'


@DATASET_REGISTRY.register()
class Svhn(Aircraft):
    dataset_dir = "Decathlon/svhn"
    dataset_dir_val = "Decathlon/svhn"
    dataset_name = 'svhn'


@DATASET_REGISTRY.register()
class Ucf101(Aircraft):
    dataset_dir = "Decathlon/ucf101"
    dataset_dir_val = "Decathlon/ucf101"
    dataset_name = 'ucf101'


@DATASET_REGISTRY.register()
class Omniglot(Aircraft):
    dataset_dir = "Decathlon/omniglot"
    dataset_dir_val = "Decathlon/omniglot"
    dataset_name = 'omniglot'


@DATASET_REGISTRY.register()
class VggFlowers(Aircraft):
    dataset_dir = "Decathlon/vgg-flowers"
    dataset_dir_val = "Decathlon/vgg-flowers"
    dataset_name = 'vgg-flowers'