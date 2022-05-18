# encoding: utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import pandas as pd
import scipy.io as scio

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["PatchCamelyon", "GTSRB", "Fer2013", "Retinopathy", "Resisc45",
           "EuroSAT", "SVHN", "FGVCaircraft", "Caltech101", "DTD", "SUN397",
           "Oxford102Flower", "OxfordPet", "Food101", "CIFAR10", "CIFAR100", "DMLAB_full", "DMLAB_10p", "DMLAB_1p"]


@DATASET_REGISTRY.register()
class PatchCamelyon(ImageDataset):
    dataset_dir = 'intern_test/patchcamelyon'
    dataset_name = 'patchcamelyon'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'valid')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_valid):
        data = []
        data_dir = os.path.join(data_dir, train_or_valid)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name), '0'])
        return data


@DATASET_REGISTRY.register()
class GTSRB(ImageDataset):
    dataset_dir = 'intern_test/gtsrb/GTSRB_official'
    train_dir = 'GTSRB/Final_Training/Images'
    test_dir = 'GTSRB/Final_Test/Images'
    test_label = 'GT-final_test.csv'
    dataset_name = 'gtsrb'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, self.train_dir)
        self.test_dir = os.path.join(self.dataset_dir, self.test_dir)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.test_dir, self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, label_dir=None):
        data = []
        if label_dir is None:
            all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
            for dir_name in all_dirs:
                all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.ppm"))
                for img_name in all_imgs:
                    data.append([img_name, int(dir_name), '0'])
        else:
            with open(label_dir, 'r') as f:
                list_line = f.readlines()
            for idx, line in enumerate(list_line):
                if idx:
                    line = line.strip().split(';')
                    img_name = os.path.join(data_dir, line[0])
                    class_id = int(line[-1])
                    data.append([img_name, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class Fer2013(ImageDataset):
    dataset_dir = 'intern_test/fer2013'
    dataset_name = 'fer2013'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'test')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_valid):
        data = []
        data_dir = os.path.join(data_dir, train_or_valid)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name), '0'])
        return data

@DATASET_REGISTRY.register()
class Retinopathy(ImageDataset):
    dataset_dir = 'intern_test/retinopathy'
    train_name = 'train'
    train_label_name = 'trainLabels.csv'
    test_name = 'test'
    test_label_name = 'retinopathy_solution.csv'
    dataset_name = 'retinopathy'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, self.train_name)
        self.train_label_dir = os.path.join(self.dataset_dir, self.train_label_name)
        self.test_dir = os.path.join(self.dataset_dir, self.test_name)
        self.test_label_dir = os.path.join(self.dataset_dir, self.test_label_name)
        train = self.process_dir(self.train_dir, self.train_label_dir)
        query = self.process_dir_test(self.test_dir, self.test_label_dir)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        list_line = list_line[1:]
        for idx, line in enumerate(list_line):
            line = line.strip().split(',')
            img_name = os.path.join(data_dir, line[0] + '.jpeg')
            class_id = int(line[1])
            data.append([img_name, class_id, '0'])
        return data

    def process_dir_test(self, data_dir, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        list_line = list_line[1:]
        for idx, line in enumerate(list_line):
            line = line.strip().split(',')
            if line[-1] == 'Public':
                continue
            img_name = os.path.join(data_dir, line[0] + '.jpeg')
            class_id = int(line[1])
            data.append([img_name, class_id, '0'])
        return data

@DATASET_REGISTRY.register()
class Resisc45(ImageDataset):
    dataset_dir = 'intern_test/resisc45/NWPU-RESISC45'
    dataset_name = 'resisc45'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dict_label = self.init_dict_label(self.dataset_dir)
        train = self.process_lst(self.dataset_dir)
        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, data_dir):
        dict_label = {}
        list_class_id = os.listdir(data_dir)
        count = 0
        for class_id in list_class_id:
            dict_label[class_id] = count
            count += 1
        return dict_label

    def process_lst(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, self.dict_label[dir_name], '0'])
        return data


@DATASET_REGISTRY.register()
class EuroSAT(ImageDataset):
    dataset_dir = 'intern_test/eurosat/2750'
    dataset_name = 'eurosat'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dict_label = self.init_dict_label(self.dataset_dir)
        train = self.process_lst(self.dataset_dir)
        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, data_dir):
        dict_label = {}
        list_class_id = os.listdir(data_dir)
        count = 0
        for class_id in list_class_id:
            dict_label[class_id] = count
            count += 1
        return dict_label

    def process_lst(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, self.dict_label[dir_name], '0'])
        return data


@DATASET_REGISTRY.register()
class SVHN(ImageDataset):
    dataset_dir = 'intern_test/svhn'
    dataset_name = 'svhn'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.extra_dir = os.path.join(self.dataset_dir, 'extra')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        self.dict_label = self.init_dict_label(self.train_dir)
        train = self.process_dir(self.train_dir, self.extra_dir)
        query = self.process_dir(self.test_dir)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, data_dir):
        dict_label = {}
        list_class_id = os.listdir(data_dir)
        count = 0
        for class_id in list_class_id:
            dict_label[class_id] = count
            count += 1
        return dict_label

    def process_dir(self, data_dir, extra_dir=None):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, self.dict_label[dir_name], '0'])
        if extra_dir is not None:
            all_dirs = [d.name for d in os.scandir(extra_dir) if d.is_dir()]
            for dir_name in all_dirs:
                all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
                for img_name in all_imgs:
                    data.append([img_name, self.dict_label[dir_name], '0'])
        return data


@DATASET_REGISTRY.register()
class FGVCaircraft(ImageDataset):
    dataset_dir = 'intern_test/fgvcaircraft/fgvc-aircraft-2013b/data'
    image_dir = 'images'
    train_label = 'images_variant_train.txt'
    val_label = 'images_variant_val.txt'
    test_label = 'images_variant_test.txt'
    dataset_name = 'fgvcaircraft'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.train_label = os.path.join(self.dataset_dir, self.train_label)
        self.val_label = os.path.join(self.dataset_dir, self.val_label)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        self.dict_label = self.init_dict_label(self.train_label)
        train = self.process_dir(self.train_label, self.val_label)
        query = self.process_dir(self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = (' ').join(line.strip().split(' ')[1:])
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, label_dir, extra_dir=None):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip().split(' ')
            img_name = os.path.join(self.image_dir, line[0] + '.jpg')
            class_id = (' ').join(line[1:])
            data.append([img_name, self.dict_label[class_id], '0'])
        if extra_dir is not None:
            with open(extra_dir, 'r') as f:
                list_line = f.readlines()
            for line in list_line:
                line = line.strip().split(' ')
                img_name = os.path.join(self.image_dir, line[0] + '.jpg')
                class_id = (' ').join(line[1:])
                data.append([img_name, self.dict_label[class_id], '0'])
        return data


@DATASET_REGISTRY.register()
class Caltech101(ImageDataset):
    dataset_dir = 'intern_test/caltech101/Caltech101/101_ObjectCategories'
    dataset_name = 'caltech101'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dict_label = self.init_dict_label(self.dataset_dir)
        train = self.process_lst(self.dataset_dir)
        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, data_dir):
        dict_label = {}
        list_class_id = os.listdir(data_dir)
        count = 0
        for class_id in list_class_id:
            dict_label[class_id] = count
            count += 1
        return dict_label

    def process_lst(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, self.dict_label[dir_name], '0'])
        return data


@DATASET_REGISTRY.register()
class DTD(ImageDataset):
    dataset_dir = 'intern_test/dtd'
    image_dir = 'images'
    train_label = 'labels/train1.txt'
    val_label = 'labels/val1.txt'
    test_label = 'labels/test1.txt'
    dataset_name = 'dtd'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.train_label = os.path.join(self.dataset_dir, self.train_label)
        self.val_label = os.path.join(self.dataset_dir, self.val_label)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        self.dict_label = self.init_dict_label(self.train_label)
        train = self.process_dir(self.train_label, self.val_label)
        query = self.process_dir(self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = line.strip().split('/')[0]
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, label_dir, extra_dir=None):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            img_name = os.path.join(self.image_dir, line)
            class_id = line.split('/')[0]
            data.append([img_name, self.dict_label[class_id], '0'])
        if extra_dir is not None:
            with open(extra_dir, 'r') as f:
                list_line = f.readlines()
            for line in list_line:
                line = line.strip()
                img_name = os.path.join(self.image_dir, line)
                class_id = line.split('/')[0]
                data.append([img_name, self.dict_label[class_id], '0'])
        return data


@DATASET_REGISTRY.register()
class StanfordCars(ImageDataset):
    dataset_dir = 'intern_test/stanfordcars/Stanford_Cars'
    image_dir = 'cars_train'
    label_dir = 'devkit/cars_train_annos.mat'
    image_dir_test = 'cars_test'
    label_dir_test = 'devkit/cars_test_annos_withlabels.mat'
    dataset_name = 'stanfordcars'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.label_dir = os.path.join(self.dataset_dir, self.label_dir)
        self.image_dir_test = os.path.join(self.dataset_dir, self.image_dir_test)
        self.label_dir_test = os.path.join(self.dataset_dir, self.label_dir_test)
        self.dict_label = self.init_dict_label(self.label_dir)
        train = self.process_lst(self.label_dir, self.image_dir)
        query = self.process_lst(self.label_dir_test, self.image_dir_test)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        label_mat = scio.loadmat(label_dir)
        for mat in label_mat['annotations'][0]:
            img_name = os.path.join(self.image_dir, mat[5][0])
            class_id = int(mat[4][0][0])
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_lst(self, label_dir, img_dir):
        data = []
        label_mat = scio.loadmat(label_dir)
        for mat in label_mat['annotations'][0]:
            img_name = os.path.join(img_dir, mat[5][0])
            class_id = int(mat[4][0][0])
            data.append([img_name, self.dict_label[class_id], '0'])
        return data


@DATASET_REGISTRY.register()
class Oxford102Flower(ImageDataset):
    dataset_dir = 'intern_test/oxford102flower/Oxford_102_Flower/dataset'
    dataset_name = 'oxford102flower'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'valid')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_valid):
        data = []
        data_dir = os.path.join(data_dir, train_or_valid)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name) - 1, '0'])
        return data


@DATASET_REGISTRY.register()
class OxfordPet(ImageDataset):
    dataset_dir = 'intern_test/oxfordpet/Oxford_IIIT_Pet'
    image_dir = 'images'
    train_label = 'annotations/trainval.txt'
    test_label = 'annotations/test.txt'
    dataset_name = 'oxfordpet'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.train_label = os.path.join(self.dataset_dir, self.train_label)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        self.dict_label = self.init_dict_label(self.train_label)
        train = self.process_dir(self.train_label)
        query = self.process_dir(self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = ('_').join(line.strip().split('_')[:-1])
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            img_name = os.path.join(self.image_dir, line.split(' ')[0] + '.jpg')
            class_id = ('_').join(line.split('_')[:-1])
            data.append([img_name, self.dict_label[class_id], '0'])
        return data


@DATASET_REGISTRY.register()
class Food101(ImageDataset):
    dataset_dir = 'intern_test/food101/food-101'
    image_dir = 'images'
    train_label = 'meta/train.txt'
    test_label = 'meta/test.txt'
    dataset_name = 'food101'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.train_label = os.path.join(self.dataset_dir, self.train_label)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        self.dict_label = self.init_dict_label(self.train_label)
        train = self.process_dir(self.train_label)
        query = self.process_dir(self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = line.strip().split('/')[0]
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            img_name = os.path.join(self.image_dir, line + '.jpg')
            class_id = line.split('/')[0]
            data.append([img_name, self.dict_label[class_id], '0'])
        return data


@DATASET_REGISTRY.register()
class CIFAR10(ImageDataset):
    dataset_dir = 'intern_test/cifar10'
    dataset_name = 'cifar10'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'test')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_valid):
        data = []
        data_dir = os.path.join(data_dir, train_or_valid)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name), '0'])
        return data


@DATASET_REGISTRY.register()
class CIFAR100(ImageDataset):
    dataset_dir = 'intern_test/cifar100'
    dataset_name = 'cifar100'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'test')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_valid):
        data = []
        data_dir = os.path.join(data_dir, train_or_valid)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name), '0'])
        return data


@DATASET_REGISTRY.register()
class SUN397(ImageDataset):
    dataset_dir = 'intern_test/sun397'
    image_dir = 'SUN397'
    train_label = 'Training_01.txt'
    test_label = 'Testing_01.txt'
    dataset_name = 'sun397'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_dir)
        self.train_label = os.path.join(self.dataset_dir, self.train_label)
        self.test_label = os.path.join(self.dataset_dir, self.test_label)
        self.dict_label = self.init_dict_label(self.train_label)
        train = self.process_dir(self.train_label)
        query = self.process_dir(self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = line.strip().split('/')[-2]
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            img_name = os.path.join(self.image_dir, line[1:])
            class_id = line.split('/')[-2]
            data.append([img_name, self.dict_label[class_id], '0'])
        return data

@DATASET_REGISTRY.register()
class DMLAB_full(ImageDataset):
    dataset_dir = 'intern_test/dmlab_VTAB'
    dataset_name = "DMLAB_full"
    train_label = 'dmlab_train_full.filelist'
    test_label = 'dmlab_test_full.filelist'
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_filelist(f'{self.dataset_dir}/{self.train_label}')
        query = self.process_filelist(f'{self.dataset_dir}/{self.test_label}')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)
    def process_filelist(self, filelist):
        data = []
        with open(filelist, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split('\t')
            data.append([os.path.join(self.dataset_dir, path), int(class_id), '0'])
        return data


@DATASET_REGISTRY.register()
class DMLAB_10p(ImageDataset):
    dataset_dir = 'intern_test/dmlab_VTAB'
    dataset_name = "DMLAB_10p"
    train_label = 'dmlab_train_10p.filelist'
    test_label = 'dmlab_test_full.filelist'
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_filelist(f'{self.dataset_dir}/{self.train_label}')
        query = self.process_filelist(f'{self.dataset_dir}/{self.test_label}')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)
    def process_filelist(self, filelist):
        data = []
        with open(filelist, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split('\t')
            data.append([os.path.join(self.dataset_dir, path), int(class_id), '0'])
        return data


@DATASET_REGISTRY.register()
class DMLAB_1p(ImageDataset):
    dataset_dir = 'intern_test/dmlab_VTAB'
    dataset_name = "DMLAB_1p"
    train_label = 'dmlab_train_1p.filelist'
    test_label = 'dmlab_test_full.filelist'
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_filelist(f'{self.dataset_dir}/{self.train_label}')
        query = self.process_filelist(f'{self.dataset_dir}/{self.test_label}')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)
    def process_filelist(self, filelist):
        data = []
        with open(filelist, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split('\t')
            data.append([os.path.join(self.dataset_dir, path), int(class_id), '0'])
        return data
