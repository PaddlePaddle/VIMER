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

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ["DF20", "IWildCam2020", "TsingHuaDogs", "FoodX251", "CompCars",
           "INat2021", "ImageNet21k", "Herbarium2021", "Place365"]


@DATASET_REGISTRY.register()
class DF20(ImageDataset):
    dir_image = 'intern_train/df20/DF20'
    dir_csv_train = 'intern_train/df20/DF20-train_metadata_PROD-2.csv'
    dir_csv_val = 'intern_train/df20/DF20-public_test_metadata_PROD-2.csv'
    dataset_name = 'df20'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dir_image = os.path.join(self.root, self.dir_image)
        self.dir_csv_train = os.path.join(self.root, self.dir_csv_train)
        self.dir_csv_val = os.path.join(self.root, self.dir_csv_val)
        train = self.process_csv(self.dir_csv_train)
        query = self.process_csv(self.dir_csv_val)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_csv(self, dir_csv):
        data = []
        metadata = pd.read_csv(dir_csv)
        image_paths = metadata['image_path']
        class_ids = metadata['class_id']
        for idx in range(len(class_ids)):
            image_path = os.path.join(self.dir_image, image_paths[idx])
            data.append([image_path, int(class_ids[idx]), '0'])
        return data


@DATASET_REGISTRY.register()
class IWildCam2020(ImageDataset):
    dataset_dir = 'intern_train/iwildcam2020/train_crop'
    dataset_name = 'iwildcam2020'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir)
        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, int(dir_name), '0'])
        return data


@DATASET_REGISTRY.register()
class TsingHuaDogs(ImageDataset):
    dataset_dir = 'intern_train/tsinghuadogs/low-resolution'
    dir_lst_train = 'intern_train/tsinghuadogs/TrainAndValList/train.lst'
    dir_lst_val = 'intern_train/tsinghuadogs/TrainAndValList/validation.lst'
    dataset_name = 'tsinghuadogs'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dir_lst_train = os.path.join(self.root, self.dir_lst_train)
        self.dir_lst_val = os.path.join(self.root, self.dir_lst_val)
        self.dict_label = self.init_dict_label(self.dir_lst_train)
        train = self.process_lst(self.dir_lst_train)
        query = self.process_lst(self.dir_lst_val)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, dir_lst):
        dict_label = {}
        count = 0
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = line.split('-')[1]
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_lst(self, dir_lst):
        data = []
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            line = '/'.join(line.strip().split('/')[-2:])
            img_path = os.path.join(self.dataset_dir, line)
            class_id = self.dict_label[line.split('-')[1]]
            data.append([img_path, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class FoodX251(ImageDataset):
    dataset_dir = 'intern_train/foodx251'
    dir_lst_train = 'intern_train/foodx251/train_info.csv'
    dir_lst_val = 'intern_train/foodx251/val_info.csv'
    dataset_name = 'foodx251'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dir_lst_train = os.path.join(self.root, self.dir_lst_train)
        self.dir_lst_val = os.path.join(self.root, self.dir_lst_val)
        train = self.process_lst(self.dir_lst_train, 'train_set')
        query = self.process_lst(self.dir_lst_val, 'val_set')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_lst(self, dir_lst, train_or_val):
        data = []
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip().split(',')
            img_path = os.path.join(self.dataset_dir, train_or_val, line[0])
            class_id = int(line[1])
            data.append([img_path, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class CompCars(ImageDataset):
    dataset_dir = 'intern_train/compcars/image'
    dir_lst_train = 'intern_train/compcars/train_test_split/classification/train.txt'
    dir_lst_val = 'intern_train/compcars/train_test_split/classification/test.txt'
    dataset_name = 'compcars'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dir_lst_train = os.path.join(self.root, self.dir_lst_train)
        self.dir_lst_val = os.path.join(self.root, self.dir_lst_val)
        self.dict_label = self.init_dict_label(self.dir_lst_train)
        train = self.process_lst(self.dir_lst_train)
        query = self.process_lst(self.dir_lst_val)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, dir_lst):
        dict_label = {}
        count = 0
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            class_id = ('/').join(line.strip().split('/')[:-2])
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_lst(self, dir_lst):
        data = []
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            img_path = os.path.join(self.dataset_dir, line)
            class_id = self.dict_label[('/').join(line.split('/')[:-2])]
            data.append([img_path, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class INat2021(ImageDataset):
    dataset_dir = 'intern_train/inat2021'
    dataset_name = 'inat2021'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir, 'train')
        query = self.process_dir(self.dataset_dir, 'val')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir, train_or_val):
        data = []
        data_dir = os.path.join(data_dir, train_or_val)
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            class_id = int(dir_name.split('_')[0])
            for img_name in all_imgs:
                data.append([img_name, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class ImageNet21k(ImageDataset):
    dataset_dir = '.'
    dir_lst_train = 'imagenet21k_train.txt'
    dir_lst_val = 'imagenet21k_val.txt'
    dataset_name = 'imagenet21k'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dir_lst_train = os.path.join(self.dataset_dir, self.dir_lst_train)
        self.dir_lst_val = os.path.join(self.dataset_dir, self.dir_lst_val)
        train = self.process_lst(self.dir_lst_train)
        query = self.process_lst(self.dir_lst_val)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_lst(self, dir_lst):
        data = []
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip().split(' ')
            img_path = os.path.join(self.dataset_dir, 'imagenet22k-pth', line[0])
            class_id = int(line[1])
            data.append([img_path, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class Herbarium2021(ImageDataset):
    dataset_dir = 'intern_train/herbarium2021/train/images'
    dataset_name = 'herbarium2021'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train = self.process_dir(self.dataset_dir)
        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            data_subdir = os.path.join(data_dir, dir_name)
            all_subdirs = [d.name for d in os.scandir(data_subdir) if d.is_dir()]
            for subdir_name in all_subdirs:
                all_imgs = glob.glob(os.path.join(data_subdir, subdir_name, "*.jpg"))
                class_id = int(dir_name + subdir_name)
                for img_name in all_imgs:
                    data.append([img_name, class_id, '0'])
        return data


@DATASET_REGISTRY.register()
class Place365(ImageDataset):
    dataset_dir = 'intern_train/place365'
    dir_lst_train = 'places365_train_challenge.txt'
    dir_lst_val = 'places365_val.txt'
    dataset_name = 'places365'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.dir_lst_train = os.path.join(self.dataset_dir, self.dir_lst_train)
        self.dir_lst_val = os.path.join(self.dataset_dir, self.dir_lst_val)
        train = self.process_lst(self.dir_lst_train, 'data_large')
        query = self.process_lst(self.dir_lst_val, 'val_large')
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def process_lst(self, dir_lst, train_or_val):
        data = []
        with open(dir_lst, 'r', encoding='utf8') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip().split(' ')
            if train_or_val == 'data_large':
                img_name = line[0][1:]
            else:
                img_name = line[0]
            img_path = os.path.join(self.dataset_dir, train_or_val, img_name)
            class_id = int(line[1])
            data.append([img_path, class_id, '0'])
        return data
