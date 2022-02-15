# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import bcolz
import numpy as np

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset


__all__ = ["MS1MV2", "CPLFW", "VGG2_FP", "AgeDB_30", "CALFW", "CFP_FF", "CFP_FP", "LFW"]


@DATASET_REGISTRY.register()
class MS1MV2(ImageDataset):
    """MS1MV2 dataset
    """
    dataset_dir = "ms1mv2"
    dataset_name = "ms1mv2"

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        # train = self.process_dirs()[:10000]
        train = self.process_dirs()
        super().__init__(train, [], [], **kwargs)

    def process_dirs(self):
        """process_dirs
        """
        train_list = []

        fid_list = os.listdir(self.dataset_dir)

        for fid in fid_list:
            all_imgs = glob.glob(os.path.join(self.dataset_dir, fid, "*.jpg"))
            for img_path in all_imgs:
                train_list.append([img_path, self.dataset_name + '_' + fid, '0'])

        return train_list


@DATASET_REGISTRY.register()
class CPLFW(ImageDataset):
    """CPLFW dataset
    """
    dataset_dir = "faces_emore_val"
    dataset_name = "cplfw"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        required_files = [self.dataset_dir]
        self.check_before_run(required_files)
        filelist_path = os.path.join(self.dataset_dir, "{}.filelist".format(self.dataset_name))
        label_path = os.path.join(self.dataset_dir, "{}_label.npy".format(self.dataset_name))
        with open(filelist_path) as fin:
            filelist = [p.strip().split(' ') for p in fin]
        self.is_same = np.load(label_path)
        self.img_paths = self.process_dirs(filelist)
        super().__init__([], [], [], **kwargs)

    def process_dirs(self, filelist):
        """process_dirs
        """
        img_paths = []
        for meta in filelist:
            img_path = os.path.join(self.dataset_dir, meta[0])
            img_paths.append(img_path)
        return img_paths


@DATASET_REGISTRY.register()
class VGG2FP(CPLFW):
    """VGG2_FP dataset
    """
    dataset_name = "vgg2_fp"
VGG2_FP=VGG2FP

@DATASET_REGISTRY.register()
class AgeDB30(CPLFW):
    """AgeDB_30 dataset
    """
    dataset_name = "agedb_30"
AgeDB_30=AgeDB30

@DATASET_REGISTRY.register()
class CALFW(CPLFW):
    """CALFW dataset
    """
    dataset_name = "calfw"


@DATASET_REGISTRY.register()
class CFPFF(CPLFW):
    """CFP_FF dataset
    """
    dataset_name = "cfp_ff"
CFP_FF=CFPFF

@DATASET_REGISTRY.register()
class CFPFP(CPLFW):
    """CFP_FP dataset
    """
    dataset_name = "cfp_fp"
CFP_FP=CFPFP

@DATASET_REGISTRY.register()
class LFW(CPLFW):
    """LFW dataset
    """
    dataset_name = "lfw"