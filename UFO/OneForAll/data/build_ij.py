"""
data/build_ij.py
"""
import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle
import cv2
from PIL import Image

from utils import comm
from fastreid.data import samplers
from fastreid.data import CommDataset
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from tools import moe_group_utils

_root = os.getenv("FASTREID_DATASETS", "datasets")

def similar_transform_umeyama(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]
    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num
    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T[:dim]


class TestIJBCDataset(CommDataset):
    def __init__(self, img_items, dict_label, transforms=None, dataset_name=None):
        self.img_items = img_items
        self.labels = dict_label
        self.dir_img_root = self.labels['dir_img_root']
        self.transforms = transforms
        self.dataset_name = dataset_name
        src = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                        [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src

    def __getitem__(self, index):
        each_line = self.img_items[index]
        name_lmk_score = each_line.strip().split(' ')
        landmark = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        landmark = landmark.reshape((5, 2))
        assert landmark.shape[0]==68 or landmark.shape[0]==5
        assert landmark.shape[1]==2
        if landmark.shape[0]==68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36]+landmark[39])/2
            landmark5[1] = (landmark[42]+landmark[45])/2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        M = similar_transform_umeyama(landmark5, self.src)
        img_path = os.path.join(self.dir_img_root, name_lmk_score[0])
        img = read_image(img_path)
        img_cv = np.array(img)
        # h, w, _ = img_cv.shape
        img_cv = cv2.warpAffine(img_cv, M, (112, 112), borderValue=0.0)
        img = Image.fromarray(img_cv)
        if self.transforms is not None: img = self.transforms(img)
        return {"images": img, "targets": float(name_lmk_score[-1]),}


def build_ijbc_test_set(dataset_name=None, transforms=None, **kwargs):
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    if comm.is_main_process():
        data.show_test()
    dict_label = {}
    dict_label['templates'] = data.templates
    dict_label['medias'] = data.medias
    dict_label['p1'] = data.p1
    dict_label['p2'] = data.p2
    dict_label['label'] = data.label
    dict_label['dir_img_root'] = data.dir_img_root
    
    #test_set = TestIJBCDataset(data.list_img, dict_label, transforms, data.dataset_name)
    #test_set.num_query = len(data.list_img)
    #test_set.num_valid_samples = test_set.num_query
    #return test_set
    #data.list_img = data.list_img[:128*8]
    # style2
    # 在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
    test_items = data.list_img
    world_size = comm.get_world_size()
    if len(test_items)%world_size != 0:
        idx_list = list(range(len(data.list_img)))
        random_idx_list = [random.choice(idx_list) for _ in range(world_size - len(test_items)%world_size)]
        test_items += [data.list_img[idx] for idx in random_idx_list]
    test_set = TestIJBCDataset(test_items, dict_label, transforms, data.dataset_name)

    # Update query number
    test_set.num_query = len(data.list_img)
    # 记录data.query和data.gallery的有效长度，在评估模块中，只取出来前有效长度的数据，丢弃末尾填充的数据
    test_set.num_valid_samples = len(data.list_img)
    return test_set
