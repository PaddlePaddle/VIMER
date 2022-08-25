# !/usr/bin/env python3
"""
# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

"""
################# LIBRARIES ###############################
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import copy
import random
import os
from paddle.io import Dataset
from PIL import Image
import itertools
import math
import paddle.vision.transforms as transforms
import paddle

################ FUNCTION TO RETURN ALL DATALOADERS NECESSARY ####################
def give_dataloaders(dataset, args):
    """
    Args:
        dataset: string, name of dataset for which the dataloaders should be returned.
        args:     argparse.Namespace, contains all training-specific parameters.
    Returns:
        dataloaders: dict of dataloaders for training, testing and evaluation on training.
    """
    if args.dataset == "inshop_dataset":
        datasets = give_InShop_datasets(args)
    elif args.dataset == "Stanford_Online_Products":
        datasets = give_OnlineProducts_datasets_hard(args)
    else:
        raise Exception("No Dataset >{}< available!".format(args.dataset))

    # Move datasets to dataloaders.
    dataloaders = {}

    for key, dataset in datasets.items():
        dataloaders[key] = paddle.io.DataLoader(
            dataset, batch_size=args.bs, num_workers=0, shuffle=False, drop_last=False
        )
    return dataloaders


def give_InShop_datasets(args):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the In-Shop Clothes dataset.
    For Metric Learning, training and test sets are provided by one text file, list_eval_partition.txt.
    So no random shuffling of classes.
    Args:
        args: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing (by query and gallery separation) and evaluation.
    """
    # Load train-test-partition text file.
    assert os.path.exists(args.source_path+'/list_eval_partition.txt'), f"{args.source_path+'/list_eval_partition.txt'} NOT found."
    data_info = np.array(
        pd.read_table(
            args.source_path + "/list_eval_partition.txt",
            header=1,
            delim_whitespace=True,
        )
    )[1:, :]
    # Separate into training dataset and query/gallery dataset for testing.
    train, query, gallery = (
        data_info[data_info[:, 2] == "train"][:, :2],
        data_info[data_info[:, 2] == "query"][:, :2],
        data_info[data_info[:, 2] == "gallery"][:, :2],
    )

    # Generate conversions
    lab_conv = {
        x: i
        for i, x in enumerate(
            np.unique(np.array([int(x.split("_")[-1]) for x in train[:, 1]]))
        )
    }
    train[:, 1] = np.array([lab_conv[int(x.split("_")[-1])] for x in train[:, 1]])

    lab_conv = {
        x: i
        for i, x in enumerate(
            np.unique(
                np.array(
                    [
                        int(x.split("_")[-1])
                        for x in np.concatenate([query[:, 1], gallery[:, 1]])
                    ]
                )
            )
        )
    }
    query[:, 1] = np.array([lab_conv[int(x.split("_")[-1])] for x in query[:, 1]])
    gallery[:, 1] = np.array([lab_conv[int(x.split("_")[-1])] for x in gallery[:, 1]])

    # Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict = {}
    train_cnt = 0
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(args.source_path + "/" + img_path)
        train_cnt += 1
    print("train", train_cnt, len(train_image_dict))
    query_val_cnt = 0
    query_image_dict = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(args.source_path + "/" + img_path)
        query_val_cnt += 1

    gallery_val_cnt = 0
    gallery_image_dict = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(args.source_path + "/" + img_path)
        gallery_val_cnt += 1
    print("val query-gallery cnt", query_val_cnt, gallery_val_cnt)
    query_dataset = BaseTripletDataset(query_image_dict, args, is_validation=True)
    gallery_dataset = BaseTripletDataset(gallery_image_dict, args, is_validation=True)
    eval_dataset = BaseTripletDataset(train_image_dict, args, is_validation=True)
    return {
        "testing_query": query_dataset,
        "evaluation": eval_dataset,
        "testing_gallery": gallery_dataset,
    }


def give_OnlineProducts_datasets_hard(args):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Online-Products dataset.
    For Metric Learning, training and test sets are provided by given text-files, Ebay_train.txt & Ebay_test.txt.
    So no random shuffling of classes.

    Args:
        args: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    assert os.path.exists(args.source_path + "/Ebay_train.txt"), f"{args.source_path + '/Ebay_train.txt'} NOT found."
    image_sourcepath = args.source_path
    training_files = pd.read_table(
        args.source_path + "/Ebay_train.txt", header=0, delimiter=" "
    )
    test_files = pd.read_table(
        args.source_path + "/Ebay_test.txt", header=0, delimiter=" "
    )

    # Generate Conversion dict.
    conversion = {}
    for class_id, path in zip(training_files["class_id"], training_files["path"]):
        conversion[class_id] = path.split("/")[0]
    for class_id, path in zip(test_files["class_id"], test_files["path"]):
        conversion[class_id] = path.split("/")[0]

    # Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_cnt = 0
    train_image_dict, val_image_dict = {}, {}
    for key, img_path in zip(training_files["class_id"], training_files["path"]):
        key = key - 1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath + "/" + img_path)
        train_cnt += 1
    print(" train class {} nums".format(len(train_image_dict)))  # 11318
    print("train_cnt {} nums".format(train_cnt))  # 59551
    #
    val_cnt = 0
    for key, img_path in zip(test_files["class_id"], test_files["path"]):
        key = key - 1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_cnt += 1
        val_image_dict[key].append(image_sourcepath + "/" + img_path)
    print(" val class {} nums".format(len(val_image_dict)))  # 11316
    print("val_cnt {} nums".format(val_cnt))  # 60502

    super_dict = {}
    for cid, scid, path in zip(
        training_files["class_id"],
        training_files["super_class_id"],
        training_files["path"],
    ):
        cid = cid - 1
        scid = scid - 1
        if not scid in super_dict.keys():
            super_dict[scid] = {}
        if not cid in super_dict[scid].keys():
            super_dict[scid][cid] = []
        super_dict[scid][cid].append(image_sourcepath + "/" + path)

    val_dataset = BaseTripletDataset(val_image_dict, args, is_validation=True)
    eval_dataset = BaseTripletDataset(train_image_dict, args, is_validation=True)

    # train_dataset.conversion = conversion
    val_dataset.conversion = conversion
    eval_dataset.conversion = conversion

    return {
        "testing": val_dataset,
        "evaluation": eval_dataset,
    }


################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """

    def __init__(self, image_dict, args, samples_per_class=8, is_validation=False):
        """
        Dataset Init-Function.

        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            args:                argparse.Namespace, contains all training-specific parameters.
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        # Define length of dataset
        self.n_files = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.pars = args
        self.image_dict = image_dict

        self.avail_classes = sorted(list(self.image_dict.keys()))

        # Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict = {
            i: self.image_dict[key] for i, key in enumerate(self.avail_classes)
        }
        self.avail_classes = sorted(list(self.image_dict.keys()))

        # Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            self.current_class = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        # Data augmentation/processing methods.
        transf_list = []
        self.transform = transforms.Compose(
            [
                transforms.Resize(248),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [
            [(x, key) for x in self.image_dict[key]] for key in self.image_dict.keys()
        ]
        self.image_list = [x for y in self.image_list for x in y]

        # Flag that denotes if dataset is called for the first time.
        self.is_init = True

    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.

        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size) == 2:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.is_init:
            self.current_class = self.avail_classes[idx % len(self.avail_classes)]
            self.is_init = False
        if not self.is_validation:
            if self.samples_per_class == 1:
                return self.image_list[idx][-1], self.transform(
                    self.ensure_3dim(Image.open(self.image_list[idx][0]))
                )

            if self.n_samples_drawn == self.samples_per_class:
                # Once enough samples per class have been drawn, we choose another class to draw samples from.
                # Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                # previously or one before that.
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter:
                        counter.remove(prev_class)
                self.current_class = counter[idx % len(counter)]
                self.classes_visited = self.classes_visited[1:] + [self.current_class]
                self.n_samples_drawn = 0
            class_sample_idx = idx % len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1
            out_img = self.transform(
                self.ensure_3dim(
                    Image.open(self.image_dict[self.current_class][class_sample_idx])
                )
            )
            return self.current_class, out_img
        else:
            return self.image_list[idx][-1], self.transform(
                self.ensure_3dim(Image.open(self.image_list[idx][0]))
            )

    def __len__(self):
        return self.n_files
