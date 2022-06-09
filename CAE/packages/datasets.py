import os
import paddle

from paddle.vision import transforms

from util.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from util.data.transforms import RandomResizedCropAndInterpolationWithTwoPic
from util.data import create_transform

from dall_e.utils import map_pixels
from packages.masking_generator import MaskingGenerator, RandomMaskingGenerator
from packages.dataset_folder import DatasetFolder

ADE_DEFAULT_MEAN = (0.48897026, 0.46548377, 0.42939525)
ADE_DEFAULT_STD = (0.22846712, 0.22941928, 0.24038891)


class DataAugmentationForCAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = ADE_DEFAULT_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = ADE_DEFAULT_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.color_jitter > 0:
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
            ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)

        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForCAESingle(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = ADE_DEFAULT_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = ADE_DEFAULT_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.color_jitter > 0:
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(args.color_jitter, args.color_jitter,
                                       args.color_jitter),
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size,
                    second_size=None,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=None,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
            ])

        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )

        self.args = args
        if args.target_mode == 'random':
            self.vocab = paddle.load(args.target_path)['random_vector']
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __call__(self, image):
        for_patches = self.common_transform(image)
        transformed_patch = self.patch_transform(for_patches)

        if self.args.target_mode == 'random':
            normalized_img = self.visual_token_transform(for_patches)
            normalized_img = normalized_img.reshape([1, 3, self.args.window_size[0], self.args.patch_size[0], self.args.window_size[1], self.args.patch_size[1]])
            normalized_img = normalized_img.transpose([0, 2, 4, 1, 3, 5]).reshape([self.args.window_size[0] * self.args.window_size[1], -1])
            similarity = paddle.mm(normalized_img, self.vocab)
            # print(similarity.shape)     # (196, 8192)
            similarity = similarity.astype(paddle.float32)  # TODO fix amp bug
            target_id = similarity.argmax(axis=-1)
            return \
                transformed_patch, target_id, \
                self.masked_position_generator()
        else:
            return \
                transformed_patch, paddle.zeros_like(transformed_patch), \
                self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_cae_pretraining_dataset(args):
    low_case_path = args.data_path.lower()
    if 'imagenet' in low_case_path:
        if args.target_mode == 'clusterID':
            transform = DataAugmentationForCAE(args)
        elif args.target_mode == 'rgb' or args.target_mode == 'random':
            transform = DataAugmentationForCAESingle(args)
        print("Data Aug = %s" % str(transform))
        return DatasetFolder(args.data_path, transform=transform)
    elif 'ade' in low_case_path:
        raise NotImplementedError
        print('use ADE as the dataset.')
        from packages.ade20k_dataloader import ADE20K_Loader
        if args.target_mode == 'clusterID':
            transform = DataAugmentationForCAE(args)
        elif args.target_mode == 'rgb' or args.target_mode == 'random':
            transform = DataAugmentationForCAESingle(args)
        print("Data Aug = %s" % str(transform))
        return ADE20K_Loader(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        mode = 'train' if is_train else 'test'
        dataset = paddle.vision.datasets.Cifar100(args.data_path, mode=mode, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = DatasetFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = DatasetFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation='bicubic'),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
