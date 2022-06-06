"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import six
import cv2
import math
import random
import logging
import numpy as np

from PIL import Image
from functools import partial

from paddle.vision.transforms import ColorJitter as RawColorJitter
from .autoaugment import ImageNetPolicy
from .functional import augmentations

__all__ = ['DecodeImage',
           'NRTRDecodeImage',
           'NormalizeImage',
           'ToCHWImage',
           'DetResizeForTest',
           'E2EResizeForTest',
           'KeepKeys',
           'UnifiedResize',
           'ResizeImage',
           'CropImage',
           'RandCropImage',
           'RandFlipImage',
           'AugMix',
           'AutoAugment',
           'ColorJitter',
           'RandomSampleAndCrop',
           'TableLabelEncode',
           'ResizeTableImage']

class DecodeImage(object):
    """ decode image """

    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        if six.PY2 and type(img) is str or six.PY3 and type(img) is bytes:
            img = np.frombuffer(img, dtype='uint8')
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class NRTRDecodeImage(object):
    """ decode image """

    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')

        img = cv2.imdecode(img, 1)

        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        data['image'] = img
        return data


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    """ KeepKeys """
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    """ DetResizeForTest """
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs['resize_long']
        elif 'image_pad_shape' in kwargs:
            self.image_shape = kwargs['image_pad_shape']
            self.resize_type = 3
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        elif self.resize_type == 3:
            img, [ratio_h, ratio_w] = self.resize_image_type3(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        if 'line_bboxes' in data and len(data['line_bboxes']) > 0:
            line_bboxes = np.asarray(data['line_bboxes'])
            new_line_bboxes = []
            for line_bbox in line_bboxes:
                line_bbox = np.asarray(line_bbox)
                scale = np.tile([ratio_w, ratio_h], line_bbox.shape[-1] // 2)
                line_bbox = line_bbox * scale[np.newaxis,]
                new_line_bboxes.append(line_bbox.tolist())
            data['line_bboxes'] = new_line_bboxes
        if 'token_bboxes' in data and len(data['token_bboxes']) > 0:
            for idx, seq_bboxes in enumerate(data['token_bboxes']):
                seq_bboxes = np.array(seq_bboxes)
                scale = np.tile([ratio_w, ratio_h], seq_bboxes.shape[-1] // 2)
                seq_bboxes = seq_bboxes * scale[np.newaxis,]
                data['token_bboxes'][idx] = seq_bboxes.tolist()

        return data

    def resize_image_type1(self, img):
        """ resize_image_type1 """
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        """ resize_image_type2 """
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]

    def resize_image_type3(self, img):
        """ resize_image_type3 """
        h, w, c = img.shape
        resize_h, resize_w = self.image_shape
        ratio = float(min(resize_h, resize_w)) / float(max(h, w))

        img_pad = np.zeros((resize_h, resize_w, 3), dtype=np.float32)

        resize_w = int(w * ratio)
        resize_h = int(h * ratio)
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        img_pad[:resize_h, :resize_w, :] = img

        return img_pad, [ratio, ratio]


class ResizeTableImage(object):
    """ ResizeTableImage """
    def __init__(self, max_len, **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len

    def resize_img_table(self, img, bbox_list, xs, ys, max_len):
        height, width = img.shape[0:2]
        ratio_h = max_len / height * 1.0
        ratio_w = max_len / width * 1.0
        img_new = cv2.resize(img, (max_len, max_len))
        bbox_list_new = []
        xs = [x * ratio_w for x in xs]
        ys = [y * ratio_h for y in ys]
        for bno in range(len(bbox_list)):
            left, top, right, bottom = bbox_list[bno].copy()
            left = int(left * ratio_w)
            top = int(top * ratio_h)
            right = int(right * ratio_w)
            bottom = int(bottom * ratio_h)
            bbox_list_new.append([left, top, right, bottom])
        return img_new, bbox_list_new, xs, ys

    def resize_img_table_only(self, img, max_len):
        height, width = img.shape[0:2]
        ratio_h = max_len / height * 1.0
        ratio_w = max_len / width * 1.0
        img_new = cv2.resize(img, (max_len, max_len))

        return img_new

    def __call__(self, data):
        img = data['image']
        if 'bboxes' not in data:
            bboxes = []
        else:
            bboxes = data['bboxes']
        bbox_list = bboxes
        if 'xs' in data:
            xs = data['xs']
            ys = data['ys']
            img_new, bbox_list_new, new_xs, new_ys = self.resize_img_table(img, bbox_list, xs, ys, self.max_len)
            data['image'] = img_new
            data['bboxes'] = bbox_list_new
            data['max_len'] = self.max_len
            data['xs'] = new_xs
            data['ys'] = new_ys
        else:
            img_new = self.resize_img_table_only(img, self.max_len)
            data['image'] = img_new
        return data


class E2EResizeForTest(object):
    """ E2EResizeForTest """
    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs['max_side_len']
        self.valid_set = kwargs['valid_set']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.valid_set == 'totaltext':
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len)
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len)
        data['image'] = im_resized
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):
        """ resize_image_for_totaltext """

        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)


class UnifiedResize(object):
    def __init__(self, size, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }

        def _pil_resize(src, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:
            logging.warning(
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            self.resize_func = cv2.resize

    def __call__(self, data):
        img = data['image']
        data['image'] = self.resize_func(img)
        return data


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise ValueError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, data):
        img = data['image']
        channel_first = False
        if img.ndim == 3 and img.shape[0] == 3:
            channel_first = True
            img = np.transpose(img, (1, 2, 0))
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        img = self._resize_func(img, (w, h))
        data['image'] = np.transpose(img, (2, 0, 1)) if channel_first else img
        return data


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, data):
        img = data['image']
        w, h = self.size
        channel_first = False
        if img.ndim == 3 and img.shape[0] == 3:
            channel_first = True
            img = np.transpose(img, (1, 2, 0))
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        img = img[h_start:h_end, w_start:w_end, :]
        data['image'] = np.transpose(img, (2, 0, 1)) if channel_first else img
        return data


class RandCropImage(object):
    """ random crop image """

    def __init__(self,
                 size,
                 scale=None,
                 ratio=None,
                 interpolation=None,
                 backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, data):
        img = data['image']
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        channel_first = False
        if img.ndim == 3 and img.shape[0] == 3:
            channel_first = True
            img = np.transpose(img, (1, 2, 0))
        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = self._resize_func(img[j:j + h, i:i + w, :], size)
        data['image'] = np.transpose(img, (2, 0, 1)) if channel_first else img
        return data


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, data):
        img = data['image']
        if random.randint(0, 1) == 1:
            img = cv2.flip(img, self.flip_code)
            data['image'] = img
        return data


class AutoAugment(object):
    def __init__(self):
        self.policy = ImageNetPolicy()

    def __call__(self, data):
        img = data['image']
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        img = self.policy(img)
        img = np.asarray(img)
        data['image'] = img
        return data


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self,
                 prob=0.5,
                 aug_prob_coeff=0.1,
                 mixture_width=3,
                 mixture_depth=1,
                 aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.augmentations = augmentations
        # fmt: on

    def __call__(self, data):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return data

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(
            np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        image = Image.fromarray(data['image'])
        mix = np.zeros(image.shape)
        for i in range(self.mixture_width):
            image_aug = image.copy()
            image_aug = Image.fromarray(image_aug)
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        data['image'] = mixed.astype(np.uint8)
        return data


class ColorJitter(RawColorJitter):
    """ColorJitter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        img = data['image']
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)
        img = super()._apply_image(img)
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        data['image'] = img
        return data


class RandomSampleAndCrop(object):
    """ RandomSampleAndCrop
    random sampling tokens, and crop the sampled region.
    """
    def __init__(self, **kwargs):
        super(RandomSampleAndCrop, self).__init__()
        self.sample_num = kwargs['sample_num']
        self.task_name = kwargs['task_name']
        self.extend_pixel = 10 if "extend_pixel" not in kwargs else \
                                int(kwargs['extend_pixel'])
        self.crop_ratio = 0 if "crop_ratio" not in kwargs else \
                                float(kwargs["crop_ratio"])
        assert self.task_name in ["t2i", "endet"]

    def __call__(self, data):
        img = data['image']
        tokens = data['tokens']
        line_bboxes = data['line_bboxes']

        real_sample_num = [0] * self.sample_num
        sent_len = len(tokens)
        if sent_len >= self.sample_num:
            sample_idx = random.sample(range(sent_len), self.sample_num)
        else:
            sample_idx = list(range(sent_len))
            sample_idx += (np.ones(self.sample_num - sent_len).astype(np.int32) * -1).tolist()
            np.random.shuffle(sample_idx)
        assert len(sample_idx) == self.sample_num

        new_tokens = []
        new_line_bboxes = []
        for i, j in enumerate(sample_idx):
            if j == -1:
                continue
            new_tokens.append(tokens[j])
            new_line_bboxes.append(line_bboxes[j])
            real_sample_num[i] = 1

        data['line_bboxes'] = new_line_bboxes
        data['tokens'] = new_tokens
        data['mask_num'] = real_sample_num

        rand_crop = np.random.rand()
        if rand_crop < self.crop_ratio:
            img_h, img_w, _, = img.shape
            min_x, min_y, max_x, max_y = [0] * 4
            if self.task_name == "t2i":
                new_line_bboxes = np.asarray(new_line_bboxes, dtype=np.int32)
                min_xy = new_line_bboxes.min(axis=0)
                max_xy = new_line_bboxes.max(axis=0)
                min_x = int(max(0, min(min_xy[::2]) - self.extend_pixel))
                min_y = int(max(0, min(min_xy[1::2]) - self.extend_pixel))
                max_x = int(min(img_w, max(max_xy[::2]) + self.extend_pixel))
                max_y = int(min(img_h, max(max_xy[1::2]) + self.extend_pixel))
                new_line_bboxes = new_line_bboxes - [min_x, min_y] * (new_line_bboxes.shape[-1] // 2)
                new_line_bboxes = new_line_bboxes.tolist()
            elif self.task_name == "endet":
                minmax_xy = []
                endet_line_bboxes = []
                for line_bbox in new_line_bboxes:
                    line_bbox = np.asarray(line_bbox, dtype=np.int32)
                    min_xy = line_bbox.min(axis=0)
                    max_xy = line_bbox.max(axis=0)
                    minmax_xy.append([min(min_xy[::2]), min(min_xy[1::2]), \
                                      max(max_xy[::2]), max(max_xy[1::2])])
                minmax_xy = np.asarray(minmax_xy, dtype=np.int32)
                min_xy = minmax_xy.min(axis=0)
                max_xy = minmax_xy.max(axis=0)
                min_x = int(max(0, min(min_xy[::2]) - self.extend_pixel))
                min_y = int(max(0, min(min_xy[1::2]) - self.extend_pixel))
                max_x = int(min(img_w, max(max_xy[::2]) + self.extend_pixel))
                max_y = int(min(img_h, max(max_xy[1::2]) + self.extend_pixel))
                for line_bbox in new_line_bboxes:
                    line_bbox = np.asarray(line_bbox, dtype=np.int32)
                    line_bbox = line_bbox - [min_x, min_y] * (line_bbox.shape[-1] // 2)
                    endet_line_bboxes.append(line_bbox.tolist())
                new_line_bboxes = endet_line_bboxes

            new_img = img[min_y:max_y, min_x:max_x, :]
            data['image'] = new_img
            data['line_bboxes'] = new_line_bboxes

        return data


class TableLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_cell_num,
                 **kwargs):
        self.max_cell_num = max_cell_num

    def __call__(self, data):
        bboxes = data['bboxes']
        self.row_lines = 500
        self.col_lines = 250
        bbox_list = np.zeros((self.max_cell_num, 4), dtype=np.float32)
        row_col_index_list = np.zeros((self.max_cell_num, 2))
        row_col_indexs = data['row_col_indexs']
        row_maps = np.zeros((self.row_lines))
        col_maps = np.zeros((self.col_lines))
        xs = data['xs']
        ys = data['ys']
        xs_cls = data['xs_cls']
        ys_cls = data['ys_cls']
        link_up = np.zeros(shape= (self.row_lines, self.col_lines))
        link_down = np.zeros(shape=(self.row_lines, self.col_lines))
        link_left = np.zeros(shape=(self.row_lines, self.col_lines))
        link_right = np.zeros(shape=(self.row_lines, self.col_lines))
        link_mask = np.zeros(shape=(self.row_lines, self.col_lines))
        col_link_map_left = data['col_link_map_left']
        col_link_map_right = data['col_link_map_right']
        row_link_map_up = data['row_link_map_up']
        row_link_map_down = data['row_link_map_down']
        bbox_list_mask = np.zeros(
            (self.max_cell_num, 1), dtype=np.float32)
        img_height, img_width, img_ch = data['image'].shape
        row_height = img_height / 500
        col_width = img_width / 250
        for y_index, y in enumerate(ys):
            row_index = int(y // row_height)
            row_maps[row_index] = ys_cls[y_index]
        row_maps = row_maps.tolist()
        max_row = max(row_maps)
        final_row_maps = np.zeros((self.row_lines))
        final_row_maps_mountain = np.zeros((self.row_lines))

        rows_line = []
        cols_line = []
        for i in range(1, int(max_row) + 1):
            temp = []
            for j, item in enumerate(row_maps):
                if item == i:
                    temp.append(j)
            start_id, end_id = min(temp), max(temp)
            end_id = min(max(start_id + 4, end_id), self.row_lines - 1)
            final_row_maps[start_id: end_id] = 1
            if end_id - start_id <= 2:
                final_row_maps_mountain[start_id: end_id + 1] = 1
            else:
                for index in range(start_id, end_id + 1):
                    mid_pos = start_id + (end_id - start_id) // 2
                    final_row_maps_mountain[index] = min(1, (1. - (abs(index - mid_pos) / abs(start_id - mid_pos))) + 1e-2)
            rows_line.append(start_id)
            rows_line.append(end_id)

        for x_index, x in enumerate(xs):
            col_index = int(x // col_width)
            col_maps[col_index] = xs_cls[x_index]

        col_maps = col_maps.tolist()
        max_col = max(col_maps)
        final_col_maps = np.zeros((self.col_lines))
        final_col_maps_mountain = np.zeros((self.col_lines))
        for i in range(1, int(max_col) + 1):
            temp = []
            for j, item in enumerate(col_maps):
                if item == i:
                    temp.append(j)
            start_id, end_id = min(temp), max(temp)
            if end_id + 4 < self.col_lines - 1:
                end_id = min(max(start_id + 4, end_id), self.col_lines -1 )
            else:
                end_id = min(max(start_id + 4, end_id), self.col_lines -1 )
                start_id = end_id - 4
            final_col_maps[start_id: end_id] = 1
            if end_id - start_id <= 2:
                final_col_maps_mountain[start_id: end_id + 1] = 1
            else:
                for index in range(start_id, end_id + 1):
                    mid_pos = start_id + (end_id - start_id) // 2
                    final_col_maps_mountain[index] = min(1, (1. - (abs(index - mid_pos) / abs(start_id - mid_pos))) + 1e-2)
            cols_line.append(start_id)
            cols_line.append(end_id)

        rows = []
        cols = []
        for row_index in range(len(rows_line) - 1):
            rows.append((rows_line[row_index], rows_line[row_index + 1]))

        for col_index in range(len(cols_line) - 1):
            cols.append((cols_line[col_index], cols_line[col_index + 1]))

        rows_line = np.array(rows_line).reshape(-1, 2).tolist()#[1:]
        cols_line = np.array(cols_line).reshape(-1, 2).tolist()#[1:]
        for row_index, row_item in enumerate(rows_line[1:]):
            for col_index, col_item in enumerate(cols_line[1:]):
                row_start, row_end = row_item
                col_start, col_end = col_item
                link_up[row_start: row_end, col_start:col_end] = col_link_map_right[row_index, col_index]
                link_left[row_start: row_end, col_start:col_end] = row_link_map_down[row_index, col_index]
        for row_index, row_item in enumerate(rows_line):
            for col_index, col_item in enumerate(cols_line):
                row_start, row_end = row_item
                col_start, col_end = col_item
                link_down[row_start: row_end, col_start:col_end] = col_link_map_right[row_index, col_index - 1] if row_index < col_link_map_right.shape[0] and col_index - 1 < col_link_map_right.shape[1] else 0
                link_right[row_start: row_end, col_start:col_end] = row_link_map_down[row_index - 1, col_index] if row_index - 1 < row_link_map_down.shape[0] and col_index < row_link_map_down.shape[1] else 0
                link_mask[row_start: row_end, col_start:col_end] = 1

        data['row_col_indexs'] = row_col_index_list
        data['row_maps'] = final_row_maps
        data['col_maps'] = final_col_maps
        data['link_up'] = link_up
        data['link_down'] = link_down
        data['link_left'] = link_left
        data['link_right'] = link_right
        data['link_mask'] = link_mask
        data['final_row_maps_mountain'] = final_row_maps_mountain
        data['final_col_maps_mountain'] = final_col_maps_mountain
        return data

    def encode(self, text, char_or_elem):
        """convert text-label into text-index.
        """
        if char_or_elem == "char":
            max_len = self.max_text_length
            current_dict = self.dict_character
        else:
            max_len = self.max_elem_length
            current_dict = self.dict_elem
        if len(text) > max_len:
            return None
        if len(text) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        text_list = []
        for char in text:
            if char not in current_dict:
                return None
            text_list.append(current_dict[char])
        if len(text_list) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        return text_list

    def get_ignored_tokens(self, char_or_elem):
        beg_idx = self.get_beg_end_flag_idx("beg", char_or_elem)
        end_idx = self.get_beg_end_flag_idx("end", char_or_elem)
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end, char_or_elem):
        if char_or_elem == "char":
            if beg_or_end == "beg":
                idx = np.array(self.dict_character[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_character[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of char" \
                              % beg_or_end
        elif char_or_elem == "elem":
            if beg_or_end == "beg":
                idx = np.array(self.dict_elem[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_elem[self.end_str])
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of elem" \
                              % beg_or_end
        else:
            assert False, "Unsupport type %s in char_or_elem" \
                              % char_or_elem
        return idx
