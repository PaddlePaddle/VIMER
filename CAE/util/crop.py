import math

import paddle

from paddle.vision import transforms
from paddle.vision.transforms import functional as F


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif F._is_numpy_image(img):
        return img.shape[:2][::-1]
    elif F._is_tensor_image(img):
        return img.shape[1:][::-1]  # chw
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomResizedCrop(transforms.BaseTransform):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 keys=None):
        super(RandomResizedCrop, self).__init__(keys)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        assert (scale[0] <= scale[1]), "scale should be of kind (min, max)"
        assert (ratio[0] <= ratio[1]), "ratio should be of kind (min, max)"
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_param(self, img):
        #if isinstance(img, tuple):
        #    img = img[0]
        width, height = _get_image_size(img)
        area = height * width

        target_area = area * paddle.empty([1]).uniform_(self.scale[0],
                                                      self.scale[1]).item()
        log_ratio = paddle.log(paddle.to_tensor(self.ratio))
        aspect_ratio = paddle.exp(
            paddle.empty([1]).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = paddle.randint(0, height - h + 1, shape=(1, )).item()
        j = paddle.randint(0, width - w + 1, shape=(1, )).item()

        return i, j, h, w

    def _apply_image(self, img):
        i, j, h, w = self._get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, self.interpolation)
