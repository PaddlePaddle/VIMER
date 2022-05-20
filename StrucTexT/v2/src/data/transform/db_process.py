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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import six
import cv2
import random
import pyclipper
import numpy as np
import imgaug
import imgaug.augmenters as iaa

from shapely.geometry import Polygon


class DBTextSpottingEncode(object):
    """ DBTextSpottingEncode """
    def __init__(
        self,
        max_bbox_num,
        max_seq_len,
        size=(640, 640),
        max_tries=50,
        min_crop_side_ratio=0.1,
        keep_ratio=True,
        shrink_ratio=0.4,
        thresh_min=0.3,
        thresh_max=0.7,
        min_text_size=8,
        **kwargs):
        self.max_bbox_num = max_bbox_num
        self.max_seq_len = max_seq_len

        # crop config
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

        # DB label config
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

        self.min_text_size = min_text_size

    def draw_img_polys(self, im, text_polys=None):
        """text_polys: [num, 4, 2]
        """
        bbox_color = (255, 0, 255)

        if text_polys is not None:
            for idx in range(text_polys.shape[0]):
                poly = text_polys[idx]
                x1 = int(poly[0][0])
                y1 = int(poly[0][1])
                x2 = int(poly[2][0])
                y2 = int(poly[2][1])
                cv2.rectangle(im, (x1, y1), (x2, y2), bbox_color, 2)

    def is_poly_in_rect(self, poly, x, y, w, h):
        """is_poly_in_rect
        """
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        """is_poly_outside_rect
        """
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        """split_regions
        """
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        """random_select
        """
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        """region_wise_random_select
        """
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys, min_crop_side_ratio, max_tries):
        """crop_area
        """
        h, w, _ = im.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return {"crop_x": 0,
                "crop_y": 0,
                "crop_w": w,
                "crop_h": h}

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                            ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return {"crop_x": xmin,
                    "crop_y": ymin,
                    "crop_w": xmax - xmin,
                    "crop_h": ymax - ymin}

        return {"crop_x": 0,
            "crop_y": 0,
            "crop_w": w,
            "crop_h": h}

    def validate_polygons(self, polygons, ignore_tags, h, w):
        """polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        """
        compute polygon area
        """
        area = 0
        q = polygon[-1]
        for p in polygon:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2.0

    def draw_border_map(self, polygon, canvas, mask):
        """draw border map
        """
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (
            1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(
                0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(
                0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                             xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def _distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[
            1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[
            1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(
            point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
            2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        return result

    def extend_line(self, point_1, point_2, result, shrink_ratio):
        """extend_line
        """
        ex_point_1 = (int(
            round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
                      int(
                          round(point_1[1] + (point_1[1] - point_2[1]) * (
                              1 + shrink_ratio))))
        cv2.line(
            result,
            tuple(ex_point_1),
            tuple(point_1),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0)
        ex_point_2 = (int(
            round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (
                              1 + shrink_ratio))))
        cv2.line(
            result,
            tuple(ex_point_2),
            tuple(point_2),
            4096.0,
            1,
            lineType=cv2.LINE_AA,
            shift=0)
        return ex_point_1, ex_point_2

    def __call__(self, data):
        """call
        """
        im = data['image']
        has_multi = 'labels' in data.keys()
        if has_multi:
            labels = data['labels']
            text_polys_list = [l['polys'] for l in labels]
            texts_list = [l['texts'] for l in labels]
            ignore_tags_list = [l['ignore_tags'] for l in labels]
            classes_list = [l['classes'] for l in labels]
        else:
            text_polys_list = [data['polys']]
            texts_list = [data['texts']]
            ignore_tags_list = [data['ignore_tags']]
            classes_list = [data['classes']]
        org_data = data
        org_data['labels'] = []

        total_text_polys = np.concatenate(text_polys_list, axis=0)
        total_ignore_tags = np.concatenate(ignore_tags_list, axis=0)

        h, w = im.shape[:2]
        for i, (text_polys, ignore_tags) in enumerate(zip(text_polys_list, ignore_tags_list)):
            text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
            if ignore_tags.sum() == ignore_tags.shape[0]:
                return None
            text_polys_list[i] = text_polys
            ignore_tags_list[i] = ignore_tags

        ori_im_shape = im.shape
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Resize((1.0, 2.0)),
            iaa.Sometimes(0.1,
                iaa.Rotate((-15, 15))
                ),
            iaa.Sometimes(0.1,
                iaa.Rot90((1, 3), keep_size=False)
                ),
            ]).to_deterministic()

        im = seq.augment_image(im)
        total_all_care_polys = []
        for i, (text_polys, ignore_tags) in enumerate(zip(text_polys_list, ignore_tags_list)):
            new_polys = []
            for poly in text_polys:
                keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
                new_keypoints = seq.augment_keypoints(
                        [imgaug.KeypointsOnImage(keypoints, shape=ori_im_shape)])[0].keypoints
                new_poly = [(p.x, p.y) for p in new_keypoints]
                new_polys.append(new_poly)

            text_polys = np.array(new_polys, dtype=np.double)
            text_polys_list[i] = text_polys

            all_care_polys = [
                text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
            ]
            total_all_care_polys.append(all_care_polys)

        # caculate crop area
        crop_xywh_dict = self.crop_area(
            im, total_all_care_polys[-1], self.min_crop_side_ratio, self.max_tries)
        crop_x = crop_xywh_dict['crop_x']
        crop_y = crop_xywh_dict['crop_y']
        crop_w = crop_xywh_dict['crop_w']
        crop_h = crop_xywh_dict['crop_h']

        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)

        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], im.shape[2]),
                              im.dtype)
            padimg[:h, :w] = cv2.resize(
                im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            im = padimg
        else:
            im = cv2.resize(
                im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))

        for text_polys, texts, ignore_tags, classes in zip(text_polys_list, texts_list, ignore_tags_list, classes_list):
            # crop text
            text_polys_crop = []
            ignore_tags_crop = []
            texts_crop = []
            classes_crop = []
            for poly, text, tag, text_class in zip(text_polys, texts, ignore_tags, classes):
                poly = ((poly - (crop_x, crop_y)) * scale).tolist()
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    text_polys_crop.append(poly)
                    ignore_tags_crop.append(tag)
                    texts_crop.append(text)
                    classes_crop.append(text_class)

            text_polys = np.array(text_polys_crop)
            ignore_tags = np.array(ignore_tags_crop)
            texts = np.array(texts_crop)
            classes = np.array(classes_crop)

            # make_border_map
            canvas = np.zeros(im.shape[:2], dtype=np.float32)
            mask = np.zeros(im.shape[:2], dtype=np.float32)

            for i in range(len(text_polys)):
                if ignore_tags[i]:
                    continue
                try:
                    self.draw_border_map(text_polys[i], canvas, mask=mask)
                except:
                    return None
            canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

            data = {}
            data['threshold_map'] = canvas
            data['threshold_mask'] = mask

            # make shrink map
            h, w = im.shape[:2]
            gt = np.zeros((h, w), dtype=np.float32)
            mask = np.ones((h, w), dtype=np.float32)
            for i in range(len(text_polys)):
                polygon = text_polys[i]
                height = max(polygon[:, 1]) - min(polygon[:, 1])
                width = max(polygon[:, 0]) - min(polygon[:, 0])
                if ignore_tags[i] or min(height, width) < self.min_text_size:
                    cv2.fillPoly(mask,
                                 polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                else:
                    polygon_shape = Polygon(polygon)
                    subject = [tuple(l) for l in polygon]
                    padding = pyclipper.PyclipperOffset()
                    padding.AddPath(subject, pyclipper.JT_ROUND,
                                    pyclipper.ET_CLOSEDPOLYGON)
                    shrinked = []

                    # Increase the shrink ratio every time we get multiple polygon returned back
                    possible_ratios = np.arange(self.shrink_ratio, 1,
                                                self.shrink_ratio)
                    np.append(possible_ratios, 1)
                    for ratio in possible_ratios:
                        distance = polygon_shape.area * (
                            1 - np.power(ratio, 2)) / polygon_shape.length
                        shrinked = padding.Execute(-distance)
                        if len(shrinked) == 1:
                            break

                    if shrinked == []:
                        cv2.fillPoly(mask,
                                     polygon.astype(np.int32)[np.newaxis, :, :], 0)
                        ignore_tags[i] = True
                        continue

                    for each_shirnk in shrinked:
                        shirnk = np.array(each_shirnk).reshape(-1, 2)
                        cv2.fillPoly(gt, [shirnk.astype(np.int32)], 1)

            data['shrink_map'] = gt
            data['shrink_mask'] = mask

            bboxes_padded_list = np.zeros((self.max_bbox_num, 4), dtype=np.float32)
            bboxes_4pts_padded_list = np.zeros((self.max_bbox_num, 8), dtype=np.float32)
            texts_padded_list = np.zeros((self.max_bbox_num, self.max_seq_len), dtype=np.int64)
            classes_padded_list = np.zeros((self.max_bbox_num), dtype=np.int64)
            masks_padded_list = np.zeros((self.max_bbox_num), dtype=np.float32)

            valid_cnt = 0
            data_list = list(zip(text_polys, texts, ignore_tags, classes))

            for i, (poly, text, tag, text_class) in enumerate(data_list):
                if valid_cnt < self.max_bbox_num:
                    if tag == True:
                        continue
                    bboxes_4pts_padded_list[valid_cnt, :] = poly.reshape(-1)
                    texts_padded_list[valid_cnt, :] = text
                    classes_padded_list[valid_cnt] = text_class
                    masks_padded_list[valid_cnt] = 1
                    valid_cnt += 1
                else:
                    break

            data['bboxes_padded_list'] = bboxes_padded_list
            data['bboxes_4pts_padded_list'] = bboxes_4pts_padded_list
            data['texts_padded_list'] = texts_padded_list
            data['classes_padded_list'] = classes_padded_list
            data['masks_padded_list'] = masks_padded_list
            org_data['labels'].append(data)

        if not has_multi:
            labels = org_data['labels']
            del org_data['labels']
            org_data.update(labels)
        data = org_data
        # normalize image and to CHW
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im / 255
        im -= img_mean
        im /= img_std
        im = im.transpose((2, 0, 1)).astype(np.float32)
        data['image'] = im
        return data


class DBTextSpottingTest(object):
    """DBTextSpottingTest
    only support batchsize 1s
    """
    def __init__(
        self,
        max_side_len,
        max_bbox_num,
        max_seq_len,
        **kwargs):
        self.max_bbox_num = max_bbox_num
        self.max_seq_len = max_seq_len
        self.max_side_len = max_side_len

    def preprocess(self, im):
        """preprocess"""
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.max_side_len) / resize_h
        else:
            ratio = float(self.max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, [ratio_h, ratio_w]

    def __call__(self, data):
        im = data['image']
        has_multi = 'labels' in data.keys()
        if has_multi:
            labels = data['labels']
            text_polys_list = [l['polys'] for l in labels]
            texts_list = [l['texts'] for l in labels]
            text_tags_list = [l['ignore_tags'] for l in labels]
            classes_list = [l['classes'] for l in labels]
        else:
            text_polys_list = [data['polys']]
            texts_list = [data['texts']]
            text_tags_list = [data['ignore_tags']]
            classes_list = [data['classes']]
        org_data = data
        org_data['labels'] = []

        h, w, _ = im.shape

        # pad and resize image
        im, ratio = self.preprocess(im)
        new_h, new_w, _ = im.shape
        ratio_h, ratio_w = ratio
        for i, text_polys, in enumerate(text_polys_list):
            text_polys[:, :, 0] *= ratio_w
            text_polys[:, :, 1] *= ratio_h
            text_polys_list[i] = text_polys

        # normalize img
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im / 255
        im -= img_mean
        im /= img_std
        im = im.transpose((2, 0, 1)).astype(np.float32)

        for text_polys, texts, text_tags, classes in zip(text_polys_list, texts_list, text_tags_list, classes_list):
            data = {}
            # [num, 4]
            bboxes_padded_list = np.zeros((self.max_bbox_num, 4), dtype=np.float32)
            bboxes_4pts_padded_list = np.zeros((self.max_bbox_num, 8), dtype=np.float32)
            texts_padded_list = np.zeros((self.max_bbox_num, self.max_seq_len), dtype=np.int64)
            classes_padded_list = np.zeros((self.max_bbox_num), dtype=np.int64)
            masks_padded_list = np.zeros((self.max_bbox_num), dtype=np.float32)

            valid_cnt = 0
            # FIXME valid_cnt may be greater than max_bbox_num
            for i, (poly, text, tag, text_class) in enumerate(zip(text_polys, texts, text_tags, classes)):
                bboxes_padded_list[valid_cnt, :] = poly[::2].reshape(-1)
                bboxes_4pts_padded_list[valid_cnt, :] = poly.reshape(-1)
                texts_padded_list[valid_cnt, :] = text
                classes_padded_list[valid_cnt] = text_class
                masks_padded_list[valid_cnt] = 1
                valid_cnt += 1

            data['bboxes_padded_list'] = bboxes_padded_list
            data['bboxes_4pts_padded_list'] = bboxes_4pts_padded_list
            data['texts_padded_list'] = texts_padded_list
            data['classes_padded_list'] = classes_padded_list
            data['masks_padded_list'] = masks_padded_list
            org_data['labels'].append(data)

        if not has_multi:
            labels = org_data['labels']
            del org_data['labels']
            org_data.update(labels)
        data = org_data
        data['image'] = im

        return data
