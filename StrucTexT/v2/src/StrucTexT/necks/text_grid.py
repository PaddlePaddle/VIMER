""" text_grid """
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle as P
from paddle import nn
import paddle.nn.functional as F

class TexTGrid(nn.Layer):
    """ TexTGrid """
    def __init__(self, shape, shrink_ratio, **kwargs):
        super(TexTGrid, self).__init__()
        self.shape = shape
        self.shrink_ratio = shrink_ratio

    def forward(self, input, polys, indexs):
        """
        Args:
        input (text embeddings):
        polys (bbox tensor):
        indexs (ids with mask, zers is pad):
        """
        assert(input.ndim == 3) #(B,L,D)
        assert(polys.ndim == 3) #(B,N,P)
        assert(indexs.ndim == 2) #(B,L)
        assert(indexs.max() <= polys.shape[1])

        h, w = self.shape
        bs, l, d = input.shape
        text_grid = P.zeros(shape=[bs, h, w, d], dtype='float32')
        for i, (x, poly, index) in enumerate(zip(input, polys, indexs)):
            max_num = index.max()
            for j in range(max_num):
                bbox = P.cast(poly[j] * self.shrink_ratio, 'int32')
                mask = P.nonzero(P.cast(index == j + 1, 'int32'))
                embeddings = P.index_select(x, mask).unsqueeze((0, 1)) #(1, 1, S, D)
                minX, maxX = bbox[0::2].min(), bbox[0::2].max()
                minY, maxY = bbox[1::2].min(), bbox[1::2].max()
                W, H = maxX - minX + 1, maxY - minY + 1
                embeddings = F.interpolate(embeddings, data_format='NHWC', size=(H, W))
                text_grid[i, minY:minY + H, minX:minX + W] = embeddings
        text_grid = text_grid.transpose((0, 3, 1, 2))

        return {'out': text_grid, 'additional_info': {'text_feat': input}}
