# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import cv2
import copy
import numpy as np
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

class TablePostProcess(object):
    def __init__(self, save_img=False,
        row_thresh=0.3,  col_threshold=0.2,
        link_thresh=0.5):
        self.save_img = save_img
        self.row_thresh = row_thresh
        self.col_thresh = col_threshold
        self.link_thresh = link_thresh

    def get_score(self, bbox, maps):
        x1, y1, x2, y2 = bbox
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        score_map = maps[y1: y2, x1: x2]
        return score_map, np.mean(score_map)

    def del_structure(self, final_group):
        final_group = np.array(final_group)
        group_map = np.zeros(shape = (np.max(final_group[:, -2]) + 1, np.max(final_group[:, -1]) + 1))
        final_group = final_group.reshape(-1, 4).tolist()
        for idx, group in enumerate(final_group):
            start_row, start_col, end_row, end_col = group
            group_map[start_row: end_row + 1, start_col: end_col + 1] = idx + 1
        miss_row_map = np.zeros(shape=[group_map.shape[0], group_map.shape[1]])
        for i in range(1, group_map.shape[0]):
            miss_row_map[i, :] = (group_map[i, :] == group_map[i - 1, :]).astype(np.float32)
        miss_row_map = (np.sum(miss_row_map, 1) != group_map.shape[1])
        miss_col_map = np.zeros(shape=[group_map.shape[0], group_map.shape[1]])
        for i in range(group_map.shape[1] - 1):
            miss_col_map[:, i] = (group_map[:, i] == group_map[:, i - 1]).astype(np.float32)
        miss_col_map = (np.sum(miss_col_map, 0) != group_map.shape[0])
        final_group_map = group_map[miss_row_map]
        final_group_map = final_group_map[:, miss_col_map]
        final_group = []
        for i in range(1, int(np.max(final_group_map) + 1)):
            try:
                ys, xs = np.where(final_group_map == i)
                group = [min(ys), min(xs), max(ys), max(xs)]
                final_group.append(group)
            except:
                continue
        return final_group

    def merge_cell(self, cell_dict, merge_dict):
        groups = []
        for key in merge_dict:
            row_index, col_index = key
            up_score, _, left_score, _ = merge_dict[key]
            if (row_index - 1, col_index) in merge_dict:
                up_score = (up_score + merge_dict[(row_index - 1, col_index)][1]) * 0.5
            if (row_index, col_index - 1) in merge_dict:
                left_score = (left_score +  merge_dict[(row_index, col_index - 1)][3]) * 0.5
            if up_score > self.link_thresh:
                save = False
                for group in groups:
                    if key in group:
                        if (row_index, col_index + 1) in cell_dict:
                            group.append((row_index, col_index + 1))
                            save = True
                            break
                if not save:
                    temp = [key]
                    if (row_index, col_index + 1) in cell_dict:
                        temp.append((row_index, col_index + 1))
                    groups.append(temp)
            if left_score > self.link_thresh:
                save = False
                for group in groups:
                    if key in group:
                        if (row_index + 1, col_index) in cell_dict:
                            group.append((row_index + 1, col_index))
                            save = True
                            break
                if not save:
                    temp = [key]
                    if (row_index + 1, col_index) in cell_dict:
                        temp.append((row_index + 1, col_index))
                    groups.append(temp)
            if up_score <= self.link_thresh and left_score <= self.link_thresh:
                save = True
                for group in groups:
                    if key in group:
                        save = False
                        break
                if save:
                    groups.append([key])
        final_gruop = []
        for group in groups:
            group = np.array(group)
            new_cell = [np.min(group[:, 0]), np.min(group[:, 1]), np.max(group[:, 0]), np.max(group[:, 1])]
            final_gruop.append(new_cell)

        final_bboxes = {}
        for key in final_gruop:
            start_row, start_col, end_row, end_col = key
            temp_bbox_1 = cell_dict[(start_row, 0)]
            temp_bbox_2 = cell_dict[(end_row, 0)]
            temp_bbox_3 = cell_dict[(0, start_col)]
            temp_bbox_4 = cell_dict[(0, end_col)]
            x1 = temp_bbox_3[0]
            y1 = temp_bbox_1[1]
            x2 = temp_bbox_4[-2]
            y2 = temp_bbox_2[-1]

            final_bbox = [x1, y1, x2, y2]
            final_bboxes[(start_row, start_col, end_row, end_col)] = final_bbox
        try:
            final_gruop = self.del_structure(final_gruop)
        except:
            final_gruop = final_gruop
        return final_bboxes, self.get_html(final_gruop)


    def get_html(self, groups):
        bboxes = []
        row_index = -1
        res_html = []
        for group in groups:
            start_row, start_col, end_row, end_col = group
            if row_index < start_row:
                if row_index == -1:
                    res_html.append('<tr>')
                else:
                    res_html.append('</tr>')
                    res_html.append('<tr>')

                row_span = end_row - start_row
                col_span = end_col - start_col

                if row_span + col_span > 0:
                    res_html.append('<td')
                    if row_span > 0:
                        res_html.append(' rowspan="' + str(row_span + 1) + '"')
                    if col_span > 0:
                        res_html.append(' colspan="' + str(col_span + 1) + '"')
                    res_html.append('>')
                    res_html.append('</td>')
                else:
                    res_html.append('<td>')
                    res_html.append('</td>')
            else:
                row_span = end_row - start_row
                col_span = end_col - start_col

                if row_span + col_span > 0:
                    res_html.append('<td')
                    if row_span > 0:
                        res_html.append(' rowspan="' + str(row_span + 1) + '"')
                    if col_span > 0:
                        res_html.append(' colspan="' + str(col_span + 1) + '"')
                    res_html.append('>')
                    res_html.append('</td>')
                else:
                    res_html.append('<td>')
                    res_html.append('</td>')
            row_index = start_row
        res_html.append('</tr>')
        return res_html

    def get_table_res(self, preds):
        row_probs = preds['row_probs']
        row_probs = row_probs[0].numpy()
        col_probs = preds['col_probs']
        col_probs = col_probs[0].numpy()
        row_m_probs = preds['row_m_probs']
        row_m_probs = row_m_probs[0].numpy()
        col_m_probs = preds['col_m_probs']
        col_m_probs = col_m_probs[0].numpy()

        link_up = preds['link_up']
        link_up = link_up[0][0].numpy()
        link_down = preds['link_down']
        link_down = link_down[0][0].numpy()
        link_left = preds['link_left']
        link_left = link_left[0][0].numpy()
        link_right = preds['link_right']
        link_right = link_right[0][0].numpy()

        show_row_lines = (row_m_probs > self.row_thresh).astype(np.float32).reshape(-1).tolist()
        show_col_lines = (col_m_probs > self.col_thresh).astype(np.float32).reshape(-1).tolist()

        show_link_up =  link_up
        show_link_down = link_down
        show_link_left = link_left
        show_link_right = link_right

        def get_lines(lines):
            res= []
            start = False
            for index, item in enumerate(lines):
                if item == 1 and start == False:
                    res.append(index)
                    start = True
                if item == 0 and start == True:
                    res.append(index - 1)
                    start = False
            if lines[-1] == 1:
                res.append(len(lines) - 1)
            return res

        row_lines = np.array(get_lines(show_row_lines)).reshape(-1, 2)
        col_lines = np.array(get_lines(show_col_lines)).reshape(-1, 2)

        row_lines[:, 1] = np.maximum(row_lines[:, 1], row_lines[:, 0] + 1)
        col_lines[:, 1] = np.maximum(col_lines[:, 1], col_lines[:, 0] + 1)
        row_lines, col_lines =  row_lines.tolist(), col_lines.tolist()

        if len(row_lines) == 0:
            row_lines = [[0, 2], [495, 500]]

        def merge_line(lines, gap):
            res = [lines[0]]
            pre = lines[0]
            for i in range(1, len(lines)):
                # print('pre', pre, 'lines[i]', lines[i])
                if abs(lines[i][0] - pre[1]) < gap:
                    # print('merge')
                    pre_line = res.pop(-1)
                    new_line = [pre_line[0], lines[i][1]]
                    res.append(new_line)
                else:
                    res.append(lines[i])
                pre = lines[i]
            return res

        row_lines, col_lines = merge_line(row_lines, 5), merge_line(col_lines, 5)

        corners = {}
        for row_index, row_line in enumerate(row_lines[1:]):
            for col_index, col_line in enumerate(col_lines[1:]):
                row_start, row_end = row_line #[int(x ) for x in row_line]
                col_start, col_end = col_line #[int(x * col_width) for x in col_line]
                corners[row_index, col_index] = [col_start, row_start, col_end, row_end]

        merge_dict = {}
        for key in corners:
            _, up_score = self.get_score(corners[key], show_link_up)
            _, down_score = self.get_score(corners[key], show_link_down)
            _, left_score = self.get_score(corners[key], show_link_left)
            _, right_score = self.get_score(corners[key], show_link_right)
            merge_dict[key] = (up_score, down_score, left_score, right_score)

        cell_row_lines  = []
        for index, row_item in enumerate(np.array(row_lines).reshape(-1, 2).tolist()):
            # if row_item == 1:
            if index == 0:
                row_item = min(row_item[0], 499)
            elif index == np.array(row_lines).shape[0] - 1:
                row_item = min(row_item[1], 499)
            else:
                row_item = (row_item[0] + row_item[1]) // 2
            cell_row_lines.append(row_item)

        cell_col_lines = []
        for index, col_item in enumerate(np.array(col_lines).reshape(-1, 2).tolist()):
            if index == 0:
                col_item = min(col_item[0], 249)

            elif index == np.array(col_lines).shape[0] - 1:
                col_item = min(col_item[1], 249)

            else:
                col_item = (col_item[0] + col_item[1]) // 2
            cell_col_lines.append(col_item)

        cell_dict = {}
        for row_index, row_line in enumerate(cell_row_lines[:-1]):
            for col_index, col_line in enumerate(cell_col_lines[:-1]):
                cell_dict[row_index, col_index] = [cell_col_lines[col_index], cell_row_lines[row_index], cell_col_lines[col_index + 1], cell_row_lines[row_index + 1]]
        final_bboxes, pred_html = self.merge_cell(cell_dict, merge_dict)

        #return final_bboxes, pred_html
        return pred_html

    def __call__(self, outs_dict):
        if isinstance(outs_dict, list):
            table_res_list = []
            for outs_dict_item in outs_dict:
                table_res_list.append(self.get_table_res(outs_dict_item))
            return table_res_list
        else:
            return self.get_table_res(outs_dict)
