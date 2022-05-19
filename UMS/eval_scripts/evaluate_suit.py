#!/usr/bin/env python3
from __future__ import division
import sys
import os
import io
import os.path
import numpy as np
import json
import shutil
import argparse
import pdb


# input_dir = sys.argv[1]
# output_dir = "/data1/xl/product_retrieval/evaluate"
#
# submit_dir = "/data1/xl/product_retrieval/evaluate"
# truth_dir = "/data1/xl/product_retrieval/all_data"


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", default="eval_results_product1m.txt", type=str)
    # parser.add_argument("--retrieval_result_dir", default=".", type=str)
    parser.add_argument("--GT_dir", default="./Product1M", type=str)
    parser.add_argument("--ckpt", default="", type=str)

    return parser.parse_args()


def compute_p(rank_list, pos_set, topk):
    """compute_p"""
    intersect_size = 0
    for i in range(topk):
        if rank_list[i] in pos_set:
            intersect_size += 1

    p = float(intersect_size / topk)

    return p


def compute_ap(rank_list, pos_set, topk):
    """compute_ap"""
    intersect_size = 0
    ap = 0

    for i in range(topk):
        if rank_list[i] in pos_set:
            intersect_size += 1
            precision = intersect_size / (i + 1)
            ap += precision
    if intersect_size == 0:
        return 0
    ap /= intersect_size

    return ap


def compute_HitRate(rank_label_set, query_label_set):
    """compute_HitRate"""
    return len(rank_label_set.intersection(query_label_set)) / len(query_label_set)


def compute_ar(rank_list, GT_labels, gallery_label_ids, gallery_id_label, N):
    """compute_ar"""
    rank_list = rank_list[:N]

    GT_label_set = list(set(GT_labels))
    label_expect_count = {}
    for label in GT_labels:
        if label not in label_expect_count:
            label_expect_count[label] = int(N / len(GT_labels))
        else:
            label_expect_count[label] += int(N / len(GT_labels))

    for label in label_expect_count:
        label_expect_count[label] = min(label_expect_count[label], len(gallery_label_ids[label]))

    label_retrieval_count = {}
    for id in rank_list:
        label = gallery_id_label[id]["label"][0]
        if label in label_expect_count:
            if label not in label_retrieval_count:
                label_retrieval_count[label] = 1
            else:
                label_retrieval_count[label] += 1

    for label in label_retrieval_count:
        label_retrieval_count[label] = min(label_retrieval_count[label], label_expect_count[label])

    label_ap = 0

    for label in label_retrieval_count:
        if label_expect_count[label]:
            label_ap += label_retrieval_count[label] / label_expect_count[label]

    label_ap /= len(GT_label_set)
    return label_ap


def get_retrieval_id_list():
    """get_retrieval_id_list"""

    prefix1 = 'product1m_test'
    prefix2 = 'product1m_gallery'

    test_image_features = np.load("{}_image_features.npy".format(prefix1))
    test_text_features = np.load("{}_text_features.npy".format(prefix1))
    gallery_image_features = np.load("{}_image_features.npy".format(prefix2))
    gallery_text_features = np.load("{}_text_features.npy".format(prefix2))

    test_image_ids = []
    test_fea = []
    gallery_image_ids = []
    gallery_fea = []

    test_filename = '{}_pairs_info.txt'.format(prefix1)
    gallery_filename = '{}_pairs_info.txt'.format(prefix2)

    with open(gallery_filename, 'r') as f:
        for i, line in enumerate(f):
            conts = line.strip().split("\t")
            image_id, text, _, _, instance_text = conts

            gallery_image_ids.append(image_id)
            gallery_fea.append(np.concatenate((gallery_image_features[i], gallery_text_features[i])))
            # gallery_fea.append(gallery_image_features[i])
            # gallery_fea.append(gallery_text_features[i])

    with open(test_filename) as f:
        for i, line in enumerate(f):
            conts = line.strip().split("\t")
            image_id, text, _, _, multi_instance_text = conts

            test_image_ids.append(image_id)
            test_fea.append(np.concatenate((test_image_features[i], test_text_features[i])))
            # test_fea.append(test_image_features[i])
            # test_fea.append(test_text_features[i])

    test_feas = np.array(test_fea)
    gallery_feas = np.array(gallery_fea)

    print(len(test_image_ids), test_feas.shape)
    print(len(gallery_image_ids), gallery_feas.shape)

    predict_dict = {}
    for i, test_id in enumerate(test_image_ids):
        test_fea = test_feas[i].reshape(1, -1)  # (1,1024)

        # cosine similarities
        # norm1 = norm(gallery_feas,axis=-1).reshape(gallery_feas.shape[0],1)
        # norm2 = norm(test_fea,axis=-1).reshape(1,test_fea.shape[0])
        # end_norm = np.dot(norm1,norm2)
        # similarities = (np.dot(gallery_feas, test_fea.T)/end_norm).squeeze(1)

        similarities = (gallery_feas @ test_fea.T).squeeze(1)
        predict_dict[test_id] = [similarities, np.array(gallery_image_ids)]

    print("predict_dict_len = ", len(predict_dict))

    retrieval_results = []
    for test_id in predict_dict.keys():
        similarities, gallery_ids = predict_dict[test_id]
        topN_indexs_per = np.argsort(-similarities)[:100]
        topN_results_per = np.array(gallery_ids)[topN_indexs_per]

        data = [test_id] + topN_results_per.tolist()
        retrieval_results.append(data)

    print("retrieval_results_len = ", len(retrieval_results))
    return retrieval_results


def main():
    args = parse_args()

    gallery_unit_id_label_txt = open("{}/product1m_gallery_ossurl_v2.txt".format(args.GT_dir)).readlines()
    test_query_suit_id_label_txt = open("{}/product1m_test_ossurl_v2.txt".format(args.GT_dir)).readlines()
    # dev_query_suit_id_label_txt=open("{}/product1m_dev_ossurl_v2.txt".format(args.GT_dir)).readlines()

    gallery_unit_id_label = {}
    for line in gallery_unit_id_label_txt:
        line = line.strip()
        line_split = line.split("#####")
        item_id = line_split[0]
        label_list = line_split[4].split("#;#")
        gallery_unit_id_label[item_id] = {
            "label": label_list
        }

    test_query_suit_id_label = {}
    for line in test_query_suit_id_label_txt:
        line = line.strip()
        line_split = line.split("#####")
        item_id = line_split[0]
        label_list = line_split[4].split("#;#")
        test_query_suit_id_label[item_id] = {
            "label": label_list
        }
    # for line in dev_query_suit_id_label_txt:
    #     line=line.strip()
    #     line_split=line.split("#####")
    #     item_id=line_split[0]
    #     label_list=line_split[4].split("#;#")
    #     test_query_suit_id_label[item_id]={
    #         "label":label_list
    #     }

    gallery_unit_label_id = {}
    for item_id, info in gallery_unit_id_label.items():
        label = info["label"][0]
        if label not in gallery_unit_label_id:
            gallery_unit_label_id[label] = [item_id]
        else:
            gallery_unit_label_id[label] += [item_id]

    results = {}

    retrieval_results = get_retrieval_id_list()

    topk_list = [1, 10, 50, 100]
    mAPs = []
    mARs = []
    Ps = []
    for topk in topk_list:
        topk_temp = topk
        mAP = 0
        mP = 0
        mAR = 0
        cnt = 0
        for index, each in enumerate(retrieval_results):
            query_id = each[0]
            rank_id_list = each[1:]
            pos_set = []

            cnt += 1
            query_suit_labels = test_query_suit_id_label[query_id]["label"]
            for label in query_suit_labels:
                pos_set += gallery_unit_label_id[label]

            topk = min(topk_temp, len(pos_set), len(rank_id_list))

            ap = compute_ap(rank_id_list, pos_set, topk)

            p = compute_p(rank_id_list, pos_set, topk)

            mAP += ap
            mP += p

            Ar = compute_ar(rank_list=rank_id_list,
                            GT_labels=query_suit_labels,
                            gallery_label_ids=gallery_unit_label_id,
                            gallery_id_label=gallery_unit_id_label,
                            N=topk_temp)
            mAR += Ar

        mAP /= cnt
        mP /= cnt
        mAR /= cnt

        mAPs.append(mAP)
        mARs.append(mAR)
        Ps.append(mP)

    print("top1:%.4f" % (mAPs[0]))
    print("mAP@10:%.4f, mAP@50:%.4f, mAP@100:%.4f" % (mAPs[1], mAPs[2], mAPs[3]))
    print("mAR@10:%.4f, mAR@50:%.4f, mAR@100:%.4f" % (mARs[1], mARs[2], mARs[3]))
    print("Prec@10:%.4f, Prec@50:%.4f, Prec@100:%.4f" % (Ps[1], Ps[2], Ps[3]))

    str_write = args.ckpt + " { top1:%.4f; mAP@10:%.4f, mAP@50:%.4f, mAP@100:%.4f; mAR@10:%.4f, mAR@50:%.4f, mAR@100:%.4f;prec@10:%.4f, prec@50:%.4f, prec@100:%.4f ; }\n" % (
        mAPs[0], mAPs[1], mAPs[2], mAPs[3], mARs[1], mARs[2], mARs[3], Ps[1], Ps[2], Ps[3])

    with open(args.output_path, "a") as f:
        f.write(str_write)

    return


if __name__ == '__main__':
    main()
