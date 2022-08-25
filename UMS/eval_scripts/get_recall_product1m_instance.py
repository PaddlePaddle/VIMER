# !/usr/bin/env python3
import sys
import numpy as np
import pdb
from numpy.linalg import norm
import pickle

from sklearn.metrics.pairwise import cosine_similarity, paired_distances

print(sys.argv, len(sys.argv))

ckpt = ""
prefix1 = "product1m_test"
prefix2 = "product1m_gallery"

if len(sys.argv) == 2:
    ckpt = sys.argv[1]
elif len(sys.argv) == 4:
    ckpt = sys.argv[1]
    prefix1 = sys.argv[2]
    prefix2 = sys.argv[3]
elif len(sys.argv) > 1:
    print("[ERROR] illegal input params.")

test_image_features = np.load("{}_image_features.npy".format(prefix1))
test_text_features = np.load("{}_text_features.npy".format(prefix1))
gallery_image_features = np.load("{}_image_features.npy".format(prefix2))
gallery_text_features = np.load("{}_text_features.npy".format(prefix2))

test_image_ids = []
test_fea = []

gallery_image_ids = []
gallery_fea = []

test_filename = "{}_pairs_info.txt".format(prefix1)
gallery_filename = "{}_pairs_info.txt".format(prefix2)
# test_filename = '../../Product1M/product1m_test_ossurl_v2.txt'
# gallery_filename = '../../Product1M/product1m_gallery_ossurl_v2.txt'

# target_dict = {}

gallery_instances = {}
gallery_category_image_ids = {}
with open(gallery_filename, "r") as f:
    for i, line in enumerate(f):
        # print(line)
        conts = line.strip().split("\t")
        if len(conts) == 5:
            image_id, text, _, _, instance_text = conts
        else:
            print("[ERROR] illegal input instance text.")

        gallery_instances[image_id] = instance_text

        if instance_text not in gallery_category_image_ids:
            gallery_category_image_ids[instance_text] = []
        gallery_category_image_ids[instance_text].append(image_id)

        gallery_image_ids.append(image_id)
        gallery_fea.append(
            np.concatenate((gallery_image_features[i], gallery_text_features[i]))
        )
        # gallery_fea.append(gallery_image_features[i])
        # gallery_fea.append(gallery_text_features[i])

# pdb.set_trace()
print("gallery_instances_len = ", len(gallery_instances))

target_dict = {}
with open(test_filename) as f:
    for i, line in enumerate(f):
        conts = line.strip().split("\t")
        if len(conts) == 5:
            image_id, text, _, _, multi_instance_text = conts
        else:
            print("[ERROR] illegal input instance text.")

        target_instances_text = multi_instance_text.split("#;#")

        test_image_ids.append(image_id)
        test_fea.append(np.concatenate((test_image_features[i], test_text_features[i])))
        # test_fea.append(test_image_features[i])
        # test_fea.append(test_text_features[i])

        target_dict[image_id] = target_instances_text

test_feas = np.array(test_fea)
gallery_feas = np.array(gallery_fea)

print(len(test_image_ids), test_feas.shape)
print(len(gallery_image_ids), gallery_feas.shape)

predict_dict = {}
for i, test_id in enumerate(test_image_ids):
    test_fea = test_feas[i].reshape(1, -1)  # (1,1024)
    # pdb.set_trace()

    # cosine similarities
    # norm1 = norm(gallery_feas,axis=-1).reshape(gallery_feas.shape[0],1)
    # norm2 = norm(test_fea,axis=-1).reshape(1,test_fea.shape[0])
    # end_norm = np.dot(norm1,norm2)
    # similarities = (np.dot(gallery_feas, test_fea.T)/end_norm).squeeze(1)

    similarities = (gallery_feas @ test_fea.T).squeeze(1)  # (40031)
    predict_dict[test_id] = [similarities, np.array(gallery_image_ids)]

print("predict_dict_len = ", len(predict_dict))
print("target_dict_len = ", len(target_dict))

targets_flatten = {}
for test_id, target_instances in target_dict.items():
    targets_flatten[test_id] = []
    for gallery_id, instance_name in gallery_instances.items():
        if instance_name in target_instances:
            targets_flatten[test_id].append(gallery_id)

print("targets_flatten_len = ", len(targets_flatten))


def cal_prec(targets, predictions, gallery_instances, N):
    count = len(predictions)
    ans = []
    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:N]
        ans_idx = 0
        for topN_retrieved_gallery_id in gallery_ids[topN_indexs]:
            if gallery_instances[topN_retrieved_gallery_id] in targets[test_id]:
                ans_idx += 1

        ans.append(ans_idx / N)

        # pdb.set_trace()
    return np.mean(ans)


def cal_multi_prec(targets, predictions, gallery_instances):
    count = len(predictions)
    ans_10 = 0
    ans_50 = 0
    ans_100 = 0
    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:100]
        ans_10_idx = 0
        ans_50_idx = 0
        ans_100_idx = 0
        for i, topN_retrieved_gallery_id in enumerate(gallery_ids[topN_indexs]):
            if gallery_instances[topN_retrieved_gallery_id] in targets[test_id]:
                if i < 10:
                    ans_10_idx += 1
                if i < 50:
                    ans_50_idx += 1
                if i < 100:
                    ans_100_idx += 1

        ans_10 += ans_10_idx / 10
        ans_50 += ans_50_idx / 50
        ans_100 += ans_100_idx / 100

    return ans_10 / count, ans_50 / count, ans_100 / count


def cal_mAP(targets, predictions, gallery_instances, targets_flatten, N):
    count = len(predictions)
    ans = 0
    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:N]

        ans_rank = 0
        ans_idx = 0
        for i, topN_retrieved_gallery_id in enumerate(gallery_ids[topN_indexs]):
            if gallery_instances[topN_retrieved_gallery_id] in targets[test_id]:
                ans_rank += 1
                ans_idx += ans_rank / (i + 1)

        m_q = len(targets_flatten[test_id])
        # pdb.set_trace()
        ans = ans + ans_idx / min(m_q, N)

    ans = ans / count

    return ans


def cal_multi_mAP(targets, predictions, gallery_instances, targets_flatten):
    count = len(predictions)
    ans_10 = 0
    ans_50 = 0
    ans_100 = 0
    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:100]
        ans_rank = 0
        ans_10_idx = 0
        ans_50_idx = 0
        ans_100_idx = 0
        for i, topN_retrieved_gallery_id in enumerate(gallery_ids[topN_indexs]):
            if gallery_instances[topN_retrieved_gallery_id] in targets[test_id]:
                ans_rank += 1
                if i < 10:
                    ans_10_idx += ans_rank / (i + 1)
                if i < 50:
                    ans_50_idx += ans_rank / (i + 1)
                if i < 100:
                    ans_100_idx += ans_rank / (i + 1)

        m_q = len(targets_flatten[test_id])

        ans_10 += ans_10_idx / min(m_q, 10)
        ans_50 += ans_50_idx / min(m_q, 50)
        ans_100 += ans_100_idx / min(m_q, 100)

    return ans_10 / count, ans_50 / count, ans_100 / count


def cal_mAR(targets, predictions, gallery_instances, gallery_category_image_ids, N):
    count = len(predictions)
    ans = 0

    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:N]

        ans_idx = 0
        for target_category in set(targets[test_id]):

            r_q = 0
            for instance in targets[test_id]:
                if instance == target_category:
                    r_q += 1
            r_q = r_q / len(targets[test_id])

            RETR = 0
            for i, topN_retrieved_gallery_id in enumerate(gallery_ids[topN_indexs]):
                if gallery_instances[topN_retrieved_gallery_id] == target_category:
                    RETR += 1

            G_c = len(gallery_category_image_ids[target_category])
            ans_idx += min(1, RETR / min(round(r_q * N), G_c))

        C_q = len(set(targets[test_id]))
        ans += ans_idx / C_q
    ans = ans / count
    return ans


def cal_multi_mAR(targets, predictions, gallery_instancess, gallery_category_image_ids):
    count = len(predictions)
    ans_10 = 0
    ans_50 = 0
    ans_100 = 0

    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs = np.argsort(-similarities)[:100]

        ans_10_idx = 0
        ans_50_idx = 0
        ans_100_idx = 0
        for target_category in set(targets[test_id]):
            r_q = 0
            for instance in targets[test_id]:
                if instance == target_category:
                    r_q += 1
            r_q = r_q / len(targets[test_id])

            RETR_10 = 0
            RETR_50 = 0
            RETR_100 = 0
            for i, topN_retrieved_gallery_id in enumerate(gallery_ids[topN_indexs]):
                if gallery_instances[topN_retrieved_gallery_id] == target_category:
                    if i < 10:
                        RETR_10 += 1
                    if i < 50:
                        RETR_50 += 1
                    if i < 100:
                        RETR_100 += 1

            G_c = len(gallery_category_image_ids[target_category])
            ans_10_idx += min(1, RETR_10 / min(round(r_q * 10), G_c))
            ans_50_idx += min(1, RETR_50 / min(round(r_q * 50), G_c))
            ans_100_idx += min(1, RETR_100 / min(round(r_q * 100), G_c))

        C_q = len(set(targets[test_id]))
        ans_10 += ans_10_idx / C_q
        ans_50 += ans_50_idx / C_q
        ans_100 += ans_100_idx / C_q

    return ans_10 / count, ans_50 / count, ans_100 / count


def save_prediction(predictions):
    test_ids = []
    topN_results = []
    topN_similarities = []

    for test_id in predictions.keys():
        similarities, gallery_ids = predictions[test_id]
        topN_indexs_per = np.argsort(-similarities)[:50]
        topN_results_per = np.array(gallery_ids)[topN_indexs_per]
        topN_similarities_per = similarities[topN_indexs_per]

        test_ids.append(test_id)
        topN_results.append(topN_results_per)
        topN_similarities.append(topN_similarities_per)

    np.save("test_ids.npy", np.array(test_ids))
    np.save("topN_results.npy", np.array(topN_results))
    np.save("topN_similarities.npy", np.array(topN_similarities))
    # pdb.set_trace()


# mAP10 = cal_mAP(target_dict, predict_dict, gallery_instances, targets_flatten, 10)
# mAP50 = cal_mAP(target_dict, predict_dict, gallery_instances, targets_flatten, 50)
# mAP100 = cal_mAP(target_dict, predict_dict, gallery_instances, targets_flatten, 100)
# print("mAP@10:%.4f, mAP@50:%.4f, mAP@100:%.4f\n" % (mAP10, mAP50, mAP100))
# mAR10 = cal_mAR(target_dict, predict_dict, gallery_instances, gallery_category_image_ids, 10)
# mAR50 = cal_mAR(target_dict, predict_dict, gallery_instances, gallery_category_image_ids, 50)
# mAR100 = cal_mAR(target_dict, predict_dict, gallery_instances, gallery_category_image_ids, 100)
# print("mAR@10:%.4f, mAR@50:%.4f, mAR@100:%.4f\n" % (mAR10, mAR50, mAR100))
# prec10 = cal_prec(target_dict, predict_dict, gallery_instances, 10)
# prec50 = cal_prec(target_dict, predict_dict, gallery_instances, 50)
# prec100 = cal_prec(target_dict, predict_dict, gallery_instances, 100)
# print("prec@10:%.4f, prec@50:%.4f, prec@100:%.4f\n" % (prec10, prec50, prec100))

# save_prediction(predict_dict)
# print("multi")

print("Evaluation:")
mAP10, mAP50, mAP100 = cal_multi_mAP(
    target_dict, predict_dict, gallery_instances, targets_flatten
)
print("mAP@10:%.4f, mAP@50:%.4f, mAP@100:%.4f" % (mAP10, mAP50, mAP100))
mAR10, mAR50, mAR100 = cal_multi_mAR(
    target_dict, predict_dict, gallery_instances, gallery_category_image_ids
)
print("mAR@10:%.4f, mAR@50:%.4f, mAR@100:%.4f" % (mAR10, mAR50, mAR100))
prec10, prec50, prec100 = cal_multi_prec(target_dict, predict_dict, gallery_instances)
print("prec@10:%.4f, prec@50:%.4f, prec@100:%.4f" % (prec10, prec50, prec100))
top1 = cal_mAP(target_dict, predict_dict, gallery_instances, targets_flatten, 1)
print("top1:%.4f" % (top1))

str_write = (
    ckpt
    + " { mAP@10:%.4f, mAP@50:%.4f, mAP@100:%.4f; mAR@10:%.4f, mAR@50:%.4f, mAR@100:%.4f;prec@10:%.4f, prec@50:%.4f, prec@100:%.4f ;top:%.4f }\n"
    % (mAP10, mAP50, mAP100, mAR10, mAR50, mAR10, prec10, prec50, prec100, top1)
)

with open("eval_results_product1m.txt", "a") as f:
    f.write(str_write)
