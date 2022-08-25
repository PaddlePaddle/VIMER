# !/usr/bin/env python3
import sys
import numpy as np

print(sys.argv, len(sys.argv))

ckpt = ""
prefix = "ctc_1k"
feature_type = "image_only"
# feature_type = 'scene_text_only'

if len(sys.argv) == 2:
    prefix = sys.argv[1]
elif len(sys.argv) == 3:
    prefix = sys.argv[1]
    ckpt = sys.argv[2]
elif len(sys.argv) == 4:
    prefix = sys.argv[1]
    ckpt = sys.argv[2]
    feature_type = sys.argv[3]
else:
    print("[ERROR] illegal input params.")

if feature_type == "image_only":
    image_features = np.load("{}_image_features.npy".format(prefix))
elif feature_type == "scene_text_only" or feature_type == "fusion_only":
    image_features = np.load("{}_scene_text_features.npy".format(prefix))
elif feature_type == "image_fusion_average":
    image_features = np.load("{}_fusion_features.npy".format(prefix))
else:
    print("[ERROR] illegal input feature type.")

text_features = np.load("{}_text_features.npy".format(prefix))

ans_dict = {}
text_ans_dict = {}

image_ids_set = set()
sent_ids_set = set()
image_ids = []
image_fea = []
sent_ids = []
sent_fea = []
filename = "{}_pairs_info.txt".format(prefix)
with open(filename) as f:
    for i, line in enumerate(f):
        line = line.strip().split("\t")
        # if i >= 10000:
        #    break
        image_id, sent_id = line[0], line[1]
        if image_id not in image_ids_set:
            image_ids.append(image_id)
            image_fea.append(image_features[i])
            image_ids_set.add(image_id)
        if sent_id not in sent_ids_set:
            sent_ids.append(sent_id)
            sent_fea.append(text_features[i])
            sent_ids_set.add(sent_id)
        ans_dict[sent_id.strip(" ")] = image_id.strip(" ")
        text_ans_dict.setdefault(image_id.strip(" "), [])
        text_ans_dict[image_id.strip(" ")].append(sent_id.strip(" "))
image_feas = np.array(image_fea)
sent_feas = np.array(sent_fea)

print(len(image_ids), image_feas.shape)
print(len(sent_ids), sent_feas.shape)


datas = []
for i, image_id in enumerate(image_ids):
    image_fea = image_feas[i].reshape(1, -1)
    similarities = (sent_feas @ image_fea.T).squeeze(1)
    for j in range(len(sent_ids)):
        score = similarities[j]
        sent_id = sent_ids[j]
        datas.append([score, image_id, sent_id])


res_dict = {}
text_res_dict = {}
for line in datas:
    score, image_id, sent_id = float(line[0]), line[1], line[2]
    res_dict.setdefault(sent_id, [])
    res_dict[sent_id].append((score, image_id))
    text_res_dict.setdefault(image_id, [])
    text_res_dict[image_id].append((score, sent_id))

print("\n=============== Text2image: IMAGE RETRIEVAL ==================")
r1, r5, r10 = 0, 0, 0
cnt = 0
idx_all = 0.0
for sent_id in res_dict:
    res_list = res_dict[sent_id]
    res_list = sorted(res_list, reverse=True)
    ans = ans_dict[sent_id]
    image_id_sort = list(zip(*res_list))[1]
    ans_idx = image_id_sort.index(ans.strip())
    if ans_idx < 1:
        r1 += 1.0
    if ans_idx < 5:
        r5 += 1.0
    if ans_idx < 10:
        r10 += 1.0
    idx_all += ans_idx + 1
    cnt += 1
    # if cnt %  100 == 0:
    #    print(cnt, round(r1/cnt, 4), round(r5/cnt, 4), round(r10/cnt, 4), round(idx_all/cnt, 4))
print(
    "caption queries %d, avg recall:%.4f, r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f\n"
    % (cnt, (r1 + r5 + r10) / (cnt * 3), r1 / cnt, r5 / cnt, r10 / cnt, idx_all / cnt)
)

str_write = ckpt + "\tText2image %d r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f\n" % (
    cnt,
    r1 / cnt,
    r5 / cnt,
    r10 / cnt,
    idx_all / cnt,
)

print("=============== Image2text: TEXT RETRIEVAL ==================")
cnt = 0
r1, r5, r10 = 0, 0, 0
idx_all = 0.0
for image_id in text_res_dict:
    res_list = text_res_dict[image_id]
    res_list = sorted(res_list, reverse=True)
    ans = text_ans_dict[image_id]
    text_id_sort = list(zip(*res_list))[1]
    ans_idx_all = []
    for item in ans:
        ans_idx_all.append(text_id_sort.index(item.strip()))
    ans_idx = min(ans_idx_all)
    if ans_idx < 1:
        r1 += 1.0
    if ans_idx < 5:
        r5 += 1.0
    if ans_idx < 10:
        r10 += 1.0
    idx_all += ans_idx + 1
    cnt += 1
    # if cnt % 500 == 0:
    #    print (cnt, round(r1/cnt, 4), round(r5/cnt, 4), round(r10/cnt, 4), round(idx_all/cnt, 4))

print(
    "image queries %d, avg recall:%.4f, r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f\n"
    % (cnt, (r1 + r5 + r10) / (cnt * 3), r1 / cnt, r5 / cnt, r10 / cnt, idx_all / cnt)
)

str_write += ckpt + "\tImage2text %d r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f\n" % (
    cnt,
    r1 / cnt,
    r5 / cnt,
    r10 / cnt,
    idx_all / cnt,
)

with open("res.txt", "a") as f:
    f.write(str_write)
