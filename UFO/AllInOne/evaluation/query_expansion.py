"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on
# https://github.com/PyRetri/PyRetri/blob/master/pyretri/index/re_ranker/re_ranker_impl/query_expansion.py

import numpy as np
# import torch
# import torch.nn.functional as F
import paddle
import paddle.nn.functional as F

def aqe(query_feat, gallery_feat, qe_times=1, qe_k=10, alpha=3.0):
    """
    Combining the retrieved topk nearest neighbors with the original query and doing another retrieval.
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    Args :
        query_feat (paddle.tensor):
        gallery_feat (paddle.tensor):
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
        alpha (float):
    """
    num_query = query_feat.shape[0]
    all_feat = paddle.concat((query_feat, gallery_feat), axis=0)
    norm_feat = F.normalize(all_feat, p=2, axis=1)

    all_feat = all_feat.numpy()
    for i in range(qe_times):
        all_feat_list = []
        sims = paddle.mm(norm_feat, norm_feat.t())
        sims = sims.data.cpu().numpy()
        for sim in sims:
            init_rank = np.argpartition(-sim, range(1, qe_k + 1))
            weights = sim[init_rank[:qe_k]].reshape((-1, 1))
            weights = np.power(weights, alpha)
            all_feat_list.append(np.mean(all_feat[init_rank[:qe_k], :] * weights, axis=0))
        all_feat = np.stack(all_feat_list, axis=0)
        norm_feat = F.normalize(paddle.to_tensor(all_feat), p=2, axis=1)

    query_feat = paddle.to_tensor(all_feat[:num_query])
    gallery_feat = paddle.to_tensor(all_feat[num_query:])
    return query_feat, gallery_feat
