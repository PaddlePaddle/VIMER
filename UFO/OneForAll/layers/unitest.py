"""unitest.py for cossoftmax
"""
import numpy as np
import random
import paddle
import torch

import any_softmax
import any_softmax_torch


if __name__ == "__main__":
    nb_classes = 2300
    batch_size = 32
    targets_array = np.array([random.choice(range(nb_classes)) for _ in range(batch_size)], dtype='int64')
    logits_array = np.random.rand(batch_size, nb_classes)

    cossoftmax_pd = any_softmax.CosSoftmax(nb_classes, 30, 0.1)
    targets_array_pd = paddle.Tensor(targets_array)
    logits_pd = paddle.Tensor(logits_array)
    feature_pd = cossoftmax_pd(logits_pd.clone(), targets_array_pd)

    cossoftmax_torch = any_softmax_torch.CosSoftmax(nb_classes, 30, 0.1)
    targets_torch = torch.tensor(targets_array, dtype=torch.int64) #.cuda()
    logits_torch = torch.tensor(logits_array) #.cuda()
    feature_torch  = cossoftmax_torch(logits_torch.clone(), targets_torch)
    
    print("the diff between paddle results {} and torch {} is {}".format(
        feature_pd, feature_torch, np.linalg.norm(feature_pd.cpu().numpy() - feature_torch.numpy())
        ))

#inplace 操作
import torch
q = torch.rand(9)
q.requires_grad=True
e = q + 0
loss = sum(e.mul_(30))
loss.backward()
print(q.grad)