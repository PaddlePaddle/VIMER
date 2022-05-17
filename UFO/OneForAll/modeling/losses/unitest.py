"""unitest.py for cross_entropy_loss
"""
import cross_entroy_loss
import cross_entropy_loss_torch

import numpy as np 
import random
import paddle
import torch

if __name__ == '__main__':
    nb_classes = 2300
    batch_size = 32
    gt_classes_array = np.array([random.choice(range(nb_classes)) for _ in range(batch_size)], dtype='int64')
    pred_class_outputs_array = np.random.rand(batch_size, nb_classes)


    pred_class_outputs_pd = paddle.Tensor(pred_class_outputs_array)
    gt_classes_pd = paddle.Tensor(gt_classes_array)
    loss_pd = cross_entroy_loss.cross_entropy_loss(pred_class_outputs_pd, gt_classes_pd, eps=0.1, alpha=0.2)

    pred_class_outputs_torch = torch.tensor(pred_class_outputs_array) #.cuda()
    gt_classes_torch = torch.tensor(gt_classes_array, dtype=torch.int64) #.cuda()
    loss_torch = cross_entropy_loss_torch.cross_entropy_loss(
        pred_class_outputs_torch, 
        gt_classes_torch, 
        eps=0.1, alpha=0.2
        )
    
    print("the diff between paddle results {} and torch {} is {}".format(
        loss_pd, loss_torch, loss_pd.cpu().numpy() - loss_torch.numpy()
        ))

    
    
    
    def soft_margin_loss(x, y):
        """
        Args:
            x: shape [N]
            y: shape [N]
        """
        return paddle.sum(paddle.log(1 + paddle.exp(-1 * x * y))) /  x.numel()
    n_length = 32 
    x, y  = np.random.rand(n_length), np.random.rand(n_length)
    x_pd = paddle.Tensor(x)
    y_pd = paddle.Tensor(y)
    x_torch = torch.Tensor(x)
    y_torch = torch.Tensor(y)
    loss_pd = soft_margin_loss(x_pd, y_pd)
    loss_torch = torch.nn.functional.soft_margin_loss(x_torch, y_torch)
    print("the diff between paddle results {} and torch {} is {}".format(
        loss_pd, loss_torch, loss_pd.cpu().numpy() - loss_torch.numpy()
        ))



