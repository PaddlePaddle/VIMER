import math
import sys
from typing import Iterable, Optional

import paddle
import paddle.nn as nn
import paddle.amp as amp


import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: nn.Layer,
                    criterion: nn.Layer,
                    data_loader: Iterable,
                    optimizer,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 

    accum_iter = args.accum_iter

    optimizer.clear_gradients()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        # forward
        with amp.auto_cast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss,
                                optimizer,
                                parameters=model.parameters(),
                                create_graph=False,
                                update_grad=(data_iter_step + 1) %
                                accum_iter == 0)
        optimizer.clear_gradients()

        paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        #min_lr = 10.
        #max_lr = 0.
        #for group in optimizer.param_groups:
        #    min_lr = min(min_lr, group["lr"])
        #    max_lr = max(max_lr, group["lr"])
        lr_ = optimizer._learning_rate
        metric_logger.update(lr=lr_)
        metric_logger.update(grad_norm=grad_norm)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in visualdl.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr_, epoch_1000x)
            log_writer.update(grad_norm=grad_norm, head="opt")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model):
    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # compute output
        with amp.auto_cast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
