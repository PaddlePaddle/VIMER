import builtins
import datetime
import os
import time
from collections import defaultdict, deque

import paddle
import paddle.amp as amp
import paddle.distributed as dist


__all__ = ['MetricLogger', 'get_world_size']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype=paddle.float64)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        try:
            for meter in self.meters.values():
                meter.synchronize_between_processes()
        except:
            print("check is_dist_avail_and_initialized()")
            pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
            'time: {time}', 'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_msg.format(i,
                                   len(iterable),
                                   eta=eta_string,
                                   meters=str(self),
                                   time=str(iter_time),
                                   data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def setup_for_distributed_each_gpu(rank):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        builtin_print('rank is: ', rank, end=' ')
        builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    # TODO
    # if not dist.is_available():
    #     return False
    # if not dist.is_initialized():
    #     return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def init_distributed_mode(args):
    dist.init_parallel_env()
    args.distributed = True
    if not args.enable_multi_print:
        setup_for_distributed(is_main_process())
    else:
        setup_for_distributed_each_gpu(args.rank)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = amp.GradScaler(
            init_loss_scaling=2.**16,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=1,
        )  # same as pytorch
    
    def __call__(self, loss, optimizer, parameters=None, clip_grad=None, create_graph=False, update_grad=True, use_amp=True):
        if use_amp:
            self._scaler.scale(loss).backward(retain_graph=create_graph)  # do backward
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = self.clip_grad_norm(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                if optimizer.type == 'lars_momentum':
                    self._scaler.minimize(optimizer,
                                        loss)  # minimize to support LARS opt
                else:
                    self._scaler.step(optimizer)  # minimize to support LARS opt
                    self._scaler.update()  # minimize to support LARS opt
            else:
                norm = None
            return norm
        else:
            loss.backward(retain_graph=create_graph)  # do backward
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = self.clip_grad_norm(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                if optimizer.type == 'lars_momentum':
                    optimizer.minimize(loss)
                else:
                    optimizer.step()
            else:
                norm = None
            return norm
    
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def clip_grad_norm(self, parameters, max_norm, norm_type=2.0):
        if isinstance(parameters, paddle.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return paddle.to_tensor(0.)
        if norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max() for p in parameters)
        else:
            total_norm = paddle.norm(
                paddle.stack([
                    paddle.norm(p.grad.detach(), norm_type) for p in parameters
                ]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().scale_(clip_coef)
                p._reset_grad_inplace_version(True)
        return total_norm


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor(0.)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = paddle.norm(
            paddle.stack(
                [paddle.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type)
    return total_norm


def save_model(args,
               epoch,
               model_without_ddp,
               optimizer,
               loss_scaler,
               tag=None,
               exp_name=None):
    to_save_state_dict = model_without_ddp.state_dict()
    for key in list(to_save_state_dict.keys()):
        if key.startswith('teacher_network.'):
            to_save_state_dict.pop(key)

    to_save = {
        'model': to_save_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = f'checkpoint-{tag or epoch}.pd'
    if exp_name is not None:
        checkpoint_name = f"{exp_name}_" + checkpoint_name
    save_on_master(to_save, os.path.join(args.output_dir, checkpoint_name))


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            raise NotImplementedError
        else:
            checkpoint = paddle.load(args.resume)

        checkpoint_model = checkpoint['model']
        model_without_ddp.set_state_dict(checkpoint_model)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (
                hasattr(args, 'eval') and args.eval):
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))
    return [
        correct[:k].reshape([-1]).astype(float).sum(0) * 100. / batch_size
        for k in topk
    ]
