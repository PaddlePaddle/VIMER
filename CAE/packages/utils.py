import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np

from pathlib import Path

import paddle
import paddle.distributed as dist

from models.modeling_discrete_vae import Dalle_VAE

from visualdl import LogWriter
from util.misc import save_on_master


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
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class VisualdlLogger(object):
    def __init__(self, log_dir):
        self.writer = LogWriter(logdir=log_dir) 
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    paddle.save(checkpoint, mem_file)
    mem_file.seek(0)
    # TODO
    model_ema._load_checkpoint(mem_file)


def load_state_dict(model,
                    state_dict,
                    prefix='',
                    ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".
            format(model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, model_ema=None, exp_name=None):
    output_dir = args.output_dir
    epoch_name = str(epoch)
    if loss_scaler is not None:
        if exp_name is not None:
            checkpoint_paths = [output_dir + '/' + '{}_checkpoint-{}.pd'.format(exp_name, epoch_name)]
        else:
            checkpoint_paths = [output_dir + '/' +  'checkpoint-%s.pd' % epoch_name]
        for checkpoint_path in checkpoint_paths:
            to_save_state_dict = model_without_ddp.state_dict()
            # all_keys = list(state_dict.keys())
                
            for key in list(to_save_state_dict.keys()):
                if key.startswith('teacher_network.'):
                    to_save_state_dict.pop(key)

            to_save = {
                'model': to_save_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        if exp_name is not None:
            model.save_checkpoint(save_dir=args.output_dir, tag="{}_checkpoint-{}".format(exp_name, epoch_name), client_state=client_state)
        else:
            model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def auto_load_model(args,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    model_ema=None):
    output_dir = Path(args.output_dir)

    # amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(
            os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir,
                                       'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        checkpoint = paddle.load(args.resume)

        # handle ema model
        need_state_dict = model_without_ddp.state_dict()
        need_ema = False
        for key in need_state_dict.keys():
            if 'teacher_network' in key:
                need_ema = True
                break

        checkpoint_model = checkpoint['model']

        if need_ema:
            all_keys = list(checkpoint_model.keys())
            all_keys = [key for key in all_keys if key.startswith('encoder.')]
            for key in all_keys:
                new_key = key.replace('encoder.', 'teacher_network.')
                checkpoint_model[new_key] = checkpoint_model[key]

        model_without_ddp.set_state_dict(checkpoint_model)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if hasattr(args, 'model_ema') and args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def create_d_vae(weight_path, d_vae_type, image_size):
    if d_vae_type == "dall-e":
        return get_dalle_vae(weight_path, image_size)
    elif d_vae_type == "customized":
        return get_d_vae(weight_path, image_size)
    elif d_vae_type == "to_tensor":
        return None
    else:
        raise NotImplementedError()


def get_dalle_vae(weight_path, image_size):
    vae = Dalle_VAE(image_size)
    vae.load_model(model_dir=weight_path)
    return vae


def get_d_vae(weight_path, image_size):
    NUM_TOKENS = 8192
    NUM_LAYERS = 3
    EMB_DIM = 512
    HID_DIM = 256

    state_dict = paddle.load(os.path.join(weight_path,
                                          "pytorch_model.bin"))["weights"]
    from models.modeling_discrete_vae import DiscreteVAE  # TODO Not Impl

    model = DiscreteVAE(
        image_size=image_size,
        num_layers=NUM_LAYERS,
        num_tokens=NUM_TOKENS,
        codebook_dim=EMB_DIM,
        hidden_dim=HID_DIM,
    )

    model.set_state_dict(state_dict)
    return model
