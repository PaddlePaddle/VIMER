import argparse
import datetime
import json
import numpy as np
import os
import sys

root_dir = os.path.abspath(__file__).split('tools')[0]
sys.path.insert(0, root_dir)

import os
import time
import random
from scipy import interpolate

import paddle
import paddle.nn as nn
from paddle.vision import transforms
from paddle.fluid.optimizer import LarsMomentum as LARS

from pathlib import Path

#assert paddle.__version__ == "2.2.2" # version check

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.crop import RandomResizedCrop
from util.engine_finetune import train_one_epoch, evaluate

from models.modeling_cae_modules import trunc_normal_
import models.models_linear_probing as models_vit_cae

from packages.dataset_folder import DatasetFolder
import packages.utils as utils


def setup_for_distributed(rank):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--clip_norm', default=None, type=float)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--sin_pos_emb', action='store_true')
    parser.set_defaults(sin_pos_emb=True)
    parser.add_argument('--disable_sin_pos_emb', action='store_false', dest='sin_pos_emb')
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--enable_linear_eval', action='store_true', default=False)
    parser.add_argument('--enable_multi_print', action='store_true', default=False,
                        help='allow each gpu prints something')

    parser.add_argument('--exp_name', default='', type=str,
                        help='name of exp. it is helpful when save the checkpoint')

    parser.add_argument('--save_freq', default=50, type=int,
                        help='freq of saving models')

    parser.add_argument('--linear_type', default='standard', type=str,
                        help='[standard, attentive, attentive_no_parameter]')
    parser.add_argument('--linear_depth', default=1, type=int,
                        help=' ')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = paddle.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)

    paddle.version.cudnn.FLAGS_cudnn_deterministic = True

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation='bicubic'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation='bicubic'),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = DatasetFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = DatasetFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_training_steps_per_epoch = len(
            dataset_train) // args.batch_size // num_tasks
        sampler_train = paddle.io.DistributedBatchSampler(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = paddle.io.DistributedBatchSampler(
                dataset_val, batch_size=args.batch_size,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = paddle.io.SequenceSampler(dataset_val)
    else:
        sampler_train = paddle.io.RandomSampler(dataset_train)
        sampler_val = paddle.io.SequenceSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.VisualdlLogger(log_dir=args.log_dir)
        log_writer = None
    else:
        log_writer = None

    data_loader_train = paddle.io.DataLoader(
        dataset_train, batch_sampler=sampler_train,
        num_workers=args.num_workers,
        places=device,
    )

    data_loader_val = paddle.io.DataLoader(
        dataset_val, batch_sampler=sampler_val,
        num_workers=args.num_workers,
        places=device,
    )

    model = models_vit_cae.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        lin_probe=args.enable_linear_eval,
        args=args,
    )

    if args.finetune and not args.eval:
        checkpoint = paddle.load(args.finetune)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        for key in list(checkpoint_model.keys()):
            if 'encoder.' in key:
                new_key = key.replace('encoder.', '')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)
            if 'teacher' in key or 'decoder' in key:
                checkpoint_model.pop(key)

        if args.rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())

        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key and args.rel_pos_bias:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q**(i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            paddle.to_tensor(f(dx, dy)).contiguous().view(-1, 1))

                    rel_pos_bias = paddle.concat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = paddle.concat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.set_state_dict(checkpoint_model)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = paddle.nn.Sequential(
        paddle.nn.BatchNorm1D(model.head.weight.shape[0], epsilon=1e-6, weight_attr=False, bias_attr=False), model.head)

    requires_grad = []

    # freeze all but the head
    for _, p in model.named_parameters():
        p.stop_gradient = True
        p.trainable = False 
    for nname, p in model.head.named_parameters():
        if '1.' in nname:
            p.stop_gradient = False
            p.trainable = True 
            requires_grad.append(nname)
        else:
            # paddle BN weight and bias is not None
            p.stop_gradient = True
            p.trainable = False 

    if args.linear_type != 'standard':
        for name, param in model.named_parameters():
            judge_requires_grad = False

            if 'linear_blocks' in name:
                judge_requires_grad = True
            if 'query_token' in name:
                judge_requires_grad = True

            if judge_requires_grad:
                requires_grad.append(name)

    print(f'require grad parameter: ', requires_grad)

    model_without_ddp = model
    n_parameters = sum(p.numel().item() for p in model.parameters()
                       if not p.stop_gradient)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers
    # define scaler for AMP training
    loss_scaler = NativeScaler()

    optimizer = LARS(parameter_list=model_without_ddp.head.parameters(),
                     momentum=0.9,
                     learning_rate=args.lr,
                     lars_weight_decay=args.weight_decay,
                     exclude_from_weight_decay=['0.','1.bias'],
                     multi_precision=True,
                     )
    print(optimizer)

    criterion = paddle.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)

        if misc.is_main_process():
            try:
                log_file = os.path.join('/root/paddlejob/',
                                        args.exp_name + '.txt')
                dst_file = os.path.join('/home/vis/bpfsrw5/cxk/cae/STDOUT_LINEAR', args.exp_name + '.txt')
                os.system("cp {} {}".format(log_file, dst_file))
                import shutil
                shutil.copy(log_file, dst_file)

                bpfs_dir = '/home/vis/bpfsrw5/cxk/cae/output/' + args.exp_name
                Path(bpfs_dir).mkdir(parents=True, exist_ok=True)
                dst_file = os.path.join(bpfs_dir, 'log.txt')
                shutil.copy(os.path.join(args.output_dir, "log.txt"), dst_file)
            except:
                print('failed: copy log file from local to BPFS !!!!')

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_freq == 0
                                or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
