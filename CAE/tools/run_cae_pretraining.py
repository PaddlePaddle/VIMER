import argparse
import datetime
import numpy as np
import os
import sys

root_dir = os.path.abspath(__file__).split('tools')[0]
sys.path.insert(0, root_dir)

import time
import json
import os
import random
import shutil

import paddle
import paddle.optimizer as optim
import paddle.nn as nn

from pathlib import Path

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from packages.datasets import build_cae_pretraining_dataset
from packages.engine_for_pretraining import train_one_epoch
import packages.utils as utils
from models import modeling_cae_pretrain


def get_args():
    parser = argparse.ArgumentParser('pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--save_ckpt_toBPFS_freq', default=50, type=int)
    parser.add_argument("--discrete_vae_weight_path", type=str)
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")
    parser.add_argument('--amp', action='store_true', default=False, help='if or not use amp')
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--sincos_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.98, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--enable_multi_print', action='store_true', default=False, help='allow each gpu prints something')
    parser.add_argument('--regressor_depth', default=4, type=int,
                        help='depth of self-attention block for decoder')
    parser.add_argument('--num_decoder_self_attention', default=0, type=int, help='number of self-attention in decoder')

    parser.add_argument('--decoder_embed_dim', default=768, type=int,
                        help='dimensionaltiy of embeddings for decoder')
    parser.add_argument('--decoder_num_heads', default=12, type=int,
                        help='Number of heads for decoder')
    parser.add_argument('--decoder_num_classes', default=8192, type=int,
                        help='Number of classes for decoder')
    parser.add_argument('--decoder_layer_scale_init_value', default=0.1, type=float,
                        help='decoder layer scale init value')

    parser.add_argument('--mask_generator', default='block', type=str,
                        help='choice = [block, random]')
    parser.add_argument('--ratio_mask_patches', default=None, type=float,
                        help="mask ratio. only use when 'mask_generator' is random")

    # color jitter, default is False
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT',
                        help='Color jitter factor (default: 0)')
    parser.add_argument('--exp_name', default='', type=str,
                        help='name of exp. it is helpful when save the checkpoint')

    parser.add_argument('--target_mode', default='clusterID', type=str,
                        help='target, [clusterID, rgb, random]')
    parser.add_argument('--target_path', default='/home/vis/bpfsrw5/cxk/dalle-weights/random_vector_768x8192.pth', type=str,
                        help='path to load target vectors')

    parser.add_argument('--normalized_pixel', default='layernorm', type=str,
                        help='how to generate the regression target, [layernorm, none, channel, patch]')
    parser.add_argument('--denorm', action='store_true', default=False, help='if true, the RGB target will be denorm first')

    parser.add_argument('--rescale_init', action='store_true', default=False,
                        help='if true, the fix_init_weight() func will be activated')
    # dual path CAE
    parser.add_argument('--dual_loss_weight', type=float, default=1, help='loss weight for the dual path loss')
    parser.add_argument('--dual_loss_type', type=str, default='mse', help='[mse, kld]')
    parser.add_argument('--dual_path_ema', type=float, default=0, help='ema weight for the dual path network')

    # crop size
    parser.add_argument('--crop_min_size', type=float, default=0.08, help='min size of crop')
    parser.add_argument('--crop_max_size', type=float, default=1.0, help='max size of crop')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = modeling_cae_pretrain.__dict__[args.model](
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        args=args,
    )

    return model


def main(args):
    misc.init_distributed_mode(args)

    print("{}".format(args).replace(', ', ',\n'))

    device = paddle.set_device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
    #random.seed(seed)

    #paddle.version.cudnn.FLAGS_cudnn_deterministic = True

    if args.target_mode == 'rgb':
        assert args.discrete_vae_type == "to_tensor"

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    if args.debug:
        args.data_path = args.data_path.replace('train', 'val')
    dataset_train = build_cae_pretraining_dataset(args)

    # prepare discrete vae
    d_vae = utils.create_d_vae(
        weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
        image_size=args.second_input_size)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        sampler_train = paddle.io.DistributedBatchSampler(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        sampler_train = paddle.io.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.VisualdlLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = paddle.io.DataLoader(
        dataset_train, batch_sampler=sampler_train,
        num_workers=args.num_workers,
        places=device,
    )

    model_without_ddp = model
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * misc.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # following timm: set wd as 0 for bias and norm layers
    parameters = [
        param for param in model_without_ddp.parameters()
        if 'teacher' not in param.name
    ]
    skip_list = getattr(model_without_ddp, 'no_weight_decay', [])
    decay_dict = {
        param.name: not (len(param.shape) == 1 or name.endswith(".bias")
                         or name in skip_list)
        for name, param in model_without_ddp.named_parameters()
        if not 'teacher' in name
    }
    if args.opt.lower() == 'adamw':
        optimizer = optim.AdamW(
            learning_rate=args.lr,
            beta1=0.9,
            beta2=0.999,
            parameters=parameters,
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda n: decay_dict[n],
        )
    else:
        raise NotImplementedError

    loss_scaler = NativeScaler()

    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    assert args.weight_decay_end is None

    utils.auto_load_model(
        args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.batch_sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        # copy the log file to BPFS
        if misc.is_main_process():
            try:
                log_file = os.path.join('/root/paddlejob/', args.exp_name + '.txt')
                dst_file = os.path.join('/home/vis/bpfsrw5/cxk/cae/STDOUT3', args.exp_name + '.txt')
                os.system("cp {} {}".format(log_file, dst_file))
                shutil.copy(log_file, dst_file)

                bpfs_dir = '/home/vis/bpfsrw5/cxk/cae/output/' + args.exp_name
                Path(bpfs_dir).mkdir(parents=True, exist_ok=True)
                dst_file = os.path.join(bpfs_dir, 'log.txt')
                shutil.copy(os.path.join(args.output_dir, "log.txt"), dst_file)
            except:
                print('failed: copy log file from local to BPFS !!!!')

        train_stats = train_one_epoch(
            model, d_vae, data_loader_train,
            optimizer, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            args=args,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, exp_name=args.exp_name)

            if (epoch + 1) % args.save_ckpt_toBPFS_freq == 0 or epoch + 1 == args.epochs:
                # copy the checkpoint to BPFS
                try:
                    if misc.is_main_process():
                        output_dir = Path(args.output_dir)
                        epoch_name = str(epoch)
                        checkpoint_paths = [output_dir / ('{}_checkpoint-{}.pth'.format(args.exp_name, epoch_name))]

                        bpfs_dir = '/home/vis/bpfsrw5/cxk/cae/output/' + args.exp_name
                        Path(bpfs_dir).mkdir(parents=True, exist_ok=True)
                        dst_file = os.path.join(bpfs_dir, '{}_checkpoint-{}.pth'.format(args.exp_name, epoch_name))

                        os.system("cp {} {}".format(checkpoint_paths[0], dst_file))
                        shutil.copy(checkpoint_paths[0], dst_file)
                except:
                    print('failed: copy checkpoint from local to BPFS !!!!')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
