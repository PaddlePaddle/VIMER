#!/usr/bin/env python
# Copyright (c) Baidu, Inc. and its affiliates.
"""
This training script is mainly constructed on train_net.py.
Additionally, this script is specialized for the training of supernet.
Moreover, this script adds a function of self-distillation.
If specifing `teacher_model_path` in the given config file, teacher model will
be built, otherwise teacher model is None.
"""
import logging
import os.path
import numpy as np
from functools import partial

import sys
sys.path.append('.')

from tools.monitor import Monitor
import paddle
paddle.seed(42)

from utils.events import CommonMetricSacredWriter
from engine.hooks import LRScheduler
from utils.config import auto_adjust_cfg
from fastreid.utils.checkpoint import Checkpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainerMoE,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
# from detectron2.engine.defaults import create_ddp_model
from evaluation import print_csv_format
from evaluation.evaluator import inference_on_dataset
from utils import comm

logger = logging.getLogger("ufo")


def do_test(cfg, model_moe, _run=None, subnet_mode="largest", moe_group=None):
    assert moe_group is not None, "moe_group cannot be None"
    if "evaluator" in cfg.dataloader:
        dataloaders = instantiate(cfg.dataloader.test)
        rets = {}
        for idx, (dataloader, evaluator_cfg) in enumerate(zip(dataloaders, cfg.dataloader.evaluator)):
            print(dataloader)
            evaluator_cfg.num_query = list(dataloader.task_loaders.values())[0].dataset.num_query
            evaluator_cfg.num_valid_samples = list(dataloader.task_loaders.values())[0].dataset.num_valid_samples
            evaluator_cfg.labels = list(dataloader.task_loaders.values())[0].dataset.labels
            evaluator = instantiate(evaluator_cfg)
            model_moe.subnet_mode = subnet_mode
            ret = inference_on_dataset(
                model_moe, dataloader, evaluator, moe_group=moe_group
            )
            print_csv_format(ret)

            # TODO: dirty, make it elegant
            task_name = '.'.join(list(dataloader.task_loaders.keys()))
            dataset_name = dataloader.task_loaders[task_name].dataset.dataset_name
            for metric, res in ret.items():
                rets['{}.{}.{}'.format(task_name, dataset_name, metric)] = res

                if _run is not None:
                    _run.log_scalar('{}.{}.{}'.format(task_name, dataset_name, metric), res, )
        return rets


def do_train(args, cfg, cfg_for_sacred=None, _run=None, monitor=None):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    logger = logging.getLogger("ufo")

    train_loader = instantiate(cfg.dataloader.train)
    auto_adjust_cfg(cfg, train_loader)
    logger.info(cfg)
    model = instantiate(cfg.model)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    cfg.optimizer.model = model
    optim = instantiate(cfg.optimizer)
    logger.info("Optim:\n{}".format(optim))
    
    print("====world_size: ", paddle.distributed.get_world_size())

    #为模型并行添加储存模型的逻辑
    rank = comm.get_rank()
    taskid = cfg.train.globalrank2taskid[str(rank)]
    taskspecific_output_dir = cfg.train.output_dir + '/task{}'.format(taskid)
    logger.info("the process of rank {} saves to {}".format(rank, taskspecific_output_dir))

    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainerMoE)(model, train_loader, optim, \
                                                monitor=monitor)

    checkpointer = Checkpointer(
        model,
        taskspecific_output_dir,
        optimizer=optim,
        trainer=trainer,
    )
    
    # # set optimizer
    # cfg.lr_multiplier.optimizer = optim

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            LRScheduler(optimizer=optim, scheduler=optim._learning_rate),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model, _run, subnet_mode="largest", moe_group=moe_group)),
            # hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model, _run, subnet_mode="smallest", moe_group=moe_group)),
            hooks.PeriodicWriter(
                default_writers(taskspecific_output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            ),
            hooks.PeriodicWriter(
                [CommonMetricSacredWriter(_run, cfg.train.max_iter)],
                period=cfg.train.log_period,
            )
            if comm.is_main_process() and _run is not None
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print('rank is {} , world_size is {}, gpu is {} '.format(args.rank, args.world_size, args.gpu))

    paddle.set_device('gpu')
    rank = paddle.distributed.get_rank()
    print('rank is {}, world size is {}'.format(rank, paddle.distributed.get_world_size()))
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    #构建 moe group
    world_size = paddle.distributed.get_world_size()
    dp_degree = cfg.train.dp_degree
    assert world_size % dp_degree == 0, "Error! be sure that world_size '%' dp_degree == 0"

    
    print("==================trainer info===================")
    m = Monitor(len(cfg.train.split_list), [int(val) for val in cfg.train.split_list.values()], dp_degree=cfg.train.dp_degree)
    print("==>sub_group: ", m.sub_group)
    print("==>moe_group: ", m.moe_group)
    print("==>dp_group: ", m.dp_group)
    print("=================================================")
    cur_moe_group = m.moe_group
    cur_dp_group = m.dp_group    

    #print("==================trainer info===================")
    #ranks = list(range(world_size))
    #group_len = len(ranks) // dp_degree
    #global_group = paddle.distributed.new_group(ranks)
    #cur_rank = paddle.distributed.get_rank()
    #print("==>cur_rank: ", cur_rank)
    #print("==>global_group: ", global_group)
    #dp_groups = [ranks[i::group_len] for i in range(group_len)]
    #moe_groups = np.split(np.array(ranks), dp_degree)
    #cur_dp_group = None
    #for dp in dp_groups:
    #    print("==>", dp)
    #    tmp = paddle.distributed.new_group(dp)
    #    if cur_rank in dp:
    #        cur_dp_group = tmp

    #cur_moe_group = None
    #for mp in moe_groups:
    #    print("==>", mp)
    #    tmp = paddle.distributed.new_group(mp.tolist())
    #    if cur_rank in mp.tolist():
    #        cur_moe_group = tmp 
    #print("==>cur_dp_group: ", cur_dp_group)
    #print("==>cur_moe_group: ", cur_moe_group)
    #print("======================================")

    #为模型并行添加载入模型的逻辑
    rank = comm.get_rank()
    taskid = cfg.train.globalrank2taskid[str(rank)]
    if cfg.train.init_checkpoint != '': #valid init_checkpoint
        parsing = cfg.train.init_checkpoint.split('/')
        parsing.insert(-1, 'task{}'.format(taskid))
        new_init_checkpoint = '/'.join(parsing)
        cfg.train.init_checkpoint = new_init_checkpoint
    logger.info("the process of rank {} loads model from {}".format(rank, cfg.train.init_checkpoint))
    
    if args.eval_only:
        train_loader = instantiate(cfg.dataloader.train)
        auto_adjust_cfg(cfg, train_loader)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        # model = create_ddp_model(model)    
        if paddle.distributed.get_world_size() > 1:
            model = paddle.DataParallel(model)
        Checkpointer(model).load(cfg.train.init_checkpoint)
        #load init model
        # state_dict= paddle.load(cfg.train.init_checkpoint)
        # if 'model' in state_dict: state_dict = state_dict['model']
        # load_state_dict = {}
        # for key, value in model.state_dict().items():
        #     if key not in state_dict:
        #         print('{} is not found in modelpth'.format(key))
        #     elif value.shape != state_dict[key].shape:
        #         print('the shape {} is unmatched: modelpath is {}, model is {}'.format(
        #             key, state_dict[key].shape, value.shape, ))
        #     else:
        #         load_state_dict[key] = state_dict[key]
        # model.set_state_dict(load_state_dict)
        print(do_test(cfg, model, moe_group=cur_moe_group))
    else:
        if cfg.train.sacred.enabled and comm.is_main_process():
            from sacred import Experiment
            from sacred.observers import MongoObserver
            from sacred.observers import FileStorageObserver
            ex = Experiment(cfg.train.output_dir)

            mongo_url = None
            # do not add mongo.txt into git repo
            if os.path.exists('mongo.txt'):
                with open('mongo.txt', 'r') as fin:
                    mongo_url = fin.readline().strip()
            else:
                print('mongo.txt does not exists, use file observer instead')

            if mongo_url is not None:
                ex.observers.append(MongoObserver(mongo_url))
            else:
                ex.observers.append(FileStorageObserver(cfg.train.output_dir))

            @ex.config
            def train_cfg():
                args = None
                cfg_for_sacred = None

            # sacred will convert `cfg` to a new datatype: `ConfigScope`
            # we want to keep the original datatype and therefore only keep a dict record, `cfg_`
            def do_train_sacred(args, cfg_for_sacred, _run):
                do_train(args, cfg, cfg_for_sacred, _run, monitor=m)

            ex.main(do_train_sacred)
            ex.run(config_updates={'args': args, 'cfg_for_sacred': LazyConfig.to_dict(cfg)})
        else:
            do_train(args, cfg, monitor=m)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    main(args)
