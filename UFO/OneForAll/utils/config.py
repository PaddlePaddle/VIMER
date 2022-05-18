"""utils/config.py
"""
import logging


def auto_adjust_cfg(cfg, train_loader):
    """auto_adjust_cfg
    """
    logger = logging.getLogger("ufo")

    # single task
    if "num_classes" in cfg.model.heads and cfg.model.heads.num_classes == 0:
        # auto scale num classes
        num_classes = train_loader.dataset.num_classes
        cfg.model.heads.num_classes = num_classes
        logger.info('Autoscale number of classes: {}'.format(num_classes))

    # multitask
    if "task_loaders" in cfg.dataloader.train:
        for task_name, task_loader in train_loader.task_loaders.items():
            num_classes = task_loader.dataset.num_classes
            cfg.model.heads[task_name].num_classes = num_classes
            logger.info('Autoscale {} number of classes: {}'.format(task_name, num_classes))

    return cfg
