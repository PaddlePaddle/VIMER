"""configs/MS1MV3_PersonAll_VeriAll_SOP_4cards/vitbasesuper_hardpolicy_moeplusbalance_8cards_0.25bs_align3.py
"""
from ..common import optimizer, train
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from data.build import build_train_set, build_reid_train_loader_lazy
from data.build import build_test_set, build_reid_test_loader_lazy
from data.build import build_face_test_set, build_face_test_loader_lazy
from data.transforms.build import build_transforms_lazy
from evaluation.reid_evaluation import ReidEvaluatorSingleTask
from evaluation.face_evaluation import FaceEvaluatorSingleTask
from evaluation.retri_evaluator import RetriEvaluatorSingleTask
from modeling.backbones.vision_transformer_super_hardpolicy_moe import build_vit_backbone_lazy
from modeling.heads.embedding_head import EmbeddingHead
from modeling.meta_arch.multitask_super_hardpolicy_moe import MultiTaskBatchFuse

#模型并行中taskid —> globalrank号的映射关系
#globalrank2taskid是dict，key的类型为字符串，代表rank号；value类型为int，代表taskid
#'0':0, 表示rank=0对应的进程（卡）对应于task0
train.dp_degree=1
train.globalrank2taskid = {'0':0, '1':100, '2':101, '3':102, '4':1, '5':2, '6':103, '7':3}
#task_names = ['task1', 'task2', 'task3', 'task4']
train.split_list = {'task1':4, 'task2':1, 'task3':2, 'task4':1}

dataloader = OmegaConf.create()

dataloader.train = L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',
             sample_prob=[]),
    task_loaders=L(OrderedDict)(
        task1=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names = ("MS1MV3",),
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[256, 256],
                    do_rea=False, #True,
                    rea_prob=0.5,
                    do_flip=False, #True,
                    do_pad=False, #True,
                    do_autoaug=False, #True,
                    autoaug_prob=0.5,
                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                )
            ),
            total_batch_size=2048,
            num_workers=8,
        ),
        task2=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names = ("PersonAll",),
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[256, 256],
                    do_rea=False, #True,
                    rea_prob=0.5,
                    do_flip=False, #True,
                    do_pad=False, #True,
                    do_autoaug=False, #True,
                    autoaug_prob=0.5,
                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                )
            ),
            total_batch_size=512,
            num_workers=8,
        ),
        task3=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names = ("VeriAll",),
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[256, 256],
                    do_rea=False, #True,
                    rea_prob=0.5,
                    do_flip=False, #True,
                    do_pad=False, #True,
                    do_autoaug=False, #True,
                    autoaug_prob=0.5,
                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                )
            ),
            total_batch_size=1024,
            num_workers=8,
        ),
        task4=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names = ("SOP",),
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[256, 256],
                    do_rea= False, #True,
                    rea_prob=0.5,
                    do_flip= False,  #True,
                    do_pad=False, #True,
                    do_autoaug=False,  #True,
                    autoaug_prob=0.5,
                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                )
            ),
            total_batch_size=512,
            num_workers=8,
        ),
    ),
)

dataloader.test = [
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task1=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name = "CALFW",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task1=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name = "CPLFW",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task2=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "Market1501",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task2=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "DukeMTMC",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task2=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "MSMT17",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task3=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "VeRi",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task3=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "LargeVehicleID",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task3=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "LargeVeRiWild",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task4=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name = "SOP",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=512,
                num_workers=8,
            ),
        )
    ),
]

dataloader.evaluator = [
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(ReidEvaluatorSingleTask)(
        cfg=dict(
            AQE=dict(enabled=False, alpha=3.0, QE_time=1, QE_k=5),
            metric='cosine',
            rerank=dict(enabled=False, k1=20, k2=6, lambda_=0.3),
            ROC=dict(enabled=False),
        ),
    ),
    L(RetriEvaluatorSingleTask)(
        cfg=dict(recalls=[1, 10, 100, 1000]),
    ),
]

weight_decay_norm=0.0001
bias_lr_factor=2.0
model = L(MultiTaskBatchFuse)(
    backbone=L(build_vit_backbone_lazy)(
        pretrain=True,
        pretrain_path='pretrained/vit_base_p16_224.pdmodel',
        input_size=[256, 256],
        depth='base',
        sie_xishu=3.0,
        stride_size=16,
        drop_ratio=0.0,
        drop_path_ratio=0.0, #0.2,
        attn_drop_rate=0.0,
        use_checkpointing=False, #True,
        share_last=True,
        n_tasks=4, 
        task_names=['task1', 'task2', 'task3', 'task4'],
    ),
    heads=L(OrderedDict)(
        task1=L(EmbeddingHead)(
            feat_dim=768,
            norm_type='SyncBN',
            with_bnneck=False, #True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=50,
            margin=0.3,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/vit_base_p16_224.pdmodel',
            depth='base',
        ),
        task2=L(EmbeddingHead)(
            feat_dim=768,
            norm_type='SyncBN',
            with_bnneck=False, #True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/vit_base_p16_224.pdmodel',
            depth='base',
        ),
        task3=L(EmbeddingHead)(
            feat_dim=768,
            norm_type='SyncBN',
            with_bnneck=False, #True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=20,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/vit_base_p16_224.pdmodel',
            depth='base',
        ),
        task4=L(EmbeddingHead)(
            feat_dim=768,
            norm_type='SyncBN',
            with_bnneck=False, #True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/vit_base_p16_224.pdmodel',
            depth='base',
        ),
    ),
    pixel_mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    pixel_std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    n_tasks=4,
    choices={
            'num_heads': [12], 'mlp_ratio': [4.0], \
                'embed_dim': [768], 'depth': [12], 'img_size': [224]
                },
    task_loss_kwargs=
    dict(
        task1={
            'loss_names': ('CrossEntropyLoss', ),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
        },
        task2={
            'loss_names': ('CrossEntropyLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            #'tri': {
            #    'margin': 0.0,
            #    'norm_feat': False,
            #    'hard_mining': True,
            #    'scale': 1.0,
            #},
        },
        task3={
            'loss_names': ('CrossEntropyLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            #'tri': {
            #    'margin': 0.0,
            #    'norm_feat': False,
            #    'hard_mining': True,
            #    'scale': 1.0,
            #},
        },
        task4={
            'loss_names': ('CrossEntropyLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            #'tri': {
            #    'margin': 0.0,
            #    'norm_feat': False,
            #    'hard_mining': True,
            #    'scale': 1.0,
            #},
        },
    ),
)

# data settings
dataloader.train.task_loaders.task1.total_batch_size = dataloader.train.task_loaders.task1.total_batch_size // 8
dataloader.train.task_loaders.task2.total_batch_size = dataloader.train.task_loaders.task2.total_batch_size // 8
dataloader.train.task_loaders.task3.total_batch_size = dataloader.train.task_loaders.task3.total_batch_size // 8
dataloader.train.task_loaders.task4.total_batch_size = dataloader.train.task_loaders.task4.total_batch_size // 8

# model settings 

# loss settings
# model.task_loss_kwargs.task2.ce.prob = 0.9
# model.task_loss_kwargs.task4.ce.prob = 0.8
model.task_loss_kwargs.task2.ce.scale = 0.5
model.task_loss_kwargs.task4.ce.scale = 0.2
# model.heads.task4.dropout = True

# learning rate settings
optimizer.base_lr = 1.25e-1
optimizer.lr_multiplier.base_lr = optimizer.base_lr
optimizer.lr_multiplier.eta_min = optimizer.base_lr * 2e-2


# iteration settings
train.max_iter = 50000
optimizer.lr_multiplier.max_epoch = 120
optimizer.lr_multiplier.warmup_iters = int(train.max_iter * 0.1)
optimizer.lr_multiplier.max_iters = train.max_iter - optimizer.lr_multiplier.warmup_iters
optimizer.lr_multiplier.iters_per_epoch = train.max_iter // optimizer.lr_multiplier.max_epoch
optimizer.lr_multiplier.warmup_factor = 2e-3

# amp settings
train.amp.enabled = False

# optimizer settings
optimizer.momentum = 0.0

train.output_dir = 'ufo_logs/ms1mv3_personall_veriall_sop/vitbasesuper_hardpolicy_moeplusbalance_initlr5e-1_'\
    'iter50000_t2s0.5_t4s0.2_dpr0.2_torchrandomerasing_bn0.9_momentum0.0_paddledevelop_4cards_0.25bs_align3'
train.eval_period = train.max_iter 
