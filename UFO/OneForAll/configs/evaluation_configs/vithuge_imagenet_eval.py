from ..common import optimizer, train
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from data.build import build_train_set, build_reid_train_loader_lazy
from data.build import build_test_set, build_reid_test_loader_lazy
from data.build import build_face_test_set, build_face_test_loader_lazy
from data.build import build_imagenet_train_set
from data.transforms.build import build_transforms_lazy
from evaluation.reid_evaluation import ReidEvaluatorSingleTask
from evaluation.face_evaluation import FaceEvaluatorSingleTask
from evaluation.retri_evaluator import RetriEvaluatorSingleTask
from evaluation.clas_evaluator import ClasEvaluatorSingleTask
from modeling.backbones.vision_transformer import build_vit_backbone_lazy
from modeling.heads.embedding_head import EmbeddingHead
from modeling.meta_arch.multitask import MultiTaskBatchFuse

dataloader = OmegaConf.create()

dataloader.train = L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',
             sample_prob=[]),
    task_loaders=L(OrderedDict)(
        task6=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("ImageNet1kBD",),
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[256, 256],
                    do_rea=True,
                    rea_prob=0.5,
                    do_flip=True,
                    do_pad=True,
                    do_autoaug=True,
                    autoaug_prob=0.5,
                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                ),
            ),
            total_batch_size=1024,
            num_workers=1,
        ),
    ),
)

dataloader.test = [
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task6=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="ImageNet1kBD",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    ),
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
]

dataloader.evaluator = [
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
]
weight_decay_norm=0.0001
bias_lr_factor=2.0
model = L(MultiTaskBatchFuse)(
    backbone=L(build_vit_backbone_lazy)(
        pretrain=False,
        input_size=[256, 256],
        depth='huge',
        sie_xishu=3.0,
        stride_size=14,
        patch_size=14,
        drop_ratio=0.0,
        drop_path_ratio=0.2,
        attn_drop_rate=0.0,
        use_checkpointing=True,
        share_last=True,
    ),
    heads=L(OrderedDict)(
        task6=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            load_head=False,
        ),
    ),
    pixel_mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    pixel_std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    task_loss_kwargs=
    dict(
        face={
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
        },
        person={
            'loss_names': ('CrossEntropyLoss', 'TripletLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            'tri': {
                'margin': 0.0,
                'norm_feat': False,
                'hard_mining': True,
                'scale': 1.0,
            },
        },
        veri={
            'loss_names': ('CrossEntropyLoss', 'TripletLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            'tri': {
                'margin': 0.0,
                'norm_feat': False,
                'hard_mining': True,
                'scale': 1.0,
            },
        },
        sop={
            'loss_names': ('CrossEntropyLoss', 'TripletLoss'),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
                'prob': 1.0,
            },
            'tri': {
                'margin': 0.0,
                'norm_feat': False,
                'hard_mining': True,
                'scale': 1.0,
            },
        },
        task1={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task2={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task3={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task4={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task5={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task6={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task7={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task8={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task9={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task0={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task10={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task11={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task12={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task13={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task14={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task15={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task16={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task17={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task18={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task19={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task20={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task21={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task22={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task23={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task24={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        task25={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 1.0,
            },
        },
        fake1={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 0.0, #fake data
            },
        },
        fake2={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 0.0, #fake data
            },
        },
        fake3={  # loss name
            'loss_names': ('CrossEntropyLoss',),
            'ce': {
                'eps': 0.0,
                'scale': 0.0, #fake data
            },
        },
    ),
)

model.backbone.drop_path_ratio = 0.2
batchsize_scale = 1.0

# learning rate settings
optimizer.base_lr = 5e-2 * batchsize_scale
optimizer.lr_multiplier.base_lr = optimizer.base_lr
optimizer.lr_multiplier.eta_min = optimizer.base_lr * 2e-2

# iteration settings
train.max_iter = int(60000 // batchsize_scale)
optimizer.lr_multiplier.max_epoch = 120
optimizer.lr_multiplier.warmup_iters = int(train.max_iter * 0.08)
optimizer.lr_multiplier.max_iters = train.max_iter - optimizer.lr_multiplier.warmup_iters
optimizer.lr_multiplier.iters_per_epoch = train.max_iter // optimizer.lr_multiplier.max_epoch
optimizer.lr_multiplier.warmup_factor = 2e-3

# amp settings
train.amp.enabled = False

# 重要设置，对齐了pytorch
optimizer.momentum = 0.0

train.output_dir = 'ufo_logs/'
train.init_checkpoint = 'UFO_2.0_17B_release.pdmodel.totask6'

train.eval_period = 500
model.backbone.use_checkpointing = True
