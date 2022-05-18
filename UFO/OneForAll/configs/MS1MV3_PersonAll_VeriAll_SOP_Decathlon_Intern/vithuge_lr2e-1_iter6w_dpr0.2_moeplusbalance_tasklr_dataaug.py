"""configs/MS1MV3_PersonAll_VeriAll_SOP_Decathlon_Intern/vithuge_lr2e-1_iter6w_dpr0.2_moeplusbalance_tasklr_dataaug.py
"""
from ..common import optimizer, train
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from data.build import build_train_set, build_reid_train_loader_lazy
from data.build import build_test_set, build_reid_test_loader_lazy
from data.build import build_face_test_set, build_face_test_loader_lazy
from data.build import build_imagenet_train_set, build_reid_train_imagenet_loader_lazy
from data.transforms.batch_ops.batch_operators import OpSampler
from data.transforms.build import build_transforms_lazy
from evaluation.reid_evaluation import ReidEvaluatorSingleTask
from evaluation.face_evaluation import FaceEvaluatorSingleTask
from evaluation.retri_evaluator import RetriEvaluatorSingleTask
from evaluation.clas_evaluator import ClasEvaluatorSingleTask
from modeling.backbones.vision_transformer_super_hardpolicy_moe import build_vit_backbone_lazy
from modeling.heads.embedding_head import EmbeddingHead
from modeling.meta_arch.multitask_super_hardpolicy_moe import MultiTaskBatchFuse

#模型并行中taskid —> globalrank号的映射关系
#globalrank2taskid是dict，key的类型为字符串，代表rank号；value类型为int，代表taskid
#'0':0, 表示rank=0对应的进程（卡）对应于task0
train.dp_degree = 1
train.globalrank2taskid = {
    '0': 0, '1': 999000, '2': 999001, '3': 999002, '4': 999003, '5': 999004, '6': 999005, '7': 999006, 
    '8': 1, '9': 999007, '10': 999008, '11': 999009, '12': 2, '13': 999010, '14': 999011, '15': 999012, 
    '16': 3, '17': 999013, '18': 999014, '19': 999015, '20': 4, '21': 5, '22': 999016, '23': 6, 
    '24': 7, '25': 8, '26': 9, '27': 999017, '28': 999018, '29': 999019, '30': 999020, '31': 999021, 
    '32': 999022, '33': 999023, '34': 10, '35': 11, '36': 12, '37': 13, '38': 14, '39': 15, 
    '40': 16, '41': 17, '42': 18, '43': 19, '44': 20, '45': 21, '46': 22, '47': 23
    }
train.split_list = {
    'face':8, 'person':4, 'veri':4, 'sop':4, 
    'task1':1, 'task2':2, 'task3':1, 'task4':1, 'task5':1, 'task6':8, 'task7':1,
    'task8':1, 'task9':1, 'task0':1, 'task10':1, 'task11':1, 
    'task16':1, 'task17':1, 'task18':1, 'task19':1, 'task20':1, 'task22':1, 'task23':1, 'task24':1, 
    }
dataloader = OmegaConf.create()
    
dataloader.train = L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',
             sample_prob=[]),
    task_loaders=L(OrderedDict)(
        face=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names=("MS1MV3",),
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
                )
            ),
            total_batch_size=1152,
            num_workers=1,
        ),
        person=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names=("PersonAll",),
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
                )
            ),
            total_batch_size=576,
            num_workers=1,
        ),
        veri=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names=("VeriAll",),
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
                )
            ),
            total_batch_size=576,
            num_workers=1,
        ),
        sop=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names=("SOP",),
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
                )
            ),
            total_batch_size=576,
            num_workers=1,
        ),
        task1=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("FGVCaircraft",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task2=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("CIFAR100",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task3=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Daimlerpedcls",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task4=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("DTD",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task5=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("GTSRB",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task6=L(build_reid_train_imagenet_loader_lazy)(
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
            total_batch_size=1152,
            num_workers=1,
            batch_ops=L(OpSampler)(
                class_num=1000, 
                CutmixOperator={"prob": 0.5, "alpha": 0.8}, 
                MixupOperator={"prob": 0.5, "alpha": 1.0}
                )
        ),
        task7=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Omniglot",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task8=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("SVHN",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task9=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Ucf101",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task0=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Oxford102Flower",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task10=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("PatchCamelyon",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task11=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Fer2013",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task16=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("OxfordPet",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task17=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("Food101",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task18=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("CIFAR10",),
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
            total_batch_size=256,
            num_workers=1,
        ),
        task19=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("SUN397",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task20=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("DF20",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task22=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("TsingHuaDogs",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task23=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("FoodX251",),
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
            total_batch_size=144,
            num_workers=1,
        ),
        task24=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'TrainingSampler'},
            train_set=L(build_imagenet_train_set)(
                names=("CompCars",),
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
            total_batch_size=144,
            num_workers=1,
        ),
    ),
)

dataloader.test = [
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="CALFW",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="CPLFW",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="LFW",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="AgeDB_30",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="CFP_FF",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            face=L(build_face_test_loader_lazy)(
                test_set=L(build_face_test_set)(
                    dataset_name="CFP_FP",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            person=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Market1501",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            person=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="DukeMTMC",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            person=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="MSMT17",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            veri=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="VeRi",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            veri=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="LargeVehicleID",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            veri=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="LargeVeRiWild",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            sop=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="SOP",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    )
                ),
                test_batch_size=128,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task1=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="FGVCaircraft",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task2=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="CIFAR100",
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[256, 256],
                        mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                    ),
                ),
                test_batch_size=288,
                num_workers=1,
            ),
        )
    ),
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task3=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Daimlerpedcls",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task4=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="DTD",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task5=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="GTSRB",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task7=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Omniglot",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task8=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="SVHN",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task9=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Ucf101",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task0=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Oxford102Flower",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task10=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="PatchCamelyon",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task11=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Fer2013",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task16=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="OxfordPet",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task17=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="Food101",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task18=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="CIFAR10",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task19=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="SUN397",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task20=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="DF20",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task22=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="TsingHuaDogs",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task23=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="FoodX251",
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
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',
                 sample_prob=[]),
        task_loaders=L(OrderedDict)(
            task24=L(build_reid_test_loader_lazy)(
                test_set=L(build_test_set)(
                    dataset_name="CompCars",
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
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(FaceEvaluatorSingleTask)(
        cfg=dict(),
    ),
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
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
    L(ClasEvaluatorSingleTask)(
        cfg=dict(),
    ),
]
weight_decay_norm=0.0001
bias_lr_factor=2.0
model = L(MultiTaskBatchFuse)(
    backbone=L(build_vit_backbone_lazy)(
        pretrain=True,
        pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
        input_size=[256, 256],
        depth='huge',
        sie_xishu=3.0,
        stride_size=14,
        drop_ratio=0.0,
        drop_path_ratio=0.2,
        attn_drop_rate=0.0,
        use_checkpointing=True,
        share_last=True,
        # global_pool=True, 
        patch_size=14,
        predefined_head_dim=80,
        n_tasks=32, 
        task_names=['face', 'person', 'veri', 'sop', 'task1', 'task2', 
                    'task3', 'task4', 'task5', 'task6', 'task7',
                    'task8', 'task9', 'task0', 'task10', 'task11', 
                    'task16', 'task17', 'task18', 'task19', 'task20', 
                    'task22', 'task23', 'task24'],
        globalrank2taskid = {
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, 
            '8': 1, '9': 1, '10': 1, '11': 1, '12': 2, '13': 2, '14': 2, '15': 2, 
            '16': 3, '17': 3, '18': 3, '19': 3, '20': 4, '21': 5, '22': 5, '23': 6, 
            '24': 7, '25': 8, '26': 9, '27': 9, '28': 9, '29': 9, '30': 9, '31': 9, 
            '32': 9, '33': 9, '34': 10, '35': 11, '36': 12, '37': 13, '38': 14, '39': 15, 
            '40': 16, '41': 17, '42': 18, '43': 19, '44': 20, '45': 21, '46': 22, '47': 23
            }, 
        taskname2learningscale ={
            'face':2, 'person':1, 'veri':2, 'sop':1, 
            'task1':0.25, 'task2':0.25, 'task3':0.25, 'task4':0.25, 
            'task5':0.25, 'task6':0.5, 'task7':0.25, 'task8':0.25, 
            'task9':0.25, 'task0':0.25, 'task10':0.25, 'task11':0.25, 
            'task16':0.25, 'task17':0.25, 'task18':0.25, 'task19':0.25, 
            'task20':0.25, 'task22':0.25, 'task23':0.25, 'task24':0.25, 
        }
    ),
    heads=L(OrderedDict)(
        face=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=50,
            margin=0.3,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            depth='base',
        ),
        person=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            depth='base',
        ),
        veri=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=20,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            depth='base',
        ),
        sop=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
            dropout=False,
            share_last=True,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            depth='base',
        ),
        task1=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task2=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task3=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task4=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task5=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
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
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task7=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task8=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task9=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task0=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task10=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task11=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task16=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task17=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task18=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task19=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task20=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task22=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task23=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
        task24=L(EmbeddingHead)(
            feat_dim=1280,
            norm_type='SyncBN',
            with_bnneck=False,
            pool_type="Identity",
            neck_feat="before",
            cls_type="classification",
            scale=1,
            margin=0.0,
            num_classes=0,
            pretrain_path='pretrained/mae_finetuned_vit_huge.pdmodel',
            load_head=False,
        ),
    ),
    pixel_mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    pixel_std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    n_tasks=32,
    choices={
            'num_heads': [16], 'mlp_ratio': [4.0], \
                'embed_dim': [1280], 'depth': [32], 'img_size': [256]
                },
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


# data settings

# model settings
model.backbone.drop_path_ratio = 0.2

# loss settings
loss_scale = 1.
model.task_loss_kwargs.task1.ce.scale = loss_scale
model.task_loss_kwargs.task2.ce.scale = loss_scale
model.task_loss_kwargs.task3.ce.scale = loss_scale
model.task_loss_kwargs.task4.ce.scale = loss_scale
model.task_loss_kwargs.task5.ce.scale = loss_scale
model.task_loss_kwargs.task6.ce.scale = loss_scale
model.task_loss_kwargs.task7.ce.scale = loss_scale
model.task_loss_kwargs.task8.ce.scale = loss_scale
model.task_loss_kwargs.task9.ce.scale = loss_scale
model.task_loss_kwargs.task0.ce.scale = loss_scale
model.task_loss_kwargs.task10.ce.scale = loss_scale
model.task_loss_kwargs.task11.ce.scale = loss_scale
model.task_loss_kwargs.task12.ce.scale = loss_scale
model.task_loss_kwargs.task13.ce.scale = loss_scale
model.task_loss_kwargs.task14.ce.scale = loss_scale
model.task_loss_kwargs.task15.ce.scale = loss_scale
model.task_loss_kwargs.task16.ce.scale = loss_scale
model.task_loss_kwargs.task17.ce.scale = loss_scale
model.task_loss_kwargs.task18.ce.scale = loss_scale
model.task_loss_kwargs.task19.ce.scale = loss_scale
model.task_loss_kwargs.task20.ce.scale = loss_scale
model.task_loss_kwargs.task21.ce.scale = loss_scale
model.task_loss_kwargs.task22.ce.scale = loss_scale
model.task_loss_kwargs.task23.ce.scale = loss_scale
model.task_loss_kwargs.task24.ce.scale = loss_scale

# learning rate settings
optimizer.base_lr = 2e-1
optimizer.lr_multiplier.base_lr = optimizer.base_lr
optimizer.lr_multiplier.eta_min = optimizer.base_lr * 2e-2

# iteration settings
train.max_iter = 60000
optimizer.lr_multiplier.max_epoch = 120
optimizer.lr_multiplier.warmup_iters = int(train.max_iter * 0.1)
optimizer.lr_multiplier.max_iters = train.max_iter - optimizer.lr_multiplier.warmup_iters
optimizer.lr_multiplier.iters_per_epoch = train.max_iter // optimizer.lr_multiplier.max_epoch
optimizer.lr_multiplier.warmup_factor = 2e-3

# amp settings
train.amp.enabled = False

# optimizer settings
optimizer.momentum = 0.0

train.output_dir = 'ufo_logs/ufo_decathlon_intern/vithuge_lr2e-1_iter6w_dpr0.2_moeplusbalance_tasklr_dataaug'
train.eval_period = train.max_iter
model.backbone.use_checkpointing = True
