"""unitest.py 
"""
from collections import OrderedDict

from data.build import MultiTaskDataLoader
from data.build import build_train_set, build_reid_train_loader_lazy
from data.build import build_test_set, build_reid_test_loader_lazy
from data.build import build_face_test_set, build_face_test_loader_lazy
from data.transforms.build import build_transforms_lazy

def LazyCall(cls_name):
    """LazyCall
    """
    return cls_name


class Dict(object):
    """Dict
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

L=LazyCall
dataloader_train = L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',
             sample_prob=[]),
    task_loaders=L(OrderedDict)(
        # task1=L(build_reid_train_loader_lazy)(
        #     sampler_config={'sampler_name': 'TrainingSampler',
        #                     'num_instance': 4},
        #     train_set=L(build_train_set)(
        #         names = ("MS1MV3",),
        #         transforms=L(build_transforms_lazy)(
        #             is_train=True,
        #             size_train=[256, 256],
        #             do_rea=True,
        #             rea_prob=0.5,
        #             do_flip=True,
        #             do_pad=True,
        #             do_autoaug=True,
        #             autoaug_prob=0.5,
        #             mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #             std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #         )
        #     ),
        #     total_batch_size=1024,
        #     num_workers=8,
        # ),
        task2=L(build_reid_train_loader_lazy)(
            sampler_config={'sampler_name': 'NaiveIdentitySampler',
                            'num_instance': 4},
            train_set=L(build_train_set)(
                names = ("PersonAll",),
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
            total_batch_size=512,
            num_workers=8,
        ),
        # task3=L(build_reid_train_loader_lazy)(
        #     sampler_config={'sampler_name': 'NaiveIdentitySampler',
        #                     'num_instance': 4},
        #     train_set=L(build_train_set)(
        #         names = ("VeriAll",),
        #         transforms=L(build_transforms_lazy)(
        #             is_train=True,
        #             size_train=[256, 256],
        #             do_rea=True,
        #             rea_prob=0.5,
        #             do_flip=True,
        #             do_pad=True,
        #             do_autoaug=True,
        #             autoaug_prob=0.5,
        #             mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #             std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #         )
        #     ),
        #     total_batch_size=512,
        #     num_workers=8,
        # ),
        # task4=L(build_reid_train_loader_lazy)(
        #     sampler_config={'sampler_name': 'NaiveIdentitySampler',
        #                     'num_instance': 4},
        #     train_set=L(build_train_set)(
        #         names = ("SOP",),
        #         transforms=L(build_transforms_lazy)(
        #             is_train=True,
        #             size_train=[256, 256],
        #             do_rea=True,
        #             rea_prob=0.5,
        #             do_flip=True,
        #             do_pad=True,
        #             do_autoaug=True,
        #             autoaug_prob=0.5,
        #             mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #             std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
        #         )
        #     ),
        #     total_batch_size=512,
        #     num_workers=8,
        # ),
    ),
)


if __name__ == '__main__':
    from PIL import Image
    
    train_transform = build_transforms_lazy(
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
    image = Image.open('/root/paddlejob/workspace/env_run/data/datasets/'
    'PersonAll/mask_train_v3/1250/0403_c2s1_095646_08.jpg')
    for i in range(10):
        image = train_transform.transforms[i](image)
    image = Image.open('/root/paddlejob/workspace/env_run/data/datasets/'
    'PersonAll/mask_train_v3/1250/0403_c2s1_095646_08.jpg')
    image = train_transform(image)

    task1_dataloader = dataloader_train.task_loaders['task2']
    task1_train_set = task1_dataloader.dataset
    for data in task1_dataloader.dataset:
        break
    # for data in  task1_dataloader:
    #     print(len(data))
    #     pass
    iter1 = task1_dataloader()
    while True:
        data = next(iter1)
        print(len(data))
        iter1 = task1_dataloader()
    while True:
        data = next(iter1)
        print(len(data))
    print('ok')
    

