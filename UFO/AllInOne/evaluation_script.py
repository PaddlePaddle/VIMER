# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# The main evaluation script for 
# 
import logging
import time
import os
import random
import datetime
from collections import OrderedDict
from collections.abc import Mapping

import torch
import paddle 
import paddle.distributed as dist
import numpy as np
import paddle.vision.transforms as T
from paddle.vision.transforms import to_tensor

from fastreid.data import samplers
from fastreid.data import CommDataset
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from modeling.backbones.vision_transformer import build_vit_backbone_lazy 
from modeling.meta_arch.multitask import MultiTaskBatchFuse
from modeling.heads.embedding_head import EmbeddingHead 
from evaluation.face_evaluation import FaceEvaluatorSingleTask
from evaluation.reid_evaluation import ReidEvaluatorSingleTask
from evaluation.retri_evaluator import RetriEvaluatorSingleTask
from evaluation.testing import print_csv_format
from evaluation.evaluator import inference_on_dataset

_root = os.getenv("FASTREID_DATASETS", "datasets")

def _inference_on_dataset(model, data_loader, evaluator, flip_test=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.evaLazyCall()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
        flip_test (bool): If get features with flipped images
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = dist.get_world_size()
    logger = logging.getLogger(__name__)
    if hasattr(data_loader, 'dataset'):
        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    model.evaLazyCall()
    with  paddle.no_grad():
        for idx, inputs in enumerate(data_loader):
            # print('inputs: ', inputs)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = modeLazyCall(inputs)
            # Flip test
            if flip_test:
                raise NotImplementedError
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = modeLazyCall(inputs)
                outputs = (outputs + flip_outputs) / 2

            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                logger.info(
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks.
    There is no need of transforming data to GPU in fast_batch_collator
    """
    elem = batched_inputs[0]
    if isinstance(elem, np.ndarray):
        # return paddle.to_tensor(np.concatenate([ np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0))
        return np.concatenate([ np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0)

    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}
    elif isinstance(elem, float):
        # return paddle.to_tensor(batched_inputs, dtype=paddle.float64)
        return np.array(batched_inputs, dtype=np.float64) 
    elif isinstance(elem, int):
        #return paddle.to_tensor(batched_inputs)
        return np.array(batched_inputs) 
    elif isinstance(elem, str):
        return batched_inputs

class MultiTaskDataLoader(object):
    """
    build MultiTaskDataLoader
    """
    def __init__(self, task_loaders, cfg):
        super().__init__()
        self.task_loaders = task_loaders
        self.cfg = cfg

        self.task_iters = {}
        for name, loader in self.task_loaders.items():
            self.task_iters[name] = iter(loader)

    def __iter__(self):
        return self

    def __len__(self):
        # TODO: make it more general
        return len(list(self.task_iters.values())[0])

    def __next__(self):
        batch = {}
        if self.cfg['sample_mode'] == 'batch':
            for name, iter_ in self.task_iters.items():
                batch[name] = next(iter_)
        elif self.cfg['sample_mode'] == 'sample':
            name = random.choices(self.task_iters.keys(), self.cfg.sample_prob)[0]
            batch[name] = next(self.task_iters[name])
        else:
            raise NotImplementedError

        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        for name, loader in self.task_loaders:
            loader.shutdown()

def build_reid_test_loader_lazy(test_set, test_batch_size, num_workers):
    """
    build reid test_loader for tasks of Person, Veri and Sop
    this test_loader only supports single gpu
    """
    mini_batch_size = test_batch_size // 1 #comm.get_world_size()
    data_sampler = samplers.InferenceSampler(test_set)
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=mini_batch_size)
    test_loader = paddle.io.DataLoader(
        dataset=test_set,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        num_workers=0
        )
    return test_loader


def build_test_set(dataset_name=None, transforms=None, **kwargs):
    """
    build test_set for the tasks of Person, Veri and Sop
    """
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    # if comm.is_main_process():
        # data.show_test()
    data.show_test()
    test_items = data.query + data.gallery
    test_set = CommDataset(test_items, transforms, relabel=False, dataset_name=data.dataset_name)

    # Update query number
    test_set.num_query = len(data.query)
    return test_set

def build_face_test_loader_lazy(test_set, test_batch_size, num_workers):
    """
    build face test_loader for tasks of Face, 
    this test_loader only supports single gpu
    """
    mini_batch_size = test_batch_size // 1 #comm.get_world_size()
    data_sampler = samplers.InferenceSampler(test_set)
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=mini_batch_size)
    test_loader = paddle.io.DataLoader(
            dataset=test_set,
            batch_sampler=batch_sampler,
            collate_fn=fast_batch_collator,
            num_workers=0
            )
    return test_loader

class TestFaceDataset(CommDataset):
    def __init__(self, img_items, labels, transforms=None, dataset_name=None):
        self.img_items = img_items
        self.labels = labels
        self.transforms = transforms
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        img = read_image(self.img_items[index])
        if self.transforms is not None: img = self.transforms(img)
        return {"images": img,}

def build_face_test_set(dataset_name=None, transforms=None, **kwargs):
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    # if comm.is_main_process():
        # data.show_test()
    data.show_test()
    test_set = TestFaceDataset(data.img_paths, data.is_same, transforms, data.dataset_name)
    test_set.num_query = len(data.img_paths)
    return test_set

class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        # img = img / 255.
        return img.astype('float32')

def build_transforms_lazy(is_train=True, **kwargs):
    """
    build transforms of image data,
    only support `is_train=False`
    """
    res = []
    if is_train:
        pass
    else:
        size_test = kwargs.get('size_test', [256, 128])
        do_crop = kwargs.get('do_crop', False)
        crop_size = kwargs.get('crop_size', [224, 224])
        # normalize
        mean = kwargs.get('mean', [0.485*255, 0.456*255, 0.406*255])
        std = kwargs.get('std', [0.229*255, 0.224*255, 0.225*255])

        if size_test[0] > 0:
            res.append(T.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation='bicubic'))
        if do_crop:
            res.append(T.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
        res.append(ToArray())
        res.append(T.Normalize(mean=mean, std=std))
    return T.Compose(res)

def LazyCall(cls_name):
    return cls_name

class Dict(object):
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

if __name__ == '__main__':
    # constructing dataloaders and datasets
    dataloaders = [
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="CALFW",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="CPLFW",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="LFW",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="CFP_FF",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="CFP_FP",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task1=LazyCall(build_face_test_loader_lazy)(
                    test_set=LazyCall(build_face_test_set)(
                        dataset_name="AgeDB_30",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=8,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="Market1501",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="DukeMTMC",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="MSMT17",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),               
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="VeRi",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="LargeVehicleID",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task3=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="LargeVeRiWild",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
        LazyCall(MultiTaskDataLoader)(
            cfg=dict(sample_mode='batch',
                    sample_prob=[]),
            task_loaders=LazyCall(OrderedDict)(
                task4=LazyCall(build_reid_test_loader_lazy)(
                    test_set=LazyCall(build_test_set)(
                        dataset_name="SOP",
                        transforms=LazyCall(build_transforms_lazy)(
                            is_train=False,
                            size_test=[384, 384],
                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                        )
                    ),
                    test_batch_size=8,
                    num_workers=0,
                ),
            )
        ),
    ]
    # face_test_set = build_face_test_set(
    #                     dataset_name="CALFW",
    #                     transforms=LazyCall(build_transforms_lazy)(
    #                         is_train=False,
    #                         size_test=[256, 256],
    #                         mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    #                         std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    #                     )
    # )
    # for data in face_test_set:
    #     break
    # face_data_loader = build_face_test_loader_lazy(face_test_set, 128, 4)
    # for data in face_data_loader:
    #     break



    # constructing evaluators
    evaluators_dict = [
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(FaceEvaluatorSingleTask=Dict()),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(ReidEvaluatorSingleTask=Dict(
            AQE=Dict(enabled=False), metric='cosine', rerank=Dict(enabled=False), ROC=Dict(enabled=False),) 
            ),
        OrderedDict(RetriEvaluatorSingleTask=Dict(recalls=[1, 10, 100, 1000]))
    ]

    # constructing models
    model = LazyCall(MultiTaskBatchFuse)(
    backbone=LazyCall(build_vit_backbone_lazy)(
        pretrain=False,
        pretrain_path='',
        input_size=[384, 384],
        depth='large',
        sie_xishu=3.0,
        stride_size=16,
        drop_ratio=0.0,
        drop_path_ratio=0.1,
        attn_drop_rate=0.0,
        use_checkpointing=False,
    ),
    heads=LazyCall(OrderedDict)(
        task1=LazyCall(EmbeddingHead)(
            feat_dim=1024,
            norm_type='BN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=50,
            margin=0.3,
            num_classes=0,
        ),
        task2=LazyCall(EmbeddingHead)(
            feat_dim=1024,
            norm_type='BN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
        ),
        task3=LazyCall(EmbeddingHead)(
            feat_dim=1024,
            norm_type='BN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=20,
            margin=0.2,
            num_classes=0,
        ),
        task4=LazyCall(EmbeddingHead)(
            feat_dim=1024,
            norm_type='BN',
            with_bnneck=True,
            pool_type="Identity",
            neck_feat="before",
            cls_type="CosSoftmax",
            scale=30,
            margin=0.2,
            num_classes=0,
        ),
    ),
    pixel_mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    pixel_std=[0.5 * 255, 0.5 * 255, 0.5 * 255],
    )

    #load init model
    state_dict= paddle.load('allinone_vitlarge.pdmodel')
    load_state_dict = {}
    for key, value in model.state_dict().items():
        if key not in state_dict:
            print('{} is not found in modelpth'.format(key))
        elif value.shape != state_dict[key].shape:
            print('the shape {} is unmatched: modelpath is {}, model is {}'.format(key, state_dict[key].shape, value.shape, ))
        else:
            load_state_dict[key] = state_dict[key]
    model.set_state_dict(load_state_dict)

    # run evaluation for each dataset
    rets = {}
    for idx, (dataloader, evaluator_dict) in enumerate(zip(dataloaders, evaluators_dict)):
        # evaluator_cfg.num_query = list(dataloader.task_loaders.values())[0].dataset.num_query
        # evaluator_cfg.labels = list(dataloader.task_loaders.values())[0].dataset.labels
        # evaluator = instantiate(evaluator_cfg)
        num_query = list(dataloader.task_loaders.values())[0].dataset.num_query
        labels = list(dataloader.task_loaders.values())[0].dataset.labels
        evaluator_type = list(evaluator_dict.keys())[0]
        evaluator_cfg = evaluator_dict[evaluator_type]
        kwargs = {}
        kwargs['num_query'] = num_query
        kwargs['labels'] = labels
        kwargs['cfg'] = evaluator_cfg
        evaluator=eval(evaluator_type)(**kwargs)
        ret = inference_on_dataset(
            model, dataloader, evaluator
        )
        print(ret)
        print_csv_format(ret)

        # TODO: dirty, make it elegant
        task_name = '.'.join(list(dataloader.task_loaders.keys()))
        dataset_name = dataloader.task_loaders[task_name].dataset.dataset_name
        for metric, res in ret.items():
            rets[f'{task_name}.{dataset_name}.{metric}'] = res
    print(rets)


