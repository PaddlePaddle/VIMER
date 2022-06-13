# CAE for Semantic Segmentaion

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/pdf/2202.03026.pdf). 
It is based on [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).


## Updates
***06/12/2022*** Initial commits


## Results and Models
### ADE20K
| Backbone | Method | Crop Size | Lr Schd | mIoU | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CAE-B | UperNet | 512x512 | 160K | 49.69 | 81M | 1038G | [config](https://github.com/PaddlePaddle/VIMER/blob/main/CAE/Segmentation/configs/upernet/upernet_cae_base_ade20k_512x512_160k.yml) | [github]()/[baidu]() | [model](https://vimer.bj.bcebos.com/CAE/seg_base.pdparams) |


**Notes**: 
- **Pre-trained models can be downloaded from [here](https://vimer.bj.bcebos.com/CAE/pt_ep800_fp32_checkpoint-799.pd)**.


## Usage

### Installation PaddlePaddle

This code base needs to be executed on the `PaddlePaddle 2.1.2+`. You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip:

```bash
# The installation command follows:
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

* Environment requirements
```bash
python 3.7.6
opencv-python 4.1.0.25
tqdm 4.32.2
cuda >= 10.1
cudnn >= 7.6.4
gcc >= 8.2
```

* Install requirements
CAE-Seg dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.
```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```


### Inference
```
# multi-gpu testing
python -m paddle.distributed.launch val.py \
       --config <CONFIG_FILE> --model_path <SEG_CHECKPOINT_FILE>
```
For example, to infer an UPerNet model with the given model, run:
```
python -u -m paddle.distributed.launch val.py \
      --config configs/upernet/upernet_cae_base_ade20k_512x512_160k.yml \ 
      --model_path https://vimer.bj.bcebos.com/CAE/pt_ep800_fp32_checkpoint-799.pd 
``` 

### Training

```
# multi-gpu training
python -u -m paddle.distributed.launch train.py \
--config  <CONFIG_FILE> \
--do_eval --use_vdl --save_interval <num> --save_dir output/upernet_cae_fpn \
--num_workers 4
```
For example, to train an UPerNet model with a `CAE-B` backbone and 8 gpus, run:

```
python -u -m paddle.distributed.launch train.py \
--config configs/upernet/upernet_cae_base_ade20k_512x512_160k.yml \
--do_eval --use_vdl --save_interval 8000 --save_dir output/upernet_cae_fpn \
--num_workers 4
```

**Notes:** 
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


## Citing Context Autoencoder for Self-Supervised Representation Learning
```
@article{chen2022context,
  title={Context autoencoder for self-supervised representation learning},
  author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2202.03026},
  year={2022}
}
```
