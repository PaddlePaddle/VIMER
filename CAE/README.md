<div align="center">
<h1>CAE</h1>
<h3>Context Autoencoder for Self-Supervised Representation Learning</h3>
</div>

* This repo is the official implementation of [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026). It includes codes and models for the following tasks:
> **Semantic Segmentation**: See [SEGMENTATION.md](Segmentation/README.md). 



## Updates

***09/June/2022:***  The code of Pretrain and Segmentation is released.

***07/Feb/2022:***  The newest version is public at [arxiv](https://arxiv.org/abs/2202.03026).



## Introduction
CAE is a novel masked image modeling (MIM) approach for self-supervised representation pretraining. The goal is to pretrain an encoder by solving the pretext task: estimate the masked patches from the visible patches in an image. We first feed the visible patches into the encoder, extracting the representations. Then, we make predictions from visible patches to masked patches in the encoded representation space. We introduce an alignment constraint, encouraging that the representations for masked patches, predicted from the encoded representations of visible patches, are aligned with the masked patch presentations computed from the encoder. In other words, the predicted representations are expected to lie in the encoded representation space, which empirically shows the benefit to representation learning. Last, the predicted masked patch representations are mapped to the targets of the pretext task through a decoder. 
<br />  
In comparison to previous MIM methods (e.g., BEiT) that couple the encoding and pretext task completion roles, our approach benefits the separation of the representation learning (encoding) role and the pretext task completion role, improving the representation learning capacity and accordingly helping more on downstream tasks. In addition, we present the explanations about why contrastive pretraining and supervised pretraining perform similarly and why MIM potentially performs better. We demonstrate the effectiveness of our CAE through superior transfer performance in downstream tasks: semantic segmentation, and object detection and instance segmentation.

<div align=center><img src="https://github.com/PaddlePaddle/VIMER/blob/main/CAE/figs/CAE2.png" width="80%"></div>

## Result on ImageNet-1K

|   model  | pretrain | Linear Prob | accuracy | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|
| Vit-Base |   800e   |   90e   |   69.32%  | [Vit-Base-800e](https://vimer.bj.bcebos.com/CAE/pt_ep800_fp32_checkpoint-799.pd)|



## Usage

* Environment requirements
```bash
python 3.7
cuda: 11.0
cudnn: 8.0.4
gcc 8.2
```


* Installation PaddlePaddle

This code base needs to be executed on the [develep paddle](https://vimer.bj.bcebos.com/CAE/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl):
```
python3 -m pip install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl)

```

* Install requirements
CAE dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.
```
python3 -m pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### Training
A typical command To pre-train Vit-Base (recommended default) with multi-nodes distributed training, run the following on 4 nodes with 8 GPUs each:

```
DATA_PATH=/path/to/ImageNet1K/train
TOKENIZER_PATH=/path/to/dalle-weights
OUTPUT_DIR=/path/to/output
INIT_MODEL=/path/to/init_model

FLAGS_cudnn_exhaustive_search=True

python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/run_cae_pretraining.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 64 --lr 1.5e-3 --warmup_epochs 10 --epochs 800 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0 \
  --sincos_pos_emb \
  --mask_generator block \
  --num_mask_patches 75 \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 50 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --seed 0 \
  --log_dir vdl \
  --num_decoder_self_attention 4 \
  --dual_loss_weight 2 \
  --dual_path_ema 0 \
  --resume INIT_MODEL
```
**Notes:** 
- The INIT_MODEL can be loaded from [pt_model](https://vimer.bj.bcebos.com/CAE/pt_init.pdparams)


### Linear Probing
A typical command To run Linear Probing of  Vit-Base (recommended default) with multi-nodes distributed training, run the following on 4 nodes with 8 GPUs each:

```
DATA_PATH=/path/to/ImageNet1K
OUTPUT_DIR=/path/to/output 
MODEL_PATH=/path/to/pretrain_model

python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/run_linear_probing.py \
    --model cae_base_patch16_224 \
    --finetune $MODEL_PATH \
    --nb_classes 1000 \
    --batch_size 512 \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --data_path ${DATA_PATH} \
    --output_dir $OUTPUT_DIR'_linearprobing' \
    --log_dir $OUTPUT_DIR \
    --enable_linear_eval \
    --use_cls \
    --dist_eval \
    --save_freq 50 \
    --disable_rel_pos_bias \
    --linear_type standard \

```

### Attentive Probing
A typical command to run attentive probing of  Vit-Base (recommended default) with multi-nodes distributed training, run the following on 4 nodes with 8 GPUs each:

```
DATA_PATH=/path/to/ImageNet1K
OUTPUT_DIR=/path/to/output 
MODEL_PATH=/path/to/pretrain_model

python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/run_attentive_probing.py \
    --model cae_base_patch16_224 --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 --data_set IMNET --imagenet_default_mean_and_std \
    --output_dir $OUTPUT_DIR --batch_size 256 --lr 0.4 --update_freq 1 \
    --warmup_epochs 10 --epochs 90 \
    --weight_decay 0 --smoothing 0.0 --layer_decay 1.0 --drop_path 0.0 \
    --color_jitter 0.0 --mixup 0.0 --cutmix 0.0 --reprob 0.0 \
    --opt sgd --momentum 0.9 \
    --enable_linear_eval \
    --use_cls \
    --dist_eval \
    --no_auto_resume \
    --save_ckpt_freq 50 \
    --linear_type simple 
```




### Finetune
A typical command to run finetune of  Vit-Base (recommended default) with multi-nodes distributed training, run the following on 4 nodes with 8 GPUs each:

```
DATA_PATH=/path/to/ImageNet1K
OUTPUT_DIR=/path/to/output 
MODEL_PATH=/path/to/pretrain_model

python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/run_class_finetuning.py \
    --model cae_base_patch16_224 \
    --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 \
    --data_set IMNET \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --lr 8e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --sin_pos_emb \
    --dist_eval \
    --no_auto_resume
```




## Citing Context Autoencoder for Self-Supervised Representation Learning
```
@article{chen2022context,
  title={Context autoencoder for self-supervised representation learning},
  author={Chen, Xiaokang and Ding, Mingyu and Wang, Xiaodi and Xin, Ying and Mo, Shentong and Wang, Yunhao and Han, Shumin and Luo, Ping and Zeng, Gang and Wang, Jingdong},
  journal={arXiv preprint arXiv:2202.03026},
  year={2022}
}
```
