English | [简体中文](README_ch.md)

# VIMER-UFO 2.0 (CV Foundation Model)

- [Instructions](#Instructions)
  * [Environment](#Environment)
  * [Distributed-training](#Distributed-training)
  * [Download the 17B pretrained model](#Download-17B-pretrained-model)

## Instructions

Demo is based on 48 A100 cards.

### Environment
Please use python3.7 and cuda11.0 

```bash
pip install -U pip==22.0.3
pip install -r requirements.txt.train
pip install faiss-gpu==1.7.2 --no-cache
pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

```
Configure environment variables.
```bash
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export PYTHONPATH=/path/to/AllInOne
export FASTREID_DATASETS=/path/to/data/train_datasets
export UFO_config=configs/MS1MV3_PersonAll_VeriAll_SOP_Decathlon_Intern/vithuge_lr2e-1_iter6w_dpr0.2_moeplusbalance_tasklr_dataaug.py
```

### Distributed-training

```bash
mpirun -npernode 1 --bind-to none python -m paddle.distributed.launch  --gpus "0,1,2,3,4,5,6,7" tools/ufo_trainsuper_moe.py  --config-file $UFO_config
```

### Download-17B-pretrained-model

Please send your request to vimer-ufo@baidu.com . The request may include your name and orgnization. We will notify you by email as soon as possible.

We thank for https://github.com/facebookresearch/detectron2 and https://github.com/JDAI-CV/fast-reid .

If you have any question, please contact xiteng01@baidu.com
