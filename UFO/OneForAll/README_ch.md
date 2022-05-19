简体中文 | [English](README.md)

# VIMER-UFO 2.0 (文心-CV大模型)

- [使用方案](#使用方案)
  * [环境配置](#环境配置)
  * [多机训练](#多机训练)
  * [170亿参数预训练模型下载](#170亿参数预训练模型下载)

## 使用方案

提供了Oneforall百亿大模型模型的训练方法
Demo为6机（48卡）80G A100的训练方法

### 环境配置

运行环境为python3.7，cuda11.0测试机器为A100。使用pip的安装依赖包，如下：
```bash
pip install -U pip==22.0.3
pip install -r requirements.txt.train
pip install faiss-gpu==1.7.2 --no-cache
pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

```
配置机器的环境变量
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/path/to/AllInOne
export FASTREID_DATASETS=/path/to/data/train_datasets
export UFO_config=configs/MS1MV3_PersonAll_VeriAll_SOP_Decathlon_Intern/vithuge_lr2e-1_iter6w_dpr0.2_moeplusbalance_tasklr_dataaug.py
```

### 多机训练

```bash
mpirun -npernode 1 --bind-to none python -m paddle.distributed.launch  --gpus "0,1,2,3,4,5,6,7" tools/ufo_trainsuper_moe.py  --config-file $UFO_config
```

### 170亿参数预训练模型下载

如果需要下载170亿参数的预训练模型，请发送邮件到 vimer-ufo@baidu.com . 申请邮件需要包含你的名字和组织（学校/公司）. 我们会尽快答复.

致谢：部分数据集构建和测评代码参考了https://github.com/facebookresearch/detectron2 和 https://github.com/JDAI-CV/fast-reid , 表示感谢！

如遇到问题请联系xiteng01@baidu.com
