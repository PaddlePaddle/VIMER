English | [简体中文](README_ch.md)
## _ViSTA_: Vision and Scene Text Aggregation for Cross-Modal Retrieval

- [Architecture](#architecture)
- [Pre-trained tasks](#pre-trained-tasks)
- [Cross-modal retrieval](#cross-modal-retrieval)
  * [Scene text aware cross-modal retrieval](#scene-text-aware-cross-modal-retrieval)
- [Quick experience](#quick-experience)
  * [Install PaddlePaddle](#install-paddlepaddle)
  * [Install PaddleNlp](#install-paddlenlp)
  * [Download inference models](#download-inference-models)
  * [Infer fine-tuned models](#infer-fine-tuned-models)
  * [Visulization of cross-modal retrieval results](#visualization-of-retrieval-results)
- [Citation](#citation)

## Architecture
<div align="center">
    <img src="./doc/overview-v2_s.png" width="900">
</div>
<p align="center"> Model Architecture of ViSTA </p>

For basic design concept of the architecture, please see our paper:
>[_**ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval**_](https://arxiv.org/abs/2203.16778)
>
>Mengjun Cheng\*, Yipeng Sun\*<sup>\+</sup>, Longchao Wang, Xiongwei Zhu, Kun Yao, Jie Chen, Guoli Song, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang (\*: equal contribution, <sup>\+</sup>: corresponding author)
>
>Accepted by CVPR 2022
> 

ViSTA is a full transformer architecture to effectively aggregate vision and scene text, which is applicable in both scene text aware and scene text free retrival scenarios. Visual appearance is considered to be the most important cue to understand images for cross-modal retrieval, while sometimes the scene text appearing in images can provide valuable information to understand the visual semantics. Most cross-modal retrieval approaches ignore the usage of scene text information, and directly adding this information may lead to performance degradation in scene text free scenarios. To tackle the modality missing problem of scene text, we propose a fusion token based transformer aggregation approach to exchange the relevant information among visual and scene text features. To further strengthen the visual modality, we develop dual contrastive learning losses to embed both image-text pairs and fusion-text pairs into a common cross-modal space. The proposed cross-modal retrieval framework can remarkably surpass existing methods for the scene text aware retrieval task and achieve better performance than state-of-the-art approaches on scene text free retrieval benchmarks as well. 

## Pre-trained tasks
- **ViSTA Visual Dual-Encoder Language Modeling:** We pretrained our ViSTA Visual Dual-Encoder Language Model on [Visual Genome (VG)](https://visualgenome.org/api/v0/api_home.html) dataset

## Cross-modal retrieval
To achieve a strong feature representation for better retrieval accuracy,we adopt VIT-S(vit_small_patch16_224) as image encoder, BERT-mini as scene text and text query encoder.We fine-tuned our ViSTA-S model on Flickr30K, TextCaption(TC) and COCO-Text Captioned(CTC) train set, then the cross-modal retrieval task is evaluated on COCO-Text Captioned(CTC) dataset.

## Scene text aware cross-modal retrieval
   * datasets 
     * [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset):contains 31,000 images collected from Flickr, together with 5 reference sentences provided by human annotators.
     * [TextCaps](https://textvqa.org/textcaps/dataset/):contains 145k captions for 28k images.
     * [COCO-Text Captioned](https://europe.naverlabs.com/research/computer-vision/stacmr-scene-text-aware-cross-modal-retrieval/):train set contains 28415 captions describing 5683 images. We conduct cross-modal retrieval task on CTC-1K and CTC-5K test set.
   * performance
     * image-to-text and text-to-image retrieval results on CTC-1K test set.

        | Model                                      | CTC-1K<br>Image-to-text<br>R@1/R@5/R@10| CTC-1K<br>Text-to-image<br>R@1/R@5/R@10|           
        | :----------------------------------------- | :------------------------------------------------: | :------------------------------------------------: |
        | [SCAN](https://arxiv.org/abs/1803.08024)   | 36.3/63.7/75.2                                     | 26.6/53.6/65.3                                     |
        | [VSRN](https://arxiv.org/abs/1909.02701)   | 38.2/67.4/79.1                                     | 26.6/54.2/66.2                                     |
        | [STARNet](https://arxiv.org/abs/2012.04329)| 44.1/74.8/82.7                                     | 31.5/60.8/72.4                                     |
        | ViSTA-S                                    | **52.6/77.9/87.2**                                 | **36.7/66.2/77.8**                                 |
     * image-to-text and text-to-image retrieval results on CTC-5K test set.

        | Model                                      | CTC-5K<br>Image-to-text<br>R@1/R@5/R@10| CTC-5K<br>Text-to-image<br>R@1/R@5/R@10|           
        | :----------------------------------------- | :------------------------------------------------: | :------------------------------------------------: |
        | [SCAN](https://arxiv.org/abs/1803.08024)   | 22.8/45.6/54.3                                     | 12.3/28.6/39.9                                     |
        | [VSRN](https://arxiv.org/abs/1909.02701)   | 23.7/47.6/59.1                                     | 14.9/34.7/45.5                                     |
        | [STARNet](https://arxiv.org/abs/2012.04329)| 26.4/51.1/63.9                                     | 17.1/37.4/48.3                                     |
        | ViSTA-S                                    | **31.8/56.6/67.8**                                 | **20.0/42.9/54.4**                                 |

## Quick experience

### Install PaddlePaddle
This code base needs to be executed on the `PaddlePaddle develop`. You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip, depending on the CUDA version, you can choose the PaddlePaddle code base corresponding to the adapted version:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### Install PaddleNlp
The installation command of PaddleNlp can find from this (https://github.com/PaddlePaddle/PaddleNLP) or use pip:

```bash
pip install paddlenlp
```

* Environment requirements
```bash
python 3.6+
numpy
Pillow
paddlenlp>=2.2.3
cuda>=10.1
cudnn>=7.6.4
gcc>=8.2
```

* Install requirements
ViSTA dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.
```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### Download inference models
| Model link                                              | Params(M) |
| :------------------------------------------------- | :-----------|
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147516" target="_blank">ViSTA-S for COCO-CTC image-text retrieval   </a>| 196 |

### Infer fine-tuned models
  
#### Evaluation on CTC-1K and CTC-5K
1. download and extract the [COCO-CTC](https://aistudio.baidu.com/aistudio/datasetdetail/147436/0) dataset at current directory <./data>
2. download infer modal [configs](https://aistudio.baidu.com/aistudio/datasetdetail/147517), vista.pdparams
3. run shell script for peformance evaluation

CTC-1K：sh eval_scripts/run_eval_ctc_1k_online_scene_text_2D_ocr.sh vista.pdparams

CTC-5K：sh eval_scripts/run_eval_ctc_5k_online_scene_text_2D_ocr.sh vista.pdparams


### Visualization of cross-modal retrieval results
image-to-text and text-to-image retrieval visualization results(compared with scene text free results)

- image-to-text results

<div align="center">
    <img src="./doc/img2text_demo_s.png" width="800">
</div>

- text-to-image results

<div align="center">
    <img src="./doc/text2image-demo_s.png" width="800">
</div>

## Citation
You can cite the related paper as below:

```
@article{cheng2022vista,
  title={ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval},
  author={Cheng, Mengjun and Sun, Yipeng and Wang, Longchao and Zhu, Xiongwei and Yao, Kun and Chen, Jie and Song, Guoli and Han, Junyu and Liu, Jingtuo and Ding, Errui and others},
  journal={arXiv preprint arXiv:2203.16778},
  year={2022}
}
```
