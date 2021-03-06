English | [简体中文](README_ch.md)
## _ViSTA_: Vision and Scene Text Aggregation for Cross-Modal Retrieval

- [Architecture](#architecture)
- [Pre-trained tasks](#pre-trained-tasks)
- [Cross-modal retrieval](#cross-modal-retrieval)
  * [Scene text aware cross-modal retrieval](#scene-text-aware-cross-modal-retrieval)
- [Quick experience](#quick-experience)
  * [Environment installation](#environment-installation)
  * [Download inference models](#download-inference-models)
  * [Infer fine-tuned models](#infer-fine-tuned-models)
  * [Visulization of cross-modal retrieval results](#visualization-of-cross-modal-retrieval-results)
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

ViSTA is a full transformer architecture to effectively aggregate vision and scene text, which is applicable in both scene text aware and scene text free retrieval scenarios. Visual appearance is considered to be the most important cue to understand images for cross-modal retrieval, while sometimes the scene text appearing in images can provide valuable information to understand the visual semantics. Most cross-modal retrieval approaches ignore the usage of scene text information, and directly adding this information may lead to performance degradation in scene text free scenarios. To tackle the modality missing problem of scene text, we propose a fusion token based transformer aggregation approach to exchange the relevant information among visual and scene text features. To further strengthen the visual modality, we develop dual contrastive learning losses to embed both image-text pairs and fusion-text pairs into a common cross-modal space. The proposed cross-modal retrieval framework can remarkably surpass existing methods for the scene text aware retrieval task and achieve better performance than state-of-the-art approaches on scene text free retrieval benchmarks as well. 

## Pre-trained tasks
We pretrained our ViSTA dual-encoder visual-language model on [Visual Genome (VG)](https://visualgenome.org/api/v0/api_home.html) dataset.

## Cross-modal retrieval
To achieve a strong feature representation for better retrieval accuracy, we adopt ViT-S(ViT-small) as image encoder, BERT-mini as scene text and text encoder. We fine-tuned our ViSTA-S model on Flickr30K, TextCaption(TC) and COCO-Text Captioned(CTC) train set. Then we evaluated our model on COCO-Text Captioned(CTC) dataset for the cross-modal retrieval task.

## Scene text aware cross-modal retrieval
   * Datasets 
     * [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset) Contains 31,000 images collected from Flickr, together with 5 reference sentences provided by human annotators.
     * [TextCaps](https://textvqa.org/textcaps/dataset/) Contains 145k captions for 28k images.
     * [COCO-Text Captioned](https://europe.naverlabs.com/research/computer-vision/stacmr-scene-text-aware-cross-modal-retrieval/) Train set contains 28415 captions describing 5683 images. We conduct cross-modal retrieval task on CTC-1K and CTC-5K test set.
   * Performance
     * image-to-text and text-to-image retrieval results on CTC-1K test set and CTC-5K test set.

        | Model                                      | CTC-1K<br>Image-to-text<br>R@1/R@5/R@10 | CTC-1K<br>Text-to-image<br>R@1/R@5/R@10 | CTC-5K<br>Image-to-text<br>R@1/R@5/R@10 | CTC-5K<br>Text-to-image<br>R@1/R@5/R@10 |          
        | :----------------------------------------- | :-------------------------------------: | :-------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
        | [SCAN](https://arxiv.org/abs/1803.08024)   | 36.3/63.7/75.2                          | 26.6/53.6/65.3                          | 22.8/45.6/54.3                                     | 12.3/28.6/39.9                                     |
        | [VSRN](https://arxiv.org/abs/1909.02701)   | 38.2/67.4/79.1                          | 26.6/54.2/66.2                          | 23.7/47.6/59.1                                     | 14.9/34.7/45.5                                     |
        | [STARNet](https://arxiv.org/abs/2012.04329)| 44.1/74.8/82.7                          | 31.5/60.8/72.4                          | 26.4/51.1/63.9                                     | 17.1/37.4/48.3                                     |
        | [ViSTA-S](https://arxiv.org/abs/2203.16778)| **52.6/77.9/87.2**                      | **36.7/66.2/77.8**                      | **31.8/56.6/67.8**                                 | **20.0/42.9/54.4**                                 |

## Quick experience

### Environment installation
Check [INSTALL.md](./doc/INSTALL.md) for installation instructions.

### Download inference models
| Model link                                              | Params(M) |
| :------------------------------------------------- | :-----------|
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147516" target="_blank">ViSTA-S for COCO-CTC image-text retrieval   </a>| 196 |

### Infer fine-tuned models
  
#### Evaluation on CTC-1K and CTC-5K
1. Download and extract the [COCO-CTC](https://aistudio.baidu.com/aistudio/datasetdetail/147436/0) dataset at current directory <./data>
2. Download infer modal [configs](https://aistudio.baidu.com/aistudio/datasetdetail/147517), vista.pdparams
3. Run shell script for peformance evaluation

CTC-1K：```sh eval_scripts/run_eval_ctc_1k_online_scene_text_2D_ocr.sh vista.pdparams```

CTC-5K：```sh eval_scripts/run_eval_ctc_5k_online_scene_text_2D_ocr.sh vista.pdparams```


### Visualization of cross-modal retrieval results
Image-to-text and text-to-image retrieval visualization results(compared with scene text free results)

- Image-to-text results

<div align="center">
    <img src="./doc/img2text_demo_s.png" width="800">
</div>

- Text-to-image results

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
