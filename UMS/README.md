English | [简体中文](README_ch.md)
## _UMS_: Unified Multi-Source Pre-training for Product
- [Model description](#model-description)
- [Principle introduction](#principle-introduction)
- [Unified product visual and image-text representation](#unified-product-visual-and-image-text-representation)
- [Extensive coverage of product search tasks](#extensive-coverage-of-product-search-tasks)
- [Model performance](#model-performance)
  * [Product visual retrieval task](#product-visual-retrieval-task)
  * [Product multi-modal retrieval task](#product-multi-modal-retrieval-task)
  * [Image-text cross-modal retrieval task](#image-text-cross-modal-retrieval-task)
- [Application scenarios](#application-scenarios)
- [Quick experience](#quick-experience)
  * [Environment installation](#environment-installation)
  * [Download inference models](#download-inference-models)
  * [Downstream task model inference](#downstream-task-model-inference)
- [Citation](#citation)

## Model description
Based on the massive amount of Internet product graphic information, Baidu proposed a product image-text representation pre-training model unified modeling multi-source information, which is named VIMER-UMS (Vision Foundation Models for Enhanced Representation - Unified Multi-Source Pre-training for Product). It is the first commodity multimodal pre-training model in the industry unifying single visual modality and multi-source image-text modality representation. 

For the problem of incomplete modal information in the multi-source image-text information application scenario, the VIMER-UMS model achieves unified image-text representation pre-training. It uses a multi-task learning framework to construct the comparison of visual features and multi-source image-text features. The VIMER-UMS model covers both visual single modal and multimodal retrieval tasks. It significantly optimizes the recommendation experience of visual commodity retrieval and multimodal commodity retrieval.

## Principle introduction
Existing multimodal image-text pre-training methods are mainly oriented to cross-modal image-text retrieval, multimodal understanding, and generation tasks. They focus on the relationship representation of image and text modal features while are insufficient to support the downstream visual-only tasks. The large-scale image-text pre-training methods represented by OpenAI CLIP and Google Align rely on a large number of training resources and billion-level big data. The optimization of visual downstream tasks also relies heavily on massive data. Such high cost restricts the large-scale application of large multimodal models.

<div align="center">
    <img src="./doc/fig1.png" width="800">
</div>
<p align="center">Figure1. Principle introduction </p>

Furthermore, multimodal linked data in real scenes is not limited to simple image-text pairs. As shown in Fig.1, compared with existing large-scale image-text pre-training, multi-source information refers to information sources with multiple dimensions. Taking the commodity retrieval scenario as an example, multi-source information includes the multi-dimensional multimodal information of text modality (For example, search input, scene text, text titles, and category labels) and visual modality (For example, commodity images and same labels). They contain rich semantic associations and have great potential and application value. However, in practical applications, multi-source information is incomplete usually, which is a severe challenge for multi-source information modal modeling applications.

In response to the above problems, Baidu proposed a multi-source information unified modeling commodity image-text representation pre-training model VIMER-UMS for the commodity retrieval scenario. It aims to unify the visual modal, image-text multimodal retrieval tasks. It overcomes the problem of incomplete modal information in multi-source information scenarios and also improves the performance of visual and multimodal retrieval tasks.

## Unified product visual and image-text representation
Based on the end-to-end Transformer training method, VIMER-UMS provides a unified expression structure for multi-source commodity information through visual encoding, text encoding, fusion encoding, and search query encoding. Since the existing mainstream multimodal pre-training methods rely on language as weakly supervised correlation signals, the visual representation ability is degraded. To solve this problem, VIMER-UMS establishes a multi-task pre-training framework of visual and multi-source image-text comparison, which achieves a unified enhanced representation of visual features and image-text features.

<div align="center">
    <img src="./doc/fig2.png" width="500">
</div>
<p align="center">Figure2. VIMER-UMS </p>

## Extensive coverage of product search tasks
For practical business applications, the VIMER-UMS model could use a small amount of labeled or unlabeled data to efficiently achieve downstream commodity visual retrieval and multimodal retrieval capabilities.

<div align="center">
    <img src="./doc/fig3.png" width="600">
</div>
<p align="center">Figure3. Product search tasks </p>

## Model performance
The VIMER-UMS model realizes the SOTA performance of multiple commodity downstream visual retrieval and cross-modal retrieval tasks. It also supports direct deployment and pre-training fine-tuning applications.

### Product visual retrieval task
  * Dataset
    * [SOP](https://cvgl.stanford.edu/projects/lifted_struct/): This dataset contains 22,634 classes and a total of 120,053 annotated product images. 11,318 classes (59,551 images) are used for training and the other 11,316 (60,502 images) classes are used for testing.
    * [InShop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html): This dataset contains 7,982 classes and a total of 52,712 annotated images. The different views of each product instance are included.
  * Implementation: The downstream fine-tuning visual retrieval tasks of SOP and InShop are evaluated by Recall@1. It bases on the prediction library of PaddlePaddle GPU and supports fast deployment of single-card GPU.

    | Model                                          |                Downstream fine-tuning method                | Resolution |    SOP    |  InShop   |
    | :-------------------------------------------------- | :-----------------------------------------: | :----: | :-------: | :-------: |
    | UMS(ViT-Base)                                       |                  Rank Loss                  |  224   | 88.72 | 94.70 |

### Product multi-modal retrieval task
  * Dataset
    * [Product1M](https://github.com/zhanxlin/Product1M): The multimodal product dataset contains 1,182,083 training samples (a pair of product images and caption text descriptions). It used 2,673 test samples and 40,033 products gallery samples as retrieval evaluation data.
  * Implementation: The downstream fine-tuning multimodal product retrieval task of the Product1M uses the mAP@N (mean Average Precision, N=10, 50, 100) to evaluate the ranking effect of the retrieval.

    | Model                                          | Resolution |  mAP@10   |  mAP@50   |  mAP@100  |
    | :-------------------------------------------------- | :----: | :-------: | :-------: | :-------: |
    | UMS(ViT-Base)                                       |  224   | 85.68 | 83.10 | 81.13 |


### Image-text cross-modal retrieval task
  * Datasets
     * [COCO-Text Captioned (CTC)](https://europe.naverlabs.com/research/computer-vision/stacmr-scene-text-aware-cross-modal-retrieval/): The train set contains 28,415 captions describing 5,683 images. The CTC-1K test set is used for both image-to-text and text-to-image retrieval tasks.
  * Implementation: The downstream image-to-text and text-to-image retrieval tasks of the CTC are evaluated by Recall@N (N=1, 5, 10).

     |                     Model                      | Resolution | CTC-1K<br>Image-to-text<br>R@1/R@5/R@10 | CTC-1K<br>Text-to-image<br>R@1/R@5/R@10 |
     | :-------------------------------------------------: | :----: | :------------------------------: | :------------------------------: |
     |                    UMS(ViT-Base)                    |  224   |    64.90/88.20/93.90     |        49.24/77.08/86.18         |


## Application scenarios
The VIMER-UMS model could be implemented in multiple business scenarios. It could effectively solve the various problems of single modal and multimodal downstream tasks, and alleviates the industry pain points of inefficient identification, customization, and optimization of offline retail products.

1. **Product search**: The functions of retrieving products by text or images are used to find products of the same or similar style, or take pictures to identify products, which is convenient for searching products and recommending related products.

2. **Product recommendation**: Facing the e-commerce search platform, identify the content of goods and the intention to bring goods, improve the quality and quantity of goods displayed, and then improve the platform's conversion and monetization capabilities.

3. **Offline retail digitalization**: Facing the fast-moving consumer goods industry, accurately identifies the types and quantities of shelves, freezers, and end racks, as well as displayed goods, enabling brands to achieve digital insights and efficient sales decisions at outlets.

VIMER-UMS image-text representation pre-training will be integrated into Baidu's zero-threshold AI development platform - Retail Edition [EasyDL Retail Industry Edition](https://ai.baidu.com/easydl/retail) in the near future, so stay tuned.

## Quick experience

### Environment installation
Check [INSTALL.md](./doc/INSTALL.md) for installation instructions.

### Download inference models

| Model link                                              | Params(M) |
| :------------------------------------------------- | :-----------|
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/166172" target="_blank">UMS model for Product1M multi-modal retrieval   </a>| 1113 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/166168" target="_blank">UMS model for SOP image retrieval   </a>| 328 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/166166" target="_blank">UMS model for InShop image retrieval   </a>| 328 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/166171" target="_blank">UMS model for COCO-CTC image-text retrieval   </a>| 1168 |

### Downstream task model inference
  
#### Evaluation on Product1M

1. Download and extract the [Product1M](https://github.com/zhanxlin/Product1M) dataset at current directory <./Product1M>
2. Download infer modal [config_p1m](https://aistudio.baidu.com/aistudio/datasetdetail/166291), ums_product1m.pdparams
3. Run shell script for peformance evaluation on Product1M dataset
```
sh eval_scripts/run_eval_ums_base_product1m.sh ums_product1m.pdparams
```

#### Evaluation on SOP

1. Download and extract the [SOP](https://cvgl.stanford.edu/projects/lifted_struct/) dataset at current directory <./Stanford_Online_Products>
2. Download infer modal ums_sop.pdparams
3. Run shell script for peformance evaluation on SOP dataset
```
sh eval_scripts/run_eval_ums_base_sop.sh
```

#### Evaluation on InShop

1. Download and extract the [InShop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset at current directory <./inshop_dataset>
2. Download infer modal ums_inshop.pdparams
3. Run shell script for peformance evaluation on InShop datasets
```
sh eval_scripts/run_eval_ums_base_inshop.sh
```

#### Evaluation on COCO-CTC

1. Download and extract the [COCO-CTC](https://aistudio.baidu.com/aistudio/datasetdetail/166165) dataset at current directory <./data>
2. Download infer modal [config_ctc](https://aistudio.baidu.com/aistudio/datasetdetail/166292), ums_ctc.pdparams
3. Run shell script for peformance evaluation on COCO-CTC datasets
```
sh eval_scripts/run_eval_ums_base_ctc.sh ums_ctc.pdparams
```

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
