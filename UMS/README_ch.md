简体中文 | [English](README.md)
## _UMS_: Unified Multi-Source Pre-training for Product
- [模型说明](#模型说明)
- [原理介绍](#原理介绍)
- [统一商品视觉与图文表征](#统一商品视觉与图文表征)
- [广泛涵盖商品搜索任务](#广泛涵盖商品搜索任务)
- [模型效果](#模型效果)
  * [商品视觉检索任务](#商品视觉检索任务)
  * [商品多模态检索任务](#商品多模态检索任务)
  * [图文跨模态检索任务](#图文跨模态检索任务)
- [应用场景](#应用场景)
- [快速体验](#快速体验)
  * [安装PaddlePaddle](#安装PaddlePaddle)
  * [安装PaddleNlp](#安装PaddleNlp)
  * [下载推理模型](#下载推理模型)
  * [下游任务模型推理](#下游任务模型推理)
- [引用](#引用)

## 模型说明
基于海量的互联网商品图文信息，百度提出多源信息统一建模的商品图文表征预训练模型 **VIMER-UMS** (**Vi**sion Foundation **M**odels for **E**nhanced **R**epresentation - **U**nified **M**ulti-**S**ource Pre-training for Product)，是行业首个统一视觉与多源图文表征的商品多模态预训练模型。

针对海量多源图文模态应用场景中普遍存在的模态信息残缺问题，VIMER-UMS 通过构建视觉特征与多源图文对比的多任务学习框架，实现统一图文表征预训练，同时覆盖视觉单模态、多模态识别检索任务，显著改善商品视觉搜索、广告识别、多模态商品搜索推荐体验，高效提升线下零售数字化识别效果，解决行业小样本定制优化痛点。

## 原理介绍
现有多模态图文预训练方法主要面向图文跨模态搜索、多模态理解与生成任务，侧重对图文模态特征的关系表征，然而对单模态视觉下游任务效果支持不足。以OpenAI CLIP、Google Align为代表的图文预训练模型依赖大量训练资源及亿级大数据，视觉下游任务优化效果严重依赖海量数据，高昂成本制约了多模态大模型规模化应用。

<div align="center">
    <img src="./doc/fig1.png" width="800">
</div>
<p align="center"> 图1. 基于多源信息的商品多模态大数据 </p>

在真实多模态搜索应用场景中，多模态关联数据不仅限于两维的图文对数据形式。如图1所示，相比现有大规模图文预训练，多源信息是指具有多维度的信息来源。以商品搜索场景为例，包括文本模态（搜索输入、场景文字、文本标题、类目标签）、视觉模态（商品图、同款标签）的多维多模态信息，其中蕴含丰富的语义关联，具有极大的挖掘利用潜力与应用价值。在实际应用中，多源信息通常存在模态信息缺失的问题，多源信息的高效表征及应用面临重要挑战。


针对以上问题，百度面向商品搜索场景，提出了多源信息统一建模的商品图文表征预训练模型 VIMER-UMS，旨在统一视觉模态、图文多模态搜索表征任务，克服多源信息场景下模态信息残缺的问题，同时提升视觉、多模态搜索任务效果。

## 统一商品视觉与图文表征
VIMER-UMS 基于端到端Transformer训练方式，通过视觉编码、文本编码、融合编码、搜索查询编码，提供多源商品信息的统一表达结构。由于现有主流多模态预训练方法依靠语言作为弱监督关联信号，视觉表征能力存在退化现象。为了解决该问题，VIMER-UMS 通过建立视觉与多源图文对比多任务预训练，实现视觉特征、图文特征的统一增强表征。
<div align="center">
    <img src="./doc/fig2.png" width="500">
</div>
<p align="center"> 图2. VIMER-UMS商品图文表征预训练 </p>

## 广泛涵盖商品搜索任务
针对实际业务应用，基于 VIMER-UMS 商品图文表征预训练模型，使用少量标注或无标注数据，高效实现下游商品视觉检索、细粒度识别、多模态搜索能力。
<div align="center">
    <img src="./doc/fig3.png" width="600">
</div>
<p align="center"> 图3. 商品搜索下游任务 </p>

## 模型效果
基于 VIMER-UMS 商品图文表征预训练模型，实现多个商品下游视觉检索、跨模态检索任务 SOTA 效果，支持直接部署落地与预训练微调应用。

### 商品视觉检索任务
  * 数据集
     * [SOP](https://cvgl.stanford.edu/projects/lifted_struct/)数据集包含 22,634 款商品、共 120,053 张有标注图片用于评估商品视觉检索效果。训练图片 59,551 张、商品 11,318 类，测试图片60,502 张、商品 11,316 类。
     * [InShop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)数据集包含每个商品的不同角度图片共 52,712 张，包含 7,982 件商品。

  * 商品SOP、服饰InShop下游任务微调结果：商品视觉检索基于 Recall@1 进行效果评估，基于PaddlePaddle GPU预测库，支持单卡快速部署应用。
  
     | 模型          | 下游微调方法                                 | SOP            | InShop         | Time/ms         |
     | :------------ | :------------------------------------------: | :------------: | :------------: | :-------------: |
     | 现有SOTA      | [ROADMAP](https://arxiv.org/abs/2110.01445)  | 86.0           | 91.9           | ~12             |
     | ViT_small     | Rank Loss                                    | 85.8           | 92.9           | ~12             |
     | ViT_base      | Rank Loss                                    | 87.1           | 93.6           | ~18             |
     | UMS      | Rank Loss                                    | **88.5**       | **94.7**       | ~18             |

### 商品多模态检索任务
  * 数据集
     * [Product1M](https://github.com/zhanxlin/Product1M)多模态商品数据集包含 1,182,083 个训练样本（一对商品图与标题文本描述）、2,673 个测试样本以及 40,033 个商品底库样本作为搜索评测数据。
  * 商品多模态检索Product1M下游任务微调结果：采用 mAP@R指标（mean Average Precision）对搜索排序效果进行评估。
      | 模型                                        | mAP@10     | mAP@50     | mAP@100     | Time/ms     |
      | :------------------------------------------ | :--------: | :--------: | :---------: | :---------: |
      | [CLIP](https://arxiv.org/abs/2103.00020)    | 70.25      | 69.28      | 67.3        | ~40         |
      | [UNITER](https://arxiv.org/abs/1909.11740)  | 74.69      | 71.02      | 70.93       | ~900        |
      | [CAPTURE](https://arxiv.org/abs/2107.14572) | 79.36      | 74.79      | 74.63       | ~900        |
      | UMS                                         | **83.27**  | **78.69**  | **76.01**   | **~40**     |

### 图文跨模态检索任务
   * 数据集
     * [COCO-Text Captioned](https://europe.naverlabs.com/research/computer-vision/stacmr-scene-text-aware-cross-modal-retrieval/)数据集包含5683张训练图像，对应28,415个文本描述, CTC-1K用于图搜文、文搜图任务评测。
   
   * COCO-CTC数据集图搜文、文搜图任务：Recall评测，下游微调效果如下
 
        | 模型                                        | CTC-1K<br>图搜文<br>R@1/R@5/R@10            | CTC-1K<br>文搜图<br>R@1/R@5/R@10            |           
        | :------------------------------------------ | :-----------------------------------------: | :-----------------------------------------: |
        | [STARNet](https://arxiv.org/abs/2012.04329) | 44.1/74.8/82.7                              | 31.5/60.8/72.4                              |
        | [ViSTA](https://arxiv.org/abs/2203.16778)   | 52.5/77.9/87.2                              | 36.7/66.2/77.8                              |
        | UMS                                         | **70.8/90.2/94.3**                          | **54.0/80.1/87.8**                          |

## 应用场景
基于 VIMER-UMS 商品图文表征预训练，在实际商品拍照、多模态搜索与内容推荐、商品广告、零售线下数字化等多个业务场景取得效果提升并落地，有效解决单模态、多模态下游任务多样难题，显著缓解线下零售商品识别定制优化低效的行业痛点。

1、**商品搜索**：文本搜商品、图片搜商品等功能，用于找同款及相似框商品、拍照识货场景，便于搜索商品以及相关商品推荐。

2、**商品推荐**：面向电商搜索平台，对内容进行商品识别和带货意图识别，提升商品展现质量与数量，大幅提升平台转化和变现能力。

3、**线下零售数字化**：面向快速消费品行业，打造 [数字化访销解决方案](https://ai.baidu.com/solution/fmcg)，精准识别货架、冰柜和端架及陈列商品种类与数量，赋能品牌商实现网点数字化洞察与高效销售决策。 

VIMER-UMS 商品图文表征预训练近期将集成至百度零门槛AI开发平台-零售版[EasyDL零售行业版](https://ai.baidu.com/easydl/retail)中，敬请期待。

## 快速体验

### 安装PaddlePaddle
本代码库基于`PaddlePaddle develop`, 可参考[paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick)进行环境配置，或者使用pip进行安装，根据CUDA版本不同，可自行选择对应适配版本的PaddlePaddle代码库:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### 安装PaddleNlp
PaddleNlp代码可参考(https://github.com/PaddlePaddle/PaddleNLP) 或者使用pip进行安装:

```bash
pip install paddlenlp
```

* 环境要求
```bash
python 3.6+
numpy
Pillow
paddlenlp>=2.2.3
matplotlib
faiss-cpu
cuda>=10.1
cudnn>=7.6.4
gcc>=8.2
```

* 安装要求
UMS的依赖库已在`requirements.txt`中列出，你可以使用以下命令行进行依赖库安装：
```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### 下载推理模型
| 下载链接                                              | 参数量(M) |
| :------------------------------------------------- | :-----------|
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147518" target="_blank">UMS model for Product1M multi-modal retrieval   </a>| 751 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147519" target="_blank">UMS model for SOP image retrieval   </a>| 327 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147520" target="_blank">UMS model for InShop image retrieval   </a>| 327 |
| <a href="https://aistudio.baidu.com/aistudio/datasetdetail/147541" target="_blank">UMS model for COCO-CTC image-text retrieval   </a>| 749 |

### 下游任务模型推理

#### Product1M数据集上评测流程
1. 下载并解压[Product1M](https://github.com/zhanxlin/Product1M)数据集到当前根目录下<./Product1M>
2. 下载模型：[configs](https://aistudio.baidu.com/aistudio/datasetdetail/147523)、ums_product1m.pdparams
3. 运行shell脚本进行Product1M图文检索端到端评测
```
sh eval_scripts/run_eval_ums_base_product1m.sh itcl_product1m.pdparams
```

#### SOP数据集评测流程
1. 下载并解压[SOP](https://cvgl.stanford.edu/projects/lifted_struct/)数据集到当前根目录下<./Stanford_Online_Products>
2. 下载模型：ums_sop.pdparams
3. 运行shell脚本进行SOP商品图检索端到端评测
```
sh eval_scripts/run_eval_ums_base_sop.sh 
```

#### InShop数据集评测流程
1. 下载并解压[InShop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)数据集到当前根目录下<./inshop_dataset>
2. 下载模型：ums_inshop.pdparams
3. 运行shell脚本进行InShop商品图检索端到端评测
```
sh eval_scripts/run_eval_ums_base_inshop.sh 
```

#### CTC-1K数据集评测流程
1. 下载并解压[COCO-CTC](https://aistudio.baidu.com/aistudio/datasetdetail/147436/0)数据集到当前根目录下<./data>
2. 下载模型：[configs](https://aistudio.baidu.com/aistudio/datasetdetail/147523)、ums_ctc.pdparams
3. 运行shell脚本进行COCO_CTC图文检索端到端评测
```
sh eval_scripts/run_eval_ums_base_ctc.sh ums_ctc.pdparams 
```

## 引用
相关文献请引用：
```
@article{cheng2022vista,
  title={ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval},
  author={Cheng, Mengjun and Sun, Yipeng and Wang, Longchao and Zhu, Xiongwei and Yao, Kun and Chen, Jie and Song, Guoli and Han, Junyu and Liu, Jingtuo and Ding, Errui and others},
  journal={arXiv preprint arXiv:2203.16778},
  year={2022}
}
```
