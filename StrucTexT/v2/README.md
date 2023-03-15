English | [简体中文](README_ch.md)
## _StrucTexTv2_: Masked Visual-Textual Prediction for Document Image Pre-training

* [Architecture](#architecture)
* [Pre-trained Tasks](#pre-trained-tasks)
* [Downstream Tasks](#downstream-tasks)
* [Benchmarks](#benchmarks)
* [Quick Experience](#quick-experience)
* [Product Applications](#product-applications)
* [Citation](#citation)

## Architecture
The pre-training framework of ***single-modal image input*** and ***multi-modal knowledge learning*** is innovatively proposed on **StrucTexT 2.0** only receives a document image and efficiently captures the semantic structures. The model significantly improves the performance of document understanding tasks (such as document image classification, document layout analysis, table structure analysis, document OCR, end-to-end information extraction, etc.) by pre-trained learning on large-scale document image datasets. It also achieves a better trade-off between performance and efficiency, overcoming limited semantic representations caused by insufficient data and suboptimal optimization strategies in the two-stage pipeline: OCR+NLP. Accounting for cards, bills, invoices, and other documents, StrucTexT 2.0 can be a suitable technique for a wide application of OCR and document understanding.


![architecture](doc/architecture2.png)
<p align="center"> Model Architecture of StrucTexT 2.0 </p>

For basic design concept of the architecture, please see our paper:
>[_**StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training**_](https://arxiv.org/pdf/2303.00289.pdf)
>
>Yuechen Yu\*, Yulin Li\*, Chengquan Zhang<sup>*\+</sup>, Xiaoqiang Zhang, Zengyuan Guo, Xiameng Qin, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang (\*: equal contribution, <sup>\+</sup>: corresponding author)
>
>Accepted by ICLR 2023
>

## Pre-trained Tasks
StrucTexT 2.0 achieves rich multi-modal representations by performing masked visual-textual prediction. The pre-training procedure includes four steps as follows,

1）Collect 11 million English documents from public dataset [IIT-CDIP Test Collection 1.0](https://data.nist.gov/od/id/mds2-2531) and over 100 million Chinese images from the Internet, and use the Baidu OCR toolkit to obtain highly available text content for pre-training.  
2）Randomly Mask 30% word-level text region of the document image and input it to the encoder of StrucTexT 2.0 (a combination structure of CNN and Transformer) for efficient feature coding.  
3）Utilize the position of the masked text regions to extract the RoI features from the encoder output.   
4）Feed the features into the two pre-training task branches of image reconstruction and text reasoning, respectively. The pre-training objectives explore the potential knowledges from large-scale unsupervised data and learn multi-modal semantics.

* **Masked Image Modeling**: MIM on text-region level reconstructs image pixel values of masked text region.
* **Masked Language Modeling**: MLM on text-region level infers the word-level content of masked text region according to document RoI features.


## Downstream Tasks
The pre-trained StrucTexT 2.0 can be regarded as the basic model of downstream tasks, which can be efficiently finetuned with various task-specific heads for document understanding with the corresponding training data, including document image classification, document layout analysis, table structure analysis, document OCR and end-to-end information extraction. These tasks are described as follows:


* **Document image classification**: The document image category can generally be classified as letter, form, email, resume, memo, card, invoice, ans so on, according to the industry property of document.
* **Document layout analysis**: Document layout analysis aims to identify the layout elements (such as title, paragraph, figure, list, table, etc.) of document images by object detection.
* **Table structure analysis**: The objective of this task is to analysis the internal structure of a table which is critical for table understanding.
* **Document OCR**:  Document OCR tends to read the text of document towards end-to-end text spotting.
* **End-to-end information extraction**: The task aims to extract entity-level content of key fields from given documents without a separate OCR pre-processing.

### Datasets
* [RVL-CDIP](https://docs.google.com/u/0/uc?export=download&confirm=9NG1&id=0Bz1dfcnrpXM-MUt4cHNzUEFXcmc) contains 400 thousand scanned document images in 16 classes including 320 thousand training, 40 thousand validation and 40 thousand test images. The images are characterized by low quality, noise, and low resolution, typically 100 dpi. Average classification accuracy is used evaluate model performance.

* [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) is a dataset for document layout analysis, which consists of over 360 thousand images built by automatically parsing PubMed XML files. Five typical document layout elements: text, title, list, figure, and table are annotated.  Mean average precision (mAP) @ intersection over union (IOU) is used as the evaluation metric of document layout analysis.

* [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) is a large dataset for image-based table recognition, containing more than 568 thousand images of tabular data annotated with the corresponding HTML format. Tree-Edit-Distance-based Similarity (TEDs) metric is utilized for table recognition.

* [FUNSD](https://guillaumejaume.github.io/FUNSD) is a form understanding dataset that contains 199 forms (149 for training and 50 for testing), which refers to extract four predefined semantic entities (questions, answers, headers, and other) presented in the form. The two tasks of document OCR and end-to-end information extraction are evaluated with the normalized Levenshtein similarity (1-NED) between the predictions and the ground truth.

* [XFUND](https://github.com/doc-analysis/XFUND) is a multi-lingual extended dataset of the FUNSD in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). We benchmark the performance of our model on the Chinese subset XFUND-ZH.


## Benchmarks

| Downstream Tasks |  Benchmarks | Metrics | Performance |
|  ----  | ---- | ----  | :----: |
| Document image classification   | RVL-CDIP |  Accuray | 93.4 |
| Document layout analysis   | PubLayNet  |   F1-score  |  95.4 |
| Table structure recognition   | PubTabNet  | TEDs | 97.1 |
| Document OCR     | FUNSD  | 1-NED | 84.1 |
| End-to-end information extraction | FUNSD | 1-NED |  55.0 |
| End-to-end information extraction on Chinese | XFUND-ZH | 1-NED |  67.5 |

## Quick Experience
### Install PaddlePaddle
This code base needs to be executed on the `PaddlePaddle 2.2.0+`. You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip:

`pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple`

#### Environment requirements

* python 3.6+
* cuda >= 10.1
* cudnn >= 7.6
* gcc >= 8.2

#### Install requirements
StrucTexT 2.0 dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.

`pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple`

### Download inference models

| Models | Params(M) | Download links | 
| :---- | :---- | :---- |
| MLP Classification| 28.4 | [StrucTexT\_v2 Base for Document Classify](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |
| Cascade RCNN Detection | 50.2 | [StrucTexT\_v2 Base for Layout Analysis](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |
| Transformer Decoder | 128.5 | [StrucTexT\_v2 Base for Table Structext Recognition](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |
| DB Detection + Attention-OCR | 37.3 | [StrucTexT\_v2 Base for End2End OCR](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |
| DB Detection + Attention-OCR + Labeling | 41.3 | [StrucTexT\_v2 Base for End2End Information Extraction](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |
| DB Detection + Labeling | 27.4 | [StrucTexT\_v2 Base for End2End Information Extraction(XFUND-ZH)](https://aistudio.baidu.com/aistudio/datasetdetail/147611) |

### Infer fine-tuned models
   * Document image classification on RVL-CDIP

```python
# 1. Download and extract the RVL-CDIP dataset at `./data/`
# 2. Download the model: StrucTexT_v2_document_classify_base.pdparams
# 3. Run the following script to eval:
python -u ./tools/eval.py \
    --config_file=configs/document_classify/classify_rvlcdip_base.json \
    --task_type=document_classify \
    --label_path=./data/rvl-cdip/test.txt \
    --image_path=./data/rvl-cdip/images \
    --weights_path=StrucTexT_v2_document_classify_base.pdparams
```
   * Document layout analysis on PubLayNet

```python
# 1. enter `./src/tasks/layout_analysis/`
# 2. Download and extract the PubLayNet at `./data/`
# 3. Download the model: StrucTexT_v2_layout_analysis_base.pdparams
# 4. Run the following script to eval:
sh set_env.sh
python -u ./tools/eval.py \
	-c configs/layout_analysis/cascade_rcnn/cascade_rcnn_v2.yml \
	-o weights=StrucTexT_v2_layout_analysis_base.pdparams
```
   * Table structure recognition on PubTabNet

```python
# 1. Download and extract the PubTabNet at `./data/`
# 2. Download the model: StrucTexT_v2_table_recognition_base.pdparams
# 3. Run the following script to eval:
python -u tools/eval.py \
    --config_file=configs/table_recognition/recg_pubtabnet_base.json \
    --task_type=table_recognition \
    --label_path=./data/pubtabnet/PubTabNet_2.0.0_val.jsonl \
    --image_path=./data/pubtabnet/val/ \
    --weights_path=StrucTexT_v2_table_recognition_base.pdparams
```
   * Document OCR on FUNSD

```python
# 1. Download and extract the FUNSD at `./data/`
# 2. Download the model: StrucTexT_v2_end2end_ocr_base.pdparams
# 3. Run the following script to eval:
python -u ./tools/eval.py \
    --config_file=configs/end2end_ocr/ocr_funsd_base.json \
    --task_type=end2end_ocr \
    --label_path=./data/funsd/testing_data/annotation \
    --image_path=./data/funsd/testing_data/image \
    --weights_path=StrucTexT_v2_end2end_ocr_base.pdparams
```
   * End-to-end information extraction on FUNSD

```python
# 1. Download and extract the FUNSD at `./data/`
# 2. Download the model: StrucTexT_v2_end2end_ie_base.pdparams
# 3. Run the following script to eval:
python -u ./tools/eval.py \
    --config_file=configs/end2end_ie/ocr_funsd_base.json \
    --task_type=end2end_ie \
    --label_path=./data/funsd/testing_data/annotation \
    --image_path=./data/funsd/testing_data/image \
    --weights_path=StrucTexT_v2_end2end_ie_base.pdparams
```
   * End-to-end information extraction on XFUND-ZH
      * StrucTexT 2.0 directly adopts the text recognition model from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/applications/%E5%A4%9A%E6%A8%A1%E6%80%81%E8%A1%A8%E5%8D%95%E8%AF%86%E5%88%AB.md) since only 149 train images of XFUND-ZH can not fulfill the recognition branch training for the end-to-end information extraction task. We provide the data and processing scripts for the final evaluation.

```python
# 1. Run the following script to obtain the predictions of text detection and entity classidication
python tools/infer_xfund.py

# 2. Adopt the recognition model from PaddleOCR for text recognition
# 3. Run the following script to eval:
python tools/eval_xfund.py \
    --pred_folder=data/xfund/res/ \
    --gt_file=data/xfund/xfun_normalize_val.json
```


## Product Applications
1. Information extraction can be widely used in business fields such as identity authentication, financial account opening, credit investigation, and merchant settlement. The performance is more than **30%** higher than that of StrucTexT 1.0.
2. Our model enables document electronic in the business office, such as invoices, reports, industry papers, etc. and results in a maximum **50%** reduction of error rate on form understanding and table structural analysis.


![products](doc/products.png)

**StrucTexT 2.0** will participate in product upgrades of the OCR Matrix in [Baidu AI open platform](https://ai.baidu.com/) and [EasyDL-OCR](https://ai.baidu.com/easydl/ocr) soon. For more information and applications, please visit [Baidu OCR](https://ai.baidu.com/tech/ocr).


## Citation
You can cite the related paper as below:
```
@inproceedings{yu2023structextv,
    title={StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training},
    author={Yuechen Yu and Yulin Li and Chengquan Zhang and Xiaoqiang Zhang and Zengyuan Guo and Xiameng Qin and Kun Yao and Junyu Han and Errui Ding and Jingdong Wang},
    booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
    url={https://openreview.net/forum?id=HE_75XY5Ljh}
}
```