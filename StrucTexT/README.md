English | [简体中文](README_ch.md)

## _StrucTexT_: Structured Text Understanding with Multi-Modal Transformers

- [Architecture](#architecture)
- [Pre-trained Tasks](#pre-trained-tasks)
- [Structured Text Understanding](#structured-text-understanding)
  * [Token-based Entity Labeling](#token-based-entity-labeling)
  * [Segment-based Entity Labeling](#segment-based-entity-labeling)
  * [Segment-based Entity Linking](#segment-based-entity-linking)
- [Quick Experience](#quick-experience)
  * [Install PaddlePaddle](#install-paddlepaddle)
  * [Download inference models](#download-inference-models)
  * [Infer Fine-tuned Models](#infer-fine-tuned-models)
  * [Product Applications](#product-applications)
- [Citation](#citation)


## Architecture
![structext_arch](./doc/structext_arch.png#pic_center)
<p align="center"> Model Architecture of StrucTexT </p>

For basic design concept of the architecture, please see our paper:
>[_**StrucTexT: Structured Text Understanding with Multi-Modal Transformers**_](https://arxiv.org/abs/2108.02923)
>
>Yulin Li\*, Yuxi Qian\*, Yuechen Yu\*, Xiameng Qin, Chengquan Zhang<sup>\+</sup>, Yan Liu, Kun Yao, Junyu Han, Jingtuo Liu and Errui Ding (\*: equal contribution, <sup>\+</sup>: corresponding author)
>
>Accepted by ACM Multimedia 2021
>

StrucTexT is a joint segment-level and token-level representation enhancement model for document image understanding, such as pdf, invoice, receipt and so on. Due to the complexity of content and layout in visually rich documents (VRDs), structured text understanding has been a challenging task. Most existing studies decoupled this problem into two sub-tasks: entity labeling and entity linking, which require an entire understanding of the context of documents at both token and segment levels. StrucTexT is a unified framework, which can flexibly and effectively deal with the above subtasks in their required representation granularity. Besides, we design a novel pre-training strategy with several self-supervised tasks to learn a richer representation.

Focusing on the business document images encountered in most domestic scenarios, our models published this time have made the following adjustments comapred with the paper version:
- The pre-training tasks are carried out on the largest Chinese and English document image data in the community, in which the amount of whole image data reaches 50 million.
- In order to strength the segment-level semantic understanding, the Masked Sentence Predcition (MSP) is added.
- We have expanded a new large dictionary by merging the chinese and english ones that provied by [ERNIE](https://github.com/PaddlePaddle/ERNIE), to better support Chinese and English document images in a unified model.
- The large parameter setting of StrucTexT model is introduced in order to make a fair comparison with other methods.


## Pre-trained Tasks
- **Masked Visual Language Modeling:** We randomly select a set of tokens, then mask and ask the model to reconstruct them for learning the contextual information.
- **Segment Length Prediction:** For the alignment from vision to language, we force the model to predict the length of each text segment.
- **Paired Box Direction:** It learns the pairwise direction between any two segments in the visual side to capture a geometric topology.

*Recent updates compared to the paper*
- **Masked Sentence Predcition:** Beside MVLM, we append the MSP pre-training task that mask all the sentence tokens of selected text segments to learn comprehensive semantics.

## Structured Text Understanding
We funtune StrucTexT on three structured text understanding tasks, i.e, Token-based Entity Labeling (**T-ELB**), Segment-based Entity Labeling (**S-ELB**), and Entity Linking (**S-ELK**).

Note: Below performance of all **ELB** tasks are the **entity-level F1 score** that calculate Macro F1 scores among related entity types.
### Token-based Entity Labeling
   * datasets.
      * [EPHOIE](https://github.com/HCIILAB/EPHOIE) is collected from scanned Chinese examination papers.
      * There are 10 text types, which are marked at the character level, where means a text segment composed of characters with different categories.
      * The required token-based entity types are as follow: ``` Subject, Test Time, Name, School, Examination Number, Seat Number, Class, Student Number, Grade, and Score. ```
   * performance: Results of **T-ELB** task on EPHOIE for different settings of StrucTexT models

		| Models                         | **entity-level F1 score**      |
		| :----------------------------- | :----------------------------: |
		| StrucTexT-_eng base (paper)_   |                0.9795          |
		| StrucTexT-_chn&eng base_       |                0.9884          |
		| StrucTexT-_chn&eng large_      |                0.9930          |


### Segment-based Entity Labeling
   * datasets.
      * [SROIE](https://rrc.cvc.uab.es/?ch=13&com=introduction) is a public dateset for receipt information extraction in ICDAR 2019 Chanllenge. It contains of 626 receipts for training and 347 receipts for testing. Every receipt contains four predefined values: `company, date, address, and total`.
      * [FUNSD](https://guillaumejaume.github.io/FUNSD/) is a form understanding benchmark with 199 real, fully annotated, scanned form images, such as marketing, advertising, and scientific reports, which is split into 149 training samples and 50 testing samples. FUNSD dataset is suitable for a variety of tasks and we force on the sub-tasks of **T-ELB** and **S-ELK** for the semantic entity `question, answer and header`.
      * [XFUND](https://github.com/doc-analysis/XFUND)  is a multilingual form understanding benchmark dataset that includes human-labeled forms with key-value pairs in 7 languages. Each language includes 199 forms, where the training set includes 149 forms, and the test set includes 50 forms. We evaluate on the Chinses section of XFUND.
      * All the three dataset that we use the official OCR annotations and evaluate our model for information extraction.
   * performance: Results of **S-ELB** task on SROIE, FUNSD and XFUND for different settings of StrucTexT models.
   
		| Models                         | **SROIE**                      | **FUNSD**                      | **XFUND-ZH**                   |
		| :----------------------------- | :----------------------------: | :----------------------------: | :----------------------------: | 
		| StrucTexT-_eng base (paper)_   |           0.9688               |           0.8309               |             -                  |
		| StrucTexT-_chn&eng base_       |           0.9827               |           0.8483               |           0.9101               |
		| StrucTexT-_chn&eng large_      |           0.9870               |           0.8756               |           0.9229               |

### Segment-based Entity Linking
   Entity linking is the task of predicting the relations between semantic entities.
   * datasets
      * [FUNSD](https://guillaumejaume.github.io/FUNSD) annotates the links formatted as `(entity_from, entity_to)`, resulting in a question–answer pair. It guides the task of predicting the relations between semantic entities.
      * [XFUND](https://github.com/doc-analysis/XFUND) is the same setting as FUNSD. We evaluate on the Chinses section of the dataset.
   * performance: Results of **S-ELK** task on FUNSD and XFUND-ZH for different settings of StrucTexT models. Reference performance is the **F1 score** among possible pairs of given semantic entities.

		| Models                         | **FUNSD**                      | **XFUND-ZH**                   |
		| :----------------------------- | :----------------------------: | :----------------------------: |
		| StrucTexT-_eng base (paper)_   |           0.4410               |              -                 |
		| StrucTexT-_chn&eng base_       |           0.7045               |            0.8306              |
		| StrucTexT-_chn&eng large_      |           0.7421               |            0.8681              |


## Quick Experience
### Install PaddlePaddle
This code base needs to be executed on the `PaddlePaddle 2.1.0+`. You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

* Environment requirements

```bash
python 3.6+
opencv-python 4.2.0+
tqdm
tabulate
cuda >= 10.1
cudnn >= 7.6.4
gcc >= 8.2
```

* Install requirements

StrucTexT dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.

```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```


### Download inference models

| Model link                                         |  Params(M)  |
| :------------------------------------------------- | :---------: |
| <a href="https://aisee.bj.bcebos.com/VIMER/StrucTexT/StrucTexT_base_ephoie_labeling.pdparams.tar.gz?authorization=bce-auth-v1%2Fdb0b41e2ab894ecfb1126a768c603d79%2F2021-12-02T08%3A49%3A14Z%2F-1%2Fhost%2F6f75cab944e45627d5f1f630377f4b016dd94104322136bc060912273b852d52" target="_blank">StrucTexT Base for EPHOIE labeling</a>   | 181 |
| <a href="https://aisee.bj.bcebos.com/VIMER/StrucTexT/StrucTexT_large_ephoie_labeling.pdparams.tar.gz?authorization=bce-auth-v1%2Fdb0b41e2ab894ecfb1126a768c603d79%2F2021-12-02T08%3A50%3A36Z%2F-1%2Fhost%2Fa4b284ef849b5ada868403cc8626d1d89d4ca3b5b89b40fc3e0b8e7ffc251753" target="_blank">StrucTexT Large for EPHOIE labeling</a> | 458 |
| <a href="https://aisee.bj.bcebos.com/VIMER/StrucTexT/StrucTexT_base_funsd_labeling.pdparams.tar.gz?authorization=bce-auth-v1%2Fdb0b41e2ab894ecfb1126a768c603d79%2F2021-12-02T08%3A49%3A49Z%2F-1%2Fhost%2Fc3b3648f106aaaf1c73c7876c2012fa55c74016325f0352892323165c5b3a16c" target="_blank">StrucTexT Base for FUNSD labeling</a>     | 181 |
| <a href="https://aisee.bj.bcebos.com/VIMER/StrucTexT/StrucTexT_base_funsd_linking.pdparams.tar.gz?authorization=bce-auth-v1%2Fdb0b41e2ab894ecfb1126a768c603d79%2F2021-12-02T08%3A50%3A16Z%2F-1%2Fhost%2F50aee9546d618b296ec1dfc54084e497ee4836f7034dbe6e2131c26a4003f870" target="_blank">StrucTexT Base for FUNSD linking</a>       | 181 |



### Infer fine-tuned models
   * Token-based ELB task on EPHOIE

```python
# 1. download and extract the EPHOIE dataset at <ephoie_folder>.
# 2. download the model: StrucTexT_(base/large)_ephoie_labeling.pdparams
# 3. generate the eval dataset form.
python data/make_ephoie_data.py \
    --config_file ./configs/(base/large)/labeling_ephoie.json \
    --label_file examples/ephoie/test_list.txt \
    --label_dir <ephoie_folder>/final_release_image_20201222 \
    --kvpair_dir <ephoie_folder>/final_release_kvpair_20201222 \
    --out_dir <ephoie_folder>/test_labels/

# 4. evaluate the labeling task in the EPHOIE dataset.
python ./tools/eval_infer.py \
    --config_file ./configs/(base/large)/labeling_ephoie.json \
    --task_type labeling_token \
    --label_path <ephoie_folder>/test_labels/ \
    --image_path <ephoie_folder>/final_release_image_20201222/ \
    --weights_path StrucTexT_(base/large)_ephoie_labeling.pdparams
```
   * Segment-based ELB task on FUNSD

```python
# 1. download and extract the FUNSD dataset at <funsd_folder>.
# 2. download the model: StrucTexT_base_funsd_labeling.pdparams
# 3. generate the eval dataset form.
python data/make_funsd_data.py \
    --config_file ./configs/base/labeling_funsd.json \
    --label_dir <funsd_folder>/dataset/testing_data/annotations/ \
    --out_dir <funsd_folder>/dataset/testing_data/test_labels/

# 4. evaluate the labeling task in the FUNSD dataset.
python ./tools/eval_infer.py \
    --config_file ./configs/base/labeling_funsd.json \
    --task_type labeling_segment \
    --label_path <funsd_folder>/dataset/testing_data/test_labels/ \
    --image_path <funsd_folder>/dataset/testing_data/images/ \
    --weights_path StrucTexT_base_funsd_labeling.pdparams
```
   * Segment-based ELK task on FUNSD

```python
# 1. download and extract the FUNSD dataset at <funsd_folder>.
# 2. download the model: StrucTexT_base_funsd_linking.pdparams
# 3. generate the eval dataset form.
python data/make_funsd_data.py \
    --config_file ./configs/base/linking_funsd.json \
    --label_dir <funsd_folder>/dataset/testing_data/annotations/ \
    --out_dir <funsd_folder>/dataset/testing_data/test_labels/

# 4. evaluate the linking task in the FUNSD dataset.
python ./tools/eval_infer.py \
    --config_file ./configs/base/linking_funsd.json \
    --task_type linking \
    --label_path <funsd_folder>/dataset/testing_data/test_labels/ \
    --image_path <funsd_folder>/dataset/testing_data/images/ \
    --weights_path StrucTexT_base_funsd_linking.pdparams
```

### Product Applications
The following visualized images are sampled from the domestic practical applications of StrucTexT. *Different colors of masks represent different entity categories. There are black lines between different segments or text lines, indicating that they belong to the same entity. And the orange lines indicate the link relationship between entities.*
- Shopping receipts
<div align="center">
    <img src="./doc/receipt_vis.png" width="800">
</div>
- Bus/Ship receipts
<div align="center">
    <img src="./doc/busticket_vis.png" width="800">
</div>
- Printed receipts
<div align="center">
    <img src="./doc/print_vis.png" width="800">
</div>

For more information and applications, please visit [Baidu OCR](https://ai.baidu.com/tech/ocr) open platform.

## Citation
You can cite the related paper as below:
```
@inproceedings{li2021structext,
  title={StrucTexT: Structured Text Understanding with Multi-Modal Transformers},
  author={Li, Yulin and Qian, Yuxi and Yu, Yuechen and Qin, Xiameng and Zhang, Chengquan and Liu, Yan and Yao, Kun and Han, Junyu and Liu, Jingtuo and Ding, Errui},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1912--1920},
  year={2021}
}
```
