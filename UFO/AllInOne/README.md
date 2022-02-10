English | [简体中文](README_ch.md)

# AllInOne

- [Introduction](#Introduction)
- [Datasets](#Datasets)
  * [TrainingSet](#TrainingSet)
  * [TestSet](#TestSet)
- [Methodology](#Methodology)
  * [Framework](#Framework)
  * [UnifiedSettings](#UnifiedSettings)
  * [HeterogeneousBatchVsisomorphismBatch](#HeterogeneousBatchVsisomorphismBatch)
  * [TaskOverfitting](#TaskOverfitting)
- [ComparedWithSOTA](#ComparedWithSOTA)

## Introduction
The starting point of UFO's technical vision is the unification of vision, that is, a model can cover all mainstream visual tasks. Starting from vertical applications, we selected four tasks of Face, Person, Vehicle, and Products as the first step to unify the visual model. The goal of AllInOne is to achieve SOTA results for one model over four tasks.
## Datasets
We used the public datasets of Face, Person, Vehicle, Products as follows:
### TrainingSet

| **Tasks**                     | **Datesets**                   |       **Img Number**           |       **ID Number**            |
| :-----------------------------| :----------------------------: | :----------------------------: | :----------------------------: |
| Face                          |           MS1M-V3              |           5,179,510            |           93,431               |
| Person                        |           Market1501-Train     |           12,936               |           751                  |
| Person                        |           DukeMTMC-Train       |           16,522               |           702                  |
| Person                        |           MSMT17-Train         |           30,248               |           1,041                |
| Vehicle                       |           Veri-776-Train       |           37,778               |           576                  |
| Vehicle                       |           VehicleID-Train      |           113,346              |           13,164               |
| Vehicle                       |           VeriWild-Train       |           277,797              |           30,671               |
| Products                      |           SOP-Train            |           59,551               |           11,318               |


### TestSet

| **Tasks**                     | **Datesets**                   |       **Img Number**           |       **ID Number**            |
| :-----------------------------| :----------------------------: | :----------------------------: | :----------------------------: |
| Face                          |           LFW                  |           12,000               |           -                    |
| Face                          |           CPLFW                |           12,000               |           -                    |
| Face                          |           CFP-FF               |           14,000               |           -                    |
| Face                          |           CFP-FP               |           14,000               |           -                    |
| Face                          |           CALFW                |           12,000               |           -                    |
| Face                          |           AGEDB-30             |           12,000               |           -                    |
| Person                        |           Market1501-Test      |           19,281               |           750                  |
| Person                        |           DukeMTMC-Test        |           19,889               |           702                  |
| Person                        |           MSMT17-Test          |           93,820               |           3,060                |
| Vehicle                       |           Veri-776-Test        |           13,257               |           200                  |
| Vehicle                       |           VehicleID-Test       |           19,777               |           2,400                |
| Vehicle                       |           VeriWild-Test        |           138,517              |           10,000               |
| Products                      |           SOP-Test             |           60,502               |           11,316               |

## Methodology

### Framework
Given 4 tasks, we sample some data from each task to form a batch, input the batch to the shared Transformer backbone network, and finally separate 4 head networks, each of which is responsible for the output of a task. The 4 tasks separately calculate the loss and sum it up as the total loss. Model optimization using the SGD optimizer.

### UnifiedSettings

Because the input size and model structure used by different tasks are quite different. From the model optimization level, the batch size, learning rate and even the optimizer are all different. In order to facilitate subsequent multi-task training, we first unify the model structure and optimization method of each task. In particular, we use Transformer as the backbone network. The unified configuration is shown in the following table:
|                               |      **Face**/ **Person**/**Products**/**Vehicle**   |
| :-----------------------------| :----------------------------------------------------|
| Input Size                    |    384 × 384                                         |
| Batch Size                    |    1024/512/512/512                                  |
| Augmentation                  |    Flipping + Random Erasing + AutoAug               |
| Model                         |    ViT-Large                                         |
| Feature Dim                   |    1024                                              |
| Loss                          |    CosFace Loss/(CosFace Loss + Triplet Loss)*3      |
| Optimizer                     |    SGD                                               |
| Init LR                       |    0.2                                               |
| LR scheduler                  |    Warmup + Cosine LR                                |
| Iterations                    |    100,000                                           |

### HeterogeneousBatchVsisomorphismBatch 

The first problem faced by multi-task learning is how to build a Batch. There are two commonly used methods, one is composed of isomorphic batches, that is, the data in the batch comes from the same task, and different tasks are selected through different batches to ensure that all tasks are trained. The other is heterogeneous batch composition, that is, the data in the batch comes from different tasks.

The problem with isomorphic Batch composition is that when the common operation of Batch Norm is used in the model, because the statistical value during training (single-task statistical value) and the statistical value during testing (multi-task statistical value) are quite different, lead to poor model performance. We use the ResNet50 structure to verify this conclusion in two tasks, Person Market1501 and ProductsSOP. As shown in the table, using heterogeneous batch mixing can greatly improve the performance of the two tasks. Therefore, we adopt heterogeneous Batch.

|    Batch Type        |         Market1501 (rank1/mAP)    |        SOP (rank1)        |
| :--------------------| :--------------------------------:|:-------------------------:|
|  Heterogeneous       |           73.13 / 50.58           |          79.54            |
|  isomorphis          |           94.27 / 85.77           |          85.76            |

### TaskOverfitting

Among our four tasks, Person and Products have the smallest training sets with only about 60,000 images, while Face and Vehicle have about 5 million and 400,000 images respectively. Therefore, in the multi-task training process, Person and Products are rapidly over-fitted, while Face and Vehicle are under-fitted.
The phenomenon. To this end, we have explored many different methods, the most effective of which is to use the Drop Path regularization method. As shown in the table, after increasing the drop path rate from 0.1 to 0.2, the Person and Products tasks have a large improvement, while other tasks have the same or better performance.

|        Model     | DropPath |  CALFW | CPLFW  |  LFW  | CFP-FF | CFP-FP | AGEDB-30 | Market1501  | DukeMTMC    | MSMT17      |   Veri776   |  VehicleID  |  VeriWild   |  SOP  |
| :----------------|----------|--------| :------|-------|--------|--------|---------:|:------------|-------------|-------------|-------------|-------------|-------------|------:|
|  UFO (ViT-Large) | 0.1      |  96.18 | 94.22  | 99.83 |  99.90 |  99.09 |   98.17  | 96.17/91.67 | 92.01/84.63 | 86.21/68.94 | 97.62/88.66 | 85.35/90.09 | 93.31/77.98 | 87.11 |
|  UFO (ViT-Large) | 0.2      |  95.92 | 94.30  | 99.82 |  99.90 |  99.11 |   98.03  | 96.28/92.75 | 92.55/86.19 | 88.10/72.17 | 97.74/89.25 | 87.62/91.32 | 93.62/78.91 | 89.23 |

## ComparedWithSOTA

|        Model     |  CALFW | CPLFW  |  LFW  | CFP-FF | CFP-FP | AGEDB-30 | Market1501  | DukeMTMC    | MSMT17      |   Veri776   |  VehicleID  |  VeriWild   |  SOP  |
| :----------------|--------| :------|-------|--------|--------|---------:|:------------|-------------|-------------|-------------|-------------|-------------|------:|
|  SOTA w/o rerank |  96.20 | 93.37  | 99.85 |  99.89 |  98.99 |   98.35  | 96.3/91.5   | 92.1/83.7   | 86.2/69.4   | 97.0/87.1   | 80.3/86.4   | 92.5/77.3   | 85.9 |
|  UFO (ViT-Large) |  95.92 | 94.30  | 99.82 |  99.90 |  99.11 |   98.03  | 96.28/92.75 | 92.55/86.19 | 88.10/72.17 | 97.74/89.25 | 87.62/91.32 | 93.62/78.91 | 89.23 |


