# ProRank-DecNet

**Early Screening of Pneumoconiosis Using Clinically Aligned Deep Learning with Progressive Risk Ranking and Differentiable Decision Modeling**

## Abstract
Pneumoconiosis remains a highly prevalent occupational lung disease worldwide, with substantial morbidity and mortality, particularly in resource-limited settings. Early detection is critical to improving patient outcomes. However, current deep learning-based pneumoconiosis screening methods have made limited progress in early-stage identification, which stems from the coarse nature of existing annotations and the neglect of the inherent ordinal progression of the disease, both of which hinder the accurate delineation of early lesions. To address these challenges, we constructed the first pneumoconiosis dataset that has fine-grained lesion labels for early-stage screening. The training dataset comprises 4,998 chest radiographs with annotations specifying lesion presence in predefined sub-segments of each lung. Such detailed pathological labels can provide reliable clinical supervision for the construction of screening methods, thereby enabling accurate modeling of early morphological features. Building on this dataset, we propose ProRank-DecNet, a deep learning framework integrating progressive risk ranking, which captures the ordinal progression of disease, and differentiable decision modules that incorporate local-to-global diagnostic rules from clinical guidelines. This design improves early-stage discrimination, ensures prediction consistency across scales, and enhances interpretability. On a validation set of 786 images, our method achieved an AUC of 94.76\%, accuracy of 89.59\% for the primary screening task, outperforming multiple state-of-the-art baselines and reducing the misdiagnosis rate. For sub-regional lesion classification, the model reached an accuracy of 86.83\% and generated localization heatmaps that highlight the precise anatomical position of suspected lesions, providing actionable interpretive support for clinicians. Overall, the proposed framework demonstrates strong potential as a clinically reliable tool for early pneumoconiosis screening.
<p align="center">
<img src="assets/framework_01.png" width=100% height=100% 
class="center">
</p>

## Usage
Please refer to [GET_STARTED.md](GET_STARTED.md) for training and evaluation instructions. 
Our pretrained model can be downloaded from [this link](https://drive.google.com/drive/folders/1054wCA8jcTXEoCLGQjxaLx7vCNHGVZBe?usp=sharing).

## Visualizations
Comparative visualization of Class Activation Mapping (CAM) heatmaps demonstrating lesion localization capabilities. The proposed method (Ours) exhibits precise focus on intrapulmonary sub-regions consistent with pneumoconiosis aerodynamic deposition patterns, whereas the baseline frequently misattends to extrapulmonary artifacts.
<p align="center">
<img src="assets/visualization_01.png" width=90% height=90% 
class="center">
</p>

## Results
Comprehensive performance improvement analysis comparing the proposed ProRank-DecNet with baseline models across various backbone architectures. The bar charts illustrate the net increase in AUC, accuracy, QWK, and class-wise sensitivity, with all numerical values measured in percentages (%).
<p align="center">
<img src="assets/gain_01.png" width=90% height=90% 
class="center">
</p>

## Acknowledgements
This repository is built using the [huggingface](https://github.com/huggingface/transformers/tree/main) library and [Swin](https://github.com/microsoft/Swin-Transformer) repository.

## Contact
If you have any question, feel free to contact the authors.

Le Yang(杨乐):  nwpuyangle@gmail.com

Binglu Wang(王秉路): wbl921129@gmail.com




## Data Access

We will provide data access through an application and approval process after the official publication of the paper, and continue to expand and update the dataset in collaboration with medical institutions, to promote standardized collaboration and sustained progress in early-stage pneumoconiosis screening research.
