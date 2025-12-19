# ProRank-DecNet

**Early Screening of Pneumoconiosis Using Clinically Aligned Deep Learning with Progressive Risk Ranking and Differentiable Decision Modeling**

## Abstract

Pneumoconiosis remains a highly prevalent occupational lung disease worldwide, with substantial morbidity and mortality, particularly in resource-limited settings. Early detection is critical to improving patient outcomes. However, current deep learning-based pneumoconiosis screening methods have made limited progress in early-stage identification, which stems from the coarse nature of existing annotations and the neglect of the inherent ordinal progression of the disease, both of which hinder the accurate delineation of early lesions. To address these challenges, we constructed the first pneumoconiosis dataset that has fine-grained lesion labels for early-stage screening. The training dataset comprises 4,998 chest radiographs with annotations specifying lesion presence in predefined sub-segments of each lung. Such detailed pathological labels can provide reliable clinical supervision for the construction of screening methods, thereby enabling accurate modeling of early morphological features. Building on this dataset, we propose ProRank-DecNet, a deep learning framework integrating progressive risk ranking, which captures the ordinal progression of disease, and differentiable decision modules that incorporate local-to-global diagnostic rules from clinical guidelines. This design improves early-stage discrimination, ensures prediction consistency across scales, and enhances interpretability. On a validation set of 786 images, our method achieved an AUC of 94.76\%, accuracy of 89.59\% for the primary screening task, outperforming multiple state-of-the-art baselines and reducing the misdiagnosis rate. For sub-regional lesion classification, the model reached an accuracy of 86.83\% and generated localization heatmaps that highlight the precise anatomical position of suspected lesions, providing actionable interpretive support for clinicians. Overall, the proposed framework demonstrates strong potential as a clinically reliable tool for early pneumoconiosis screening.

<p align="center">
<img src="asserts/framework_01.png" width=100% height=100% 
class="center">
</p>

## Environmental setup

```
pip install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

## Data Access

We will provide data access through an application and approval process after the official publication of the paper, and continue to expand and update the dataset in collaboration with medical institutions, to promote standardized collaboration and sustained progress in early-stage pneumoconiosis screening research.
