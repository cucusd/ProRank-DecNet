import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForImageClassification
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from utils.datasets_file import TextFileDataset_sub_lung, TextFileDataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score,confusion_matrix
from DifferentiableAggregation import DifferentiableAggregation_avg, DifferentiableAggregation_test
import sys

# 确保Python路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

num_labels = 2
num_labels_sub = 3

def generate_tensor(n):
    # 创建一个列表，其中每个标签重复6次
    tensor_list = []
    for label in range(n):
        tensor_list.extend([label] * 6)
    # 将列表转换为tensor
    return torch.tensor(tensor_list)

# full_sub_labels
def collate_fn(batch):
    """增强版collate_fn，维护原图-子图对应关系"""
    all_sub_images = []
    all_sub_labels = []
    all_full_sub_labels = []
    sample_ids = []
    original_indices = []
    original_images = torch.stack([item["pixel_values"] for item in batch])
    original_labels = torch.tensor([item["label"] for item in batch])

    for i, item in enumerate(batch):
        num_sub = len(item["sub_pixel_values"])
        all_sub_images.extend(item["sub_pixel_values"])
        all_sub_labels.extend(item["sub_labels"])
        all_full_sub_labels.extend(item["full_sub_labels"])
        original_indices.extend([i] * num_sub)
        sample_ids.append(item["filename"])

    sub_images = torch.stack(all_sub_images) if all_sub_images else torch.empty(0)
    sub_labels = torch.tensor(all_sub_labels) if all_sub_labels else torch.empty(0)
    full_sub_labels = torch.tensor(all_full_sub_labels) if all_full_sub_labels else torch.empty(0)
    original_indices = torch.tensor(original_indices)

    n = len(original_labels)
    full_original_indices = generate_tensor(n)

    return {
        "original_images": original_images,
        "original_labels": original_labels,
        "sub_images": sub_images,
        "sub_labels": sub_labels,
        "full_sub_labels": full_sub_labels,
        "original_indices": original_indices,
        "full_original_indices":full_original_indices,
        "sample_ids": sample_ids
    }

# def calculation(all_labels, all_logits,num_labels):
#     all_targets_onehot = F.one_hot(torch.from_numpy(np.array(all_labels)).long(), num_classes=num_labels)
#     probabilities = torch.softmax(torch.from_numpy(np.array(all_logits)), dim=1).cpu().numpy()
#     predictions = np.argmax(probabilities, axis=1)
#     all_targets = all_targets_onehot.cpu().numpy().argmax(axis=1)
#     accuracy = accuracy_score(all_targets, predictions)
#
#     f1 = f1_score(all_targets, predictions, average='weighted')
#     precision = precision_score(all_targets, predictions, average='weighted')
#     recall = recall_score(all_targets, predictions, average='weighted')
#
#     class_sensitivity = []
#     class_specificity = []
#     auc_scores = []
#     class_accuracies = []
#     for i in range(num_labels):
#         binary_predictions = (predictions == i).astype(int)
#         binary_targets = (all_targets == i).astype(int)
#         preds = binary_predictions[binary_targets == 1]
#         cc = preds.sum()
#         dd = binary_targets.sum()
#         class_accuracy = cc / dd if dd > 0 else 0
#         class_accuracies.append(class_accuracy)
#
#         auc = roc_auc_score(all_targets_onehot.cpu().numpy()[:, i], probabilities[:, i])
#         auc_scores.append(auc)
#         true_positive = ((predictions == i) & (all_targets == i)).sum()
#         true_negative = ((predictions != i) & (all_targets != i)).sum()
#         false_positive = ((predictions == i) & (all_targets != i)).sum()
#         false_negative = ((predictions != i) & (all_targets == i)).sum()
#
#         sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
#         specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
#
#         class_sensitivity.append(sensitivity)
#         class_specificity.append(specificity)
#
#     sub_g_conf_matrix = confusion_matrix(all_labels, predictions, labels=range(num_labels)).tolist()
#     auc_scores = np.asarray(auc_scores)
#     auc_scores = np.mean(auc_scores[~np.isnan(auc_scores)])
#
#     return accuracy, auc_scores, f1, precision, recall, class_sensitivity, class_specificity
def calculation(all_labels, all_logits, num_labels):
    # 输入验证
    all_logits = np.nan_to_num(all_logits, nan=0.0)

    all_targets_onehot = F.one_hot(torch.from_numpy(np.array(all_labels)).long(),
                                   num_classes=num_labels)
    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).cpu().numpy()
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    predictions = np.argmax(probabilities, axis=1)
    all_targets = all_targets_onehot.cpu().numpy().argmax(axis=1)

    # 计算基础指标
    accuracy = accuracy_score(all_targets, predictions)
    f1 = f1_score(all_targets, predictions, average='weighted')
    precision = precision_score(all_targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, predictions, average='weighted')

    # 初始化各类指标
    class_sensitivity = []
    class_specificity = []
    auc_scores = []
    valid_auc_count = 0
    auc_sum = 0

    for i in range(num_labels):
        # 类别特定的统计
        binary_predictions = (predictions == i).astype(int)
        binary_targets = (all_targets == i).astype(int)

        # 处理除零情况
        true_pos = ((predictions == i) & (all_targets == i)).sum()
        true_neg = ((predictions != i) & (all_targets != i)).sum()
        false_pos = ((predictions == i) & (all_targets != i)).sum()
        false_neg = ((predictions != i) & (all_targets == i)).sum()

        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        class_sensitivity.append(sensitivity)
        class_specificity.append(specificity)

        # AUC计算（带保护）
        try:
            if len(np.unique(all_targets_onehot[:, i])) > 1:  # 确保有正负样本
                auc = roc_auc_score(all_targets_onehot[:, i], probabilities[:, i])
                auc_sum += auc
                valid_auc_count += 1
            else:
                auc = np.nan
        except ValueError:
            auc = np.nan
        auc_scores.append(auc)

    # 计算平均AUC（只计算有效的）
    mean_auc = auc_sum / valid_auc_count if valid_auc_count > 0 else np.nan

    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, predictions,
                                   labels=range(num_labels)).tolist()

    return accuracy, mean_auc, f1, precision, recall, class_sensitivity, class_specificity

def evaluate_model(model, eval_dataloader, output_dir, unless_labels):
    model.eval()
    all_preds1 = []
    all_labels1 = []
    all_preds2 = []
    all_labels2 = []
    all_logits_g = []
    all_logits_s = []
    all_sub_g_preds = []
    all_sub_g_labels = []
    all_logits_s_g = []
    accelerator = Accelerator(gradient_accumulation_steps=1)
    ori_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    sub_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    log_messages = []

    progress_bar = tqdm(eval_dataloader, desc="评估中", unit="batch")

    for batch in progress_bar:
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            batch['original_images'] = batch['original_images'].to(device)
            batch['sub_images'] = batch['sub_images'].to(device)

            outputs = model(
                original_images=batch['original_images'],
                sub_images=batch['sub_images'],
            )

            aggregator = DifferentiableAggregation_avg(k=0.2)
            logits3 = aggregator(outputs.logits2, batch['original_indices'],batch['full_sub_labels'],batch['full_original_indices'])

            preds1 = outputs.logits1.argmax(dim=-1)
            all_logits_g.extend(accelerator.gather_for_metrics(outputs.logits1).cpu().numpy())
            all_preds1.extend(accelerator.gather_for_metrics(preds1).cpu().numpy())
            all_labels1.extend(accelerator.gather_for_metrics(batch['original_labels']).cpu().numpy())

            preds2 = outputs.logits2.argmax(dim=-1)
            all_logits_s.extend(accelerator.gather_for_metrics(outputs.logits2).cpu().numpy())
            all_preds2.extend(accelerator.gather_for_metrics(preds2).cpu().numpy())
            all_labels2.extend(accelerator.gather_for_metrics(batch['sub_labels']).cpu().numpy())

            all_logits_s_g.extend(accelerator.gather_for_metrics(logits3).cpu().numpy())
            original_indices = batch['original_indices'].numpy()
            # full_sub_labels
            # sub_g_preds = []
            # for group_id in np.unique(original_indices):
            #     group_mask = (original_indices == group_id)
            #     group_preds = preds2.cpu().numpy()[group_mask]
            #     count = np.sum((group_preds == 1) | (group_preds == 2))
            #     group_size = len(group_preds)
            #     sub_g_preds.append(1 if count > 2 else 0)

            sub_g_labels = []
            for group_id in np.unique(original_indices):
                group_mask = (original_indices == group_id)
                group_labels = batch['sub_labels'].cpu().numpy()[group_mask]

                # 获取组的大小
                group_size = len(group_labels)

                if group_size < 6:
                    # 使用 full_original_indices 获取对应组的索引
                    full_group_mask = (batch['full_original_indices'] == group_id)
                    # 获取该组在 full_sub_labels 中的标签
                    group_labels_full = batch['full_sub_labels'][full_group_mask]
                    # 统计标签 4 的数量
                    # 如果 group_labels 是 PyTorch 张量，转换为 NumPy 数组
                    if not isinstance(group_labels_full, np.ndarray):
                        group_labels_full = group_labels_full.cpu().numpy()
                    count_label_4 = np.sum(group_labels_full == 4)
                else:
                    count_label_4 = 0

                # 统计原始预测中值为 1 或 2 的数量
                count = np.sum((group_labels == 1) | (group_labels == 2))
                # 将标签 4 的数量加入到 count 中
                total_count = count + count_label_4

                # 如果 total_count > 2，则添加 1，否则添加 0
                sub_g_labels.append(1 if total_count > 1 else 0)

            sub_g_preds = []
            for group_id in np.unique(original_indices):
                group_mask = (original_indices == group_id)
                group_preds = preds2.cpu().numpy()[group_mask]

                # 获取组的大小
                group_size = len(group_preds)

                if group_size < 6:
                    # 使用 full_original_indices 获取对应组的索引
                    full_group_mask = (batch['full_original_indices'] == group_id)
                    # 获取该组在 full_sub_labels 中的标签
                    group_labels = batch['full_sub_labels'][full_group_mask]
                    # 统计标签 4 的数量
                    if not isinstance(group_labels, np.ndarray):
                        group_labels = group_labels.cpu().numpy()
                    count_label_4 = np.sum(group_labels == 4)
                else:
                    count_label_4 = 0

                # 统计原始预测中值为 1 或 2 的数量
                count = np.sum((group_preds == 1) | (group_preds == 2))
                # 将标签 4 的数量加入到 count 中
                total_count = count + count_label_4

                # 如果 total_count > 2，则添加 1，否则添加 0
                sub_g_preds.append(1 if total_count > 1 else 0)

            all_sub_g_preds.extend(accelerator.gather_for_metrics(sub_g_preds))
            all_sub_g_labels.extend(accelerator.gather_for_metrics(sub_g_labels))

            for pred, label in zip(preds1.cpu().numpy(), batch['original_labels'].cpu().numpy()):
                ori_class_stats[label]['total'] += 1
                if pred == label:
                    ori_class_stats[label]['correct'] += 1
            for pred, label in zip(preds2.cpu().numpy(), batch['sub_labels'].cpu().numpy()):
                sub_class_stats[label]['total'] += 1
                if pred == label:
                    sub_class_stats[label]['correct'] += 1

            # Log batch-level information for debugging
            log_messages.append({
                "batch_sample_ids": batch['sample_ids'],
                "batch_original_preds": preds1.cpu().numpy().tolist(),
                "batch_original_labels": batch['original_labels'].cpu().numpy().tolist(),
                "batch_sub_preds": preds2.cpu().numpy().tolist(),
                "batch_sub_labels": batch['sub_labels'].cpu().numpy().tolist()
            })

    # 计算混淆矩阵
    ori_conf_matrix = confusion_matrix(all_labels1, all_preds1, labels=range(num_labels)).tolist()
    sub_conf_matrix = confusion_matrix(all_labels2, all_preds2, labels=range(num_labels_sub)).tolist()
    sub_g_conf_matrix = confusion_matrix(all_labels1, all_sub_g_preds, labels=range(num_labels)).tolist()
    sub_g_conf_matrix_1 = confusion_matrix(all_labels1, all_sub_g_labels, labels=range(num_labels)).tolist()

    ori_class_acc = {str(cls): stats['correct'] / stats['total']
                     for cls, stats in sorted(ori_class_stats.items())}
    sub_class_acc = {str(cls): stats['correct'] / stats['total']
                     for cls, stats in sorted(sub_class_stats.items())}
    accuracy1 = (np.array(all_preds1) == np.array(all_labels1)).mean()
    accuracy2 = (np.array(all_preds2) == np.array(all_labels2)).mean()
    sub_g_acc = (np.array(all_sub_g_preds) == np.array(all_labels1)).mean()

    qwk = cohen_kappa_score(
        np.array(all_labels2),
        np.array(all_preds2),
        weights='quadratic'
    )
    accuracy, auc_scores, f1, precision, recall, class_sensitivity, class_specificity = calculation(all_labels2,
                                                                                                    all_logits_s,num_labels_sub)

    # Log sub-image metrics
    log_messages.append({
        "sub_image_metrics": {
            "accuracy": float(accuracy * 100),
            "auc": float(auc_scores * 100),
            "f1_score": float(f1 * 100),
            "precision": float(precision * 100),
            "recall": float(recall * 100),
            "qwk": float(qwk),
            "sub_group_accuracy": float(sub_g_acc * 100),
            "class_sensitivity": [float(s * 100) for s in class_sensitivity],
            "class_specificity": [float(s * 100) for s in class_specificity],
            "ori_conf_matrix": [[float(val) for val in row] for row in ori_conf_matrix],
            "sub_conf_matrix": [[float(val) for val in row] for row in sub_conf_matrix],
            "sub_g_conf_matrix": [[float(val) for val in row] for row in sub_g_conf_matrix],

        }
    })

    # # Log sub-image inference for global metrics
    # accuracy_g, auc_scores_g, f1_g, precision_g, recall_g, class_sensitivity_g, class_specificity_g = calculation(
    #     all_labels1, all_logits_s_g,num_labels)
    # log_messages.append({
    #     "sub_region_inference_global_metrics": {
    #         "accuracy": float(accuracy_g * 100),
    #         "auc": float(auc_scores_g * 100),
    #         "f1_score": float(f1_g * 100),
    #         "precision": float(precision_g * 100),
    #         "recall": float(recall_g * 100),
    #         "class_sensitivity": [float(s * 100) for s in class_sensitivity_g],
    #         "class_specificity": [float(s * 100) for s in class_specificity_g]
    #     }
    # })

    # Log class distribution
    log_messages.append({
        "class_distribution_original": {str(i): int(np.sum(np.array(all_labels1) == i)) for i in range(num_labels)},
        "class_distribution_sub": {str(i): int(np.sum(np.array(all_labels2) == i)) for i in range(num_labels_sub)}
    })

    results = {
        "original_image": {
            "overall_accuracy": float(accuracy1),
            "class_accuracy": [
                {"class": str(cls), "accuracy": float(acc)}
                for cls, acc in sorted(ori_class_acc.items())
            ]
        },
        "sub_image": {
            "overall_accuracy": float(accuracy2),
            "class_accuracy": [
                {"class": str(cls), "accuracy": float(acc)}
                for cls, acc in sorted(sub_class_acc.items())
            ]
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, 'test_log.json'), 'w') as f:
        json.dump(log_messages, f, indent=4, ensure_ascii=False)

    return results

def main(output_dir):
    log_messages = []
    os.makedirs(output_dir, exist_ok=True)
    log_messages.append({"output_dir": f"{output_dir}/test_log.json"})

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        print(item_path)
        # item_path='/data1/jingyi/cjy/pyproject/transformers/results/year_sub_ori/two_module/cls3_diff_20_3_logistic_phi/vit-base/best_model_ori'
        if os.path.isdir(item_path):
            log_messages.append({"model_folder": item_path})

            transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # eval_dataset = TextFileDataset_sub_lung(
            #     txt_file='./transformers/datasets/guizhou_shanxi/test.txt',
            #     img_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all',
            #     sub_lung_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all_six',
            #     label_file='./transformers/datasets/guizhou_shanxi/all_label.txt',
            #     transform=transform, classification_type=3
            # )

            # eval_dataset = TextFileDataset_sub_lung(
            #     txt_file='./transformers/datasets/dataset_splits_year_8ka/test.txt',
            #     img_dir='./transformers/datasets/guizhou_sick_health_2',
            #     sub_lung_dir='./transformers/datasets/guizhou_sub_lung_2_error',
            #     label_file='./transformers/datasets/dataset_splits_year_8ka/sick_health885_1022_label.txt',
            #     transform=transform, classification_type=3
            # )
            #
            eval_dataset = TextFileDataset_sub_lung(
                txt_file='./transformers/datasets/shanxi_dataset_new/shanxi_4746/fold1_test.txt',
                img_dir='./transformers/datasets/shanxi_dataset_new/seg_rec_img_1024',
                sub_lung_dir='./transformers/datasets/shanxi_dataset_new/seg_rec_img_1024_six',
                label_file='./transformers/datasets/shanxi_dataset_new/labels_t.txt',
                transform=transform, classification_type=3
            )

            eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn)
            dir_or_path = item_path
            config = AutoConfig.from_pretrained(
                dir_or_path,
                image_size=512,
                finetuning_task="image-classification",
                trust_remote_code=False,
            )
            model = AutoModelForImageClassification.from_pretrained(
                dir_or_path,
                from_tf=bool(".ckpt" in dir_or_path),
                config=config,
                ignore_mismatched_sizes=False,
                trust_remote_code=False, num_labels1=num_labels, num_labels2=num_labels_sub, use_diff=False
            )

            results = evaluate_model(model, eval_dataloader, item_path, unless_labels=[3, 4])
            log_messages.append({"results": results})

    with open(os.path.join(output_dir, 'test_log.json'), 'w') as f:
        json.dump(log_messages, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    path = './transformers/engine/output/shanxi_train_baseline_old_data_baseline_Stage1x3_mix/test_on_guizhou_shanxi/'
    path_list = os.listdir(path)
    for index, sub_path in enumerate(path_list):
        output_dir = './transformers/engine/output/shanxi_train_baseline_old_data_baseline_Stage1x3_mix/test_on_guizhou_shanxi/'+sub_path
        main(output_dir)

    # output_dir = '/engine/output/shanxi_train_old_data_Stage1x3_BRCD_convnext/ori/convnextv2-tiny_aug1e-05'
    # main(output_dir)
