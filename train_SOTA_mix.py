# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
from pathlib import Path

import datasets
import evaluate
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import diffsort
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
# import transformers
from torchvision import transforms
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from utils.datasets_file import TextFileDataset, TextFileDataset_sub_lung, TextFileDataset_sub
from datasets import Dataset
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, NLLLoss
import matplotlib.pyplot as plt
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
import random
import os

# # 设置 CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib
import torch.nn.functional as F
from DifferentiableAggregation import DifferentiableAggregation_avg,DifferentiableAggregation_test
matplotlib.use('Agg')
check_min_version("4.41.0.dev0")
from test_acc_1 import main as test_function

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

import logging



def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="img",
        help="The name of the dataset column containing the image data. Defaults to 'image'.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="label",
        help="The name of the dataset column containing the labels. Defaults to 'label'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet",
        help="model name.",
    )

    parser.add_argument(
        "--num_labels",
        type=str,
        default="5",
        help="class num'.",
    )
    parser.add_argument(
        "--num_labels_sub",
        type=str,
        default="5",
        help="class num'.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="5",
        help="sigmoid k'.",
    )
    parser.add_argument(
        "--t",
        type=str,
        default="5",
        help="tem t'.",
    )

    parser.add_argument(
        "--steepness",
        type=str,
        default="30",
        help="steepness'.",
    )
    parser.add_argument(
        "--art_lambda",
        type=str,
        default="0.3",
        help="art_lambda'.",
    )
    parser.add_argument(
        "--a",
        type=str,
        default="0.3",
        help="con'.",
    )
    parser.add_argument(
        "--b",
        type=str,
        default="0.5",
        help="diff'.",
    )
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
    #     raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main(config):
    args = config

    # 创建Logger对象
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)

    # 文件处理器
    os.makedirs(args.output_dir,exist_ok=True)
    file_handler = logging.FileHandler(f'{args.output_dir}/app.log')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    seed_everything(42)
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 多线程中的确定性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA相关设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16",
                              **accelerator_log_kwargs)

    logger.warning(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logger.warning(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and image processor
    if args.model_name == 'resnet50':
        args.model_name_or_path = '../model_download/microsoft/resnet-50'
    if args.model_name == "mobilenet_v2":
        args.model_name_or_path = '../model_download/mobilenetv2'
    if args.model_name == "convnextv2-tiny":
        args.model_name_or_path = '../model_download/convnextv2-tiny-1k-224'
    if args.model_name == "swin":
        args.model_name_or_path = '../model_download/microsoft/swin-base-patch4-window7-224'
    if args.model_name == "deit":
        args.model_name_or_path = '../model_download/Deit'
    if args.model_name == "beit":
        args.model_name_or_path = '../model_download/Beit'
    if args.model_name == "swiftformer-xs":
        args.model_name_or_path = '../model_download/Swiftformer'
    if args.model_name == "vit-base":
        args.model_name_or_path = '../model_download/vit'
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,

        image_size=512,
        finetuning_task="image-classification",
        trust_remote_code=args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    if args.model_name == 'resnet50':
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=args.trust_remote_code, num_labels1=args.num_labels, num_labels2=args.num_labels_sub,
            use_diff=True
        )
        in_features = model.classifier1[1].in_features
        model.classifier1[1] = torch.nn.Linear(in_features, args.num_labels)
    if args.model_name == "swiftformer-xs":
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=args.trust_remote_code, num_labels1=args.num_labels, num_labels2=args.num_labels_sub,
            use_diff=True
        )
        in_features = model.dist_head1.in_features
        model.dist_head1 = torch.nn.Linear(in_features, args.num_labels)

    if args.model_name != 'resnet50' and args.model_name != "swiftformer-xs":
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=args.trust_remote_code, num_labels1=args.num_labels, num_labels2=args.num_labels_sub,
            use_diff=True
        )
        in_features = model.classifier1.in_features
        model.classifier1 = torch.nn.Linear(in_features, args.num_labels)

    # Define torchvision transforms
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )

    # Data loading
    def generate_tensor(n):
        # 创建一个列表，其中每个标签重复6次
        tensor_list = []
        for label in range(n):
            tensor_list.extend([label] * 6)
        # 将列表转换为tensor
        return torch.tensor(tensor_list)

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
            "full_original_indices": full_original_indices,
            "sample_ids": sample_ids
        }
    from torchvision import transforms as pth_transforms
    transform = pth_transforms.Compose([
        ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = TextFileDataset_sub_lung(
        txt_file=args.train_dir,
        img_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all',
        sub_lung_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all_six',
        label_file='./transformers/datasets/guizhou_shanxi/all_label.txt',
        transform=transform, classification_type=args.num_labels_sub
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)

    eval_dataset = TextFileDataset_sub_lung(
        txt_file=args.validation_dir,
        img_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all',
        sub_lung_dir='./transformers/datasets/guizhou_shanxi/guizhou_shanxi_all_six',
        label_file='./transformers/datasets/guizhou_shanxi/all_label.txt',
        transform=transform, classification_type=args.num_labels_sub
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_train_batch_size, collate_fn=collate_fn)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # Initialize metrics tracking
    train_losses1 = []
    train_losses2 = []
    val_losses1 = []
    val_losses2 = []
    val_accuracies_ori = []
    val_accuracies_sub = []
    # metric = evaluate.load("/data1/jingyi/cjy/pyproject/transformers/metrics/accuracy")

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0

    # 初始化最佳准确率
    best_accuracy = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)

    best_accuracy_sub = 0.0
    best_model_sub_path = os.path.join(args.output_dir, "best_model_sub")
    os.makedirs(best_model_sub_path, exist_ok=True)

    best_accuracy_ori = 0.0
    best_model_ori_path = os.path.join(args.output_dir, "best_model_ori")
    os.makedirs(best_model_ori_path, exist_ok=True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        epoch_train_loss1 = 0.0
        epoch_train_loss2 = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    original_images=batch['original_images'],
                    sub_images=batch['sub_images'],
                )
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(outputs.logits1.view(-1, args.num_labels), batch['original_labels'].view(-1))
                loss2 = loss_fct(outputs.logits2.view(-1, args.num_labels_sub), batch['sub_labels'].view(-1))

                aggregator = DifferentiableAggregation_avg(k=args.k)
                logits3 = aggregator(outputs.logits2, batch['original_indices'], batch['full_sub_labels'], batch['full_original_indices'])

                T =args.t
                soft_targets = F.softmax(outputs.logits1 / T, dim=-1)
                loss_kl = F.kl_div(F.log_softmax(logits3 / T, dim=-1), soft_targets,reduction='batchmean')

                sorter = diffsort.DiffSortNet(
                    sorting_network_type='bitonic',  # sort method
                    # size=batch['labels'].size(0), #num_compare
                    size=outputs.logits_diff.shape[1],
                    device='cuda',  # device
                    steepness=args.steepness,  # steepness
                    art_lambda=args.art_lambda,  # art_lambda
                    distribution='logistic_phi'
                )
                _, perm_prediction = sorter(outputs.logits_diff)
                # targets=batch['labels']
                targets = batch['sub_labels']
                device = outputs.logits2.device  # 选择CUDA设备0
                targets = targets.to(device)
                perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(targets, dim=-1, stable=True)).transpose(
                    -2, -1).float()
                perm_ground_truth = perm_ground_truth.unsqueeze(0)

                loss_diff = torch.nn.BCELoss()(perm_prediction, perm_ground_truth)
                loss = loss1 +  loss2 + args.a * loss_kl + args.b * loss_diff

                epoch_train_loss1 += loss1.item()
                epoch_train_loss2 += loss2.item()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # 计算平均训练损失
        avg_train_loss1 = epoch_train_loss1 / len(train_dataloader)
        avg_train_loss2 = epoch_train_loss2 / len(train_dataloader)
        train_losses1.append(avg_train_loss1)
        train_losses2.append(avg_train_loss2)

        # 验证阶段
        model.eval()
        val_loss1 = 0.0
        val_loss2 = 0.0
        all_preds1 = []
        all_labels1 = []
        all_preds2 = []
        all_labels2 = []

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(
                    original_images=batch['original_images'],
                    sub_images=batch['sub_images'],
                )
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(outputs.logits1.view(-1, args.num_labels), batch['original_labels'].view(-1))
                loss2 = loss_fct(outputs.logits2.view(-1, args.num_labels_sub), batch['sub_labels'].view(-1))

                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

                preds1 = outputs.logits1.argmax(dim=-1)
                all_preds1.extend(accelerator.gather_for_metrics(preds1).cpu().numpy())
                all_labels1.extend(accelerator.gather_for_metrics(batch['original_labels']).cpu().numpy())

                preds2 = outputs.logits2.argmax(dim=-1)
                all_preds2.extend(accelerator.gather_for_metrics(preds2).cpu().numpy())
                all_labels2.extend(accelerator.gather_for_metrics(batch['sub_labels']).cpu().numpy())

        # 计算验证指标
        avg_val_loss1 = val_loss1 / len(eval_dataloader)
        avg_val_loss2 = val_loss2 / len(eval_dataloader)
        val_losses1.append(avg_val_loss1)
        val_losses2.append(avg_val_loss2)

        accuracy1 = (np.array(all_preds1) == np.array(all_labels1)).mean()
        val_accuracies_ori.append(accuracy1)

        accuracy2 = (np.array(all_preds2) == np.array(all_labels2)).mean()
        val_accuracies_sub.append(accuracy2)

        # 保存最佳子任务模型
        if accuracy2 > best_accuracy_sub:
            best_accuracy_sub = accuracy2
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    best_model_sub_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                image_processor.save_pretrained(best_model_sub_path)
                logger.warning(f"New best SUB model saved with accuracy: {best_accuracy_sub:.4f}")

        # 保存最佳主任务模型
        if accuracy1 > best_accuracy_ori:
            best_accuracy_ori = accuracy1
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    best_model_ori_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                image_processor.save_pretrained(best_model_ori_path)
                logger.warning(f"New best ORI model saved with accuracy: {best_accuracy_ori:.4f}")

        # 保存最佳复合模型
        composite_score =accuracy1 +  accuracy2
        if composite_score > best_accuracy:
            best_accuracy = composite_score
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    best_model_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                image_processor.save_pretrained(best_model_path)
                logger.warning(f"New best COMPOSITE model saved with score: {best_accuracy:.4f}")

        # 保存每个epoch的模型
        if accelerator.is_main_process:
            epoch_model_path = f"{args.output_dir}/epoch_{epoch}"  # 每个epoch的模型保存路径
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                epoch_model_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            image_processor.save_pretrained(epoch_model_path)
            logger.warning(f"Model saved for epoch {epoch} at {epoch_model_path}")

        logger.warning(
            f"Epoch {epoch}: "
            f"Train Loss = [{avg_train_loss1:.4f}/{avg_train_loss2:.4f}], "
            f"Val Loss = [{avg_val_loss1:.4f}/{avg_val_loss2:.4f}], "
            f"Val Acc ORI = {accuracy1:.4f}, "
            f"Val Acc SUB = {accuracy2:.4f}"
        )

        # 保存指标可视化
        if accelerator.is_main_process:
            plt.figure(figsize=(16, 6))

            # 损失曲线（双Y轴）
            plt.subplot(1, 2, 1)
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(train_losses1, 'b-', label='ORI Train Loss')
            ax1.plot(val_losses1, 'b--', label='ORI Val Loss')
            ax1.set_ylabel('ORI Loss', color='b')
            ax1.tick_params(axis='y', colors='b')

            ax2.plot(train_losses2, 'r-', label='SUB Train Loss')
            ax2.plot(val_losses2, 'r--', label='SUB Val Loss')
            ax2.set_ylabel('SUB Loss', color='r')
            ax2.tick_params(axis='y', colors='r')

            ax1.set_xlabel('Epoch')
            ax1.set_title('Dual-Task Loss Curves')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

            # 准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies_ori, 'g-', label='ORI Accuracy')
            plt.plot(val_accuracies_sub, 'm-', label='SUB Accuracy')

            # 标记最佳点
            best_acc1 = max(val_accuracies_ori)
            best_epoch1 = val_accuracies_ori.index(best_acc1)
            plt.scatter(best_epoch1, best_acc1, c='g', s=100, alpha=0.5)

            best_acc2 = max(val_accuracies_sub)
            best_epoch2 = val_accuracies_sub.index(best_acc2)
            plt.scatter(best_epoch2, best_acc2, c='m', s=100, alpha=0.5)

            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'multi_task_metrics.png'), dpi=300)
            plt.close()

            # 保存详细指标到CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(train_losses1) + 1),
                'train_loss_ori': train_losses1,
                'train_loss_sub': train_losses2,
                'val_loss_ori': val_losses1,
                'val_loss_sub': val_losses2,
                'val_acc_ori': val_accuracies_ori,
                'val_acc_sub': val_accuracies_sub,
                'best_acc_ori': [max(val_accuracies_ori[:i + 1]) for i in range(len(val_accuracies_ori))],
                'best_acc_sub': [max(val_accuracies_sub[:i + 1]) for i in range(len(val_accuracies_sub))],
                'composite_score': [0.3 * val_accuracies_ori[i] + 0.7 * val_accuracies_sub[i] for i in
                                    range(len(val_accuracies_ori))]
            })
            metrics_df.to_csv(os.path.join(args.output_dir, 'multi_task_metrics.csv'), index=False)

    # Final saving
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            # 保存图像处理器
            image_processor.save_pretrained(args.output_dir)

            # 保存所有结果指标
            final_results = {
                "final_ori_accuracy": val_accuracies_ori[-1],
                "best_ori_accuracy": max(val_accuracies_ori),
                "best_ori_epoch": val_accuracies_ori.index(max(val_accuracies_ori)),
                "final_sub_accuracy": val_accuracies_sub[-1],
                "best_sub_accuracy": max(val_accuracies_sub),
                "best_sub_epoch": val_accuracies_sub.index(max(val_accuracies_sub)),
                "final_composite_score": 0.3 * val_accuracies_ori[-1] + 0.7 * val_accuracies_sub[-1],
                "best_composite_score": max(
                    [0.3 * ori + 0.7 * sub for ori, sub in zip(val_accuracies_ori, val_accuracies_sub)]
                ),
                "final_train_loss_ori": train_losses1[-1],
                "final_train_loss_sub": train_losses2[-1],
                "final_val_loss_ori": val_losses1[-1],
                "final_val_loss_sub": val_losses2[-1]
            }

            # 保存到JSON文件
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(final_results, f, indent=4)

            # 同时保存到CSV便于分析（追加模式）
            results_df = pd.DataFrame([final_results])
            results_df.to_csv(os.path.join(args.output_dir, "final_results.csv"), index=False)

if __name__ == '__main__':
    import gc
    import pandas as pd
    lrs = [1e-5]
    batch_sizes = 2
    num_labels = 2
    num_labels_sub = 3
    # model_name = ['vit-base', 'resnet50', 'mobilenet_v2', 'convnextv2-tiny', 'swiftformer-xs', 'swin', 'deit', 'beit']
    model_name = ['swin', 'deit', 'beit']
    # model_name = ['deit']
    # model_name = ['vit-base']
    # for kk in range(len(lrs)):
    for jj in range(len(model_name)):
        # lr = lrs[kk]
        args = parse_args()
        # a = 0.25
        # b = 0.45
        # for b in [0.5,0.6,0.45,0.3,0.7]:
        a = 0.3
        b = 0.25
        # for a in [0.3]:
        args.k = 3
        args.t = 10
        args.steepness = 30  # steepness
        args.art_lambda = 0.3  # art_lambda
        args.a = a
        args.b = b
        args.unless_labels = [4]
        args.num_labels = num_labels
        args.num_labels_sub = num_labels_sub
        args.train_dir = './transformers/datasets/guizhou_shanxi/train_shanxiStage1x3.txt'
        args.validation_dir = './transformers/datasets/guizhou_shanxi/val.txt'
        args.model_name = model_name[jj]
        args.per_device_train_batch_size = batch_sizes
        args.learning_rate = lrs[0]
        args.num_train_epochs = 20
        # args.output_dir = model_name
        # if model_name == 'resnet50':
        args.output_dir = f'output/shanxi_guizhou_SOTA_mix_shanxiStage1x3/{args.model_name}'+'_aug'
        main(args)
        test_function(args.output_dir)
