
from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class TextFileDataset(Dataset):
    def __init__(self, txt_file, img_dir, transform=None):
        """
        Args:
            txt_file (str): 包含图像文件名的txt文件路径
            img_dir (str): 图像文件夹路径
            transform (callable, optional): 图像变换操作
        """
        self.img_dir = img_dir
        # 确保至少包含ToTensor转换
        self.transform = transform if transform else transforms.ToTensor()

        with open(txt_file, 'r') as f:
            self.img_names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        """处理单个样本"""
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 确保图像加载为RGB格式
        image = Image.open(img_path).convert('RGB')

        # 从文件名提取标签
        label_str = img_name.split('_')[0]
        label = 1 if label_str == 'Sick' else 0

        # 应用变换（确保包含ToTensor）
        if self.transform:
            image = self.transform(image)

        return {
            'pixel_values': image,  # 此时应该是Tensor
            'label': torch.tensor(label, dtype=torch.long),
            'filename': img_name
        }


import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TextFileDataset_sub(Dataset):
    def __init__(self, txt_file, img_dir, sub_lung_dir, label_file, transform=None, classification_type=3):
        """
        Args:
            txt_file (str): 包含图像文件名的txt文件路径
            img_dir (str): 图像文件夹路径
            sub_lung_dir (str): 子肺图像文件夹路径
            label_file (str): 包含标签的txt文件路径
            transform (callable, optional): 图像变换操作
            classification_type (int): 分类类型 (3, 4 或 5)

        """
        self.img_dir = img_dir
        self.sub_lung_dir = sub_lung_dir
        self.classification_type = classification_type
        self.transform = transform if transform else transforms.ToTensor()

        # 读取图像文件名
        with open(txt_file, 'r') as f:
            self.img_names = [line.strip() for line in f if line.strip()]

        # 读取标签文件并构建字典 {filename: label}
        self.label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.label_dict[parts[0]] = int(parts[1])

        # 定义子图片的后缀
        self.sub_image_suffixes = [
            '_left_top', '_left_center', '_left_bottom',
            '_right_top', '_right_center', '_right_bottom'
        ]

    def __len__(self):
        return len(self.img_names)

    def _remap_label(self, original_label):
        """根据分类类型重新映射标签"""
        if self.classification_type == 3:
            # 3分类映射：跳过1和4
            if original_label == 0:
                return 0
            elif original_label == 2:
                return 1
            elif original_label == 3:
                return 2
            else:
                return None  # 需要跳过的标签
        elif self.classification_type == 4:
            # 4分类映射：跳过4
            if original_label == 4:
                return None
            else:
                return original_label
        else:
            # 5分类：保持原样
            return original_label

    def __getitem__(self, idx):
        """处理单个样本"""
        img_name = self.img_names[idx]
        base_name = os.path.splitext(img_name)[0]  # 去掉扩展名

        # 获取完整图像的标签
        label_str = img_name.split('_')[0]
        if label_str == 'Health':
            full_label = 0  # 健康为0
        elif label_str == 'Sick':
            full_label = 1  # 疾病为1
        else:
            raise ValueError(f"图像'{img_name}'对应的标签错误!请检查文件名格式")

        # 准备子图像数据
        sub_images = []
        sub_labels = []
        sub_filenames = []

        for suffix in self.sub_image_suffixes:
            # 构建子图像文件名
            sub_img_name = f"{base_name}{suffix}.png"

            # 检查并加载子图像
            sub_img_path = os.path.join(self.sub_lung_dir, sub_img_name)
            if not os.path.exists(sub_img_path):
                raise FileNotFoundError(f"子图像'{sub_img_name}'不存在!请检查路径")

            # 获取子图像标签
            if sub_img_name not in self.label_dict:
                raise ValueError(f"在标签字典中找不到图像'{sub_img_name}'对应的标签")

            original_label = self.label_dict[sub_img_name]
            remapped_label = self._remap_label(original_label)

            # 跳过不符合当前分类类型的标签
            if remapped_label is None:
                continue

            # 加载并转换图像
            sub_image = Image.open(sub_img_path).convert('RGB')
            if self.transform:
                sub_image = self.transform(sub_image)

            sub_images.append(sub_image)
            sub_labels.append(remapped_label)
            sub_filenames.append(sub_img_name)

        # 如果没有有效的子图像，返回None（需要在DataLoader中处理）
        if len(sub_images) == 0:
            return None

        return {
            'label': torch.tensor(full_label, dtype=torch.long),  # 完整图像标签
            'filename': img_name,  # 完整图像文件名
            'sub_pixel_values': torch.stack(sub_images),  # 子图像堆叠
            'sub_labels': torch.tensor(sub_labels, dtype=torch.long),  # 重新映射后的子图像标签
            'sub_filenames': sub_filenames  # 子图像文件名列表
        }

class TextFileDataset_sub_lung(Dataset):
    def __init__(self, txt_file, img_dir, sub_lung_dir,label_file, transform=None, classification_type=5):
        """
        Args:
            txt_file (str): 包含图像文件名的txt文件路径
            img_dir (str): 图像文件夹路径
            sub_lung_dir(str): sub lung图像文件夹路径
            label_file (str): 包含标签的txt文件路径
            transform (callable, optional): 图像变换操作
             "sample_id": str  # 唯一标识符
        """
        self.img_dir = img_dir
        self.sub_lung_dir = sub_lung_dir
        self.classification_type = classification_type
        # 确保至少包含ToTensor转换
        self.transform = transform if transform else transforms.ToTensor()

        # 读取图像文件名
        with open(txt_file, 'r') as f:
            self.img_names = [line.strip() for line in f if line.strip()]

        # 读取标签文件并构建字典 {filename: label}
        self.label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.label_dict[parts[0]] = int(parts[1])

        # 定义子图片的后缀
        self.sub_image_suffixes = [
            '_left_top', '_left_center', '_left_bottom',
            '_right_top', '_right_center', '_right_bottom'
        ]

    def __len__(self):
        return len(self.img_names)

    def _remap_label(self, original_label):
        """根据分类类型重新映射标签"""
        if self.classification_type == 3:
            # 3分类映射：跳过1和4
            if original_label == 0:
                return 0
            elif original_label == 2:
                return 1
            elif original_label == 3:
                return 2
            else:
                return None  # 需要跳过的标签
        elif self.classification_type == 4:
            # 4分类映射：跳过4
            if original_label == 4:
                return None
            else:
                return original_label
        else:
            # 5分类：保持原样
            return original_label


    def __getitem__(self, idx):
        """处理单个样本"""
        img_name = self.img_names[idx]
        base_name = os.path.splitext(img_name)[0]  # 去掉扩展名

        # 加载完整图像
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((512, 512))

        # 获取完整图像的标签
        label_str = img_name.split('_')[0]
        full_label = None
        if label_str == 'Health':
            full_label = 0
        elif label_str == 'Sick':
            full_label = 1
        else:
            raise ValueError(f"图像'{img_name}'对应的标签错误!请检查:\n")

        # 准备子图像数据
        sub_images = []
        sub_labels = []
        full_sub_labels = []  # 新增：保存所有子图像的原始标签
        full_sub_filenames = []  # 新增：保存所有子图像的文件名

        for suffix in self.sub_image_suffixes:
            # 构建子图像文件名
            sub_img_name = f"{base_name}{suffix}.png"  # 假设子图像都是png格式

            # 尝试加载子图像
            sub_img_path = os.path.join(self.sub_lung_dir, sub_img_name)
            if not os.path.exists(sub_img_path):
                raise ValueError(f"图像'{sub_img_name}'错误!请检查:\n")

            sub_image = Image.open(sub_img_path).convert('RGB')
            sub_image = sub_image.resize((512, 512))
            if sub_img_name not in self.label_dict:
                raise ValueError(f"在标签字典中找不到图像'{sub_img_name}'对应的标签！请检查:\n")

            original_label = self.label_dict[sub_img_name]
            remapped_label = self._remap_label(original_label)

            # 保存所有子图像的原始标签和文件名
            full_sub_labels.append(original_label)
            full_sub_filenames.append(sub_img_name)

            # 仅对有效标签（0、2）添加子图像和标签
            if remapped_label is not None:
                if self.transform:
                    sub_image = self.transform(sub_image)
                sub_images.append(sub_image)
                sub_labels.append(remapped_label)

        # 如果没有有效的子图像，跳过该样本
        if not sub_images:
            raise ValueError(f"没有有效的子图像或标签，跳过样本'{img_name}'")

        # 应用变换到完整图像
        if self.transform:
            image = self.transform(image)

        return {
            'pixel_values': image,  # 完整图像Tensor
            'label': torch.tensor(full_label, dtype=torch.long),
            'filename': img_name,
            'sub_pixel_values': torch.stack(sub_images),  # 子图像Tensor堆叠（仅0、2）
            'sub_labels': torch.tensor(sub_labels, dtype=torch.long),  # 有效子图像标签（仅0、2）
            'sub_filenames': [f"{base_name}{suffix}.png" for suffix in self.sub_image_suffixes if
                              self._remap_label(self.label_dict.get(f"{base_name}{suffix}.png", -1)) is not None],
            'full_sub_labels': torch.tensor(full_sub_labels, dtype=torch.long),  # 新增：所有子图像的原始标签
            'full_sub_filenames': full_sub_filenames  # 新增：所有子图像的文件名
        }


class TextFileDataset_sub_lung_error(Dataset):
    def __init__(self, txt_file, img_dir, sub_lung_dir,label_file, transform=None, classification_type=5):
        """
        Args:
            txt_file (str): 包含图像文件名的txt文件路径
            img_dir (str): 图像文件夹路径
            sub_lung_dir(str): sub lung图像文件夹路径
            label_file (str): 包含标签的txt文件路径
            transform (callable, optional): 图像变换操作
             "sample_id": str  # 唯一标识符
        """
        self.img_dir = img_dir
        self.sub_lung_dir = sub_lung_dir
        self.classification_type = classification_type
        # 确保至少包含ToTensor转换
        self.transform = transform if transform else transforms.ToTensor()

        # 读取图像文件名
        with open(txt_file, 'r') as f:
            self.img_names = [line.strip() for line in f if line.strip()]

        # 读取标签文件并构建字典 {filename: label}
        self.label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.label_dict[parts[0]] = int(parts[1])

        # 定义子图片的后缀
        self.sub_image_suffixes = [
            '_left_top', '_left_center', '_left_bottom',
            '_right_top', '_right_center', '_right_bottom'
        ]

    def __len__(self):
        return len(self.img_names)

    def _remap_label(self, original_label):
        """根据分类类型重新映射标签"""
        if self.classification_type == 3:
            # 3分类映射：跳过1和4
            if original_label == 0:
                return 0
            elif original_label == 1:
                return 1
            elif original_label == 2:
                return 2
            else:
                return None  # 需要跳过的标签
        elif self.classification_type == 4:
            # 4分类映射：跳过4
            if original_label == 4:
                return None
            else:
                return original_label
        else:
            # 5分类：保持原样
            return original_label

    def __getitem__(self, idx):
        """处理单个样本"""
        img_name = self.img_names[idx]
        base_name = os.path.splitext(img_name)[0]  # 去掉扩展名

        # 加载完整图像
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 获取完整图像的标签
        label_str = img_name.split('_')[0]
        full_label = None
        if label_str == 'Health':
            full_label = 0
        if label_str == 'Sick':
            full_label = 1
        if full_label == None:
            raise ValueError(f"图像'{img_name}'对应的标签错误!请检查:\n")
        # if img_name not in self.label_dict:
        #     raise ValueError(f"在标签字典中找不到图像'{img_name}'对应的标签！请检查:\n")
        # full_label = self.label_dict[img_name]



        # 准备子图像数据
        sub_images = []
        sub_labels = []

        for suffix in self.sub_image_suffixes:
            # 构建子图像文件名
            sub_img_name = f"{base_name}{suffix}.png"  # 假设子图像都是png格式

            # 尝试加载子图像
            sub_img_path = os.path.join(self.sub_lung_dir, sub_img_name)
            if os.path.exists(sub_img_path):
                sub_image = Image.open(sub_img_path).convert('RGB')
                # 获取子图像标签
                if sub_img_name not in self.label_dict:
                    raise ValueError(f"在标签字典中找不到图像'{sub_img_name}'对应的标签！请检查:\n")
                sub_label = self.label_dict[sub_img_name]
                # sub_label = self.label_dict.get(sub_img_name, 0)
            else:
                # 如果子图像不存在，使用完整图像和标签
                raise ValueError(f"图像'{sub_img_name}'错误!请检查:\n")

            # # 应用变换
            # if self.transform:
            #     sub_image = self.transform(sub_image)
            #
            # sub_images.append(sub_image)
            # sub_labels.append(sub_label)
            original_label = self.label_dict[sub_img_name]
            remapped_label = self._remap_label(original_label)

            # 跳过不符合当前分类类型的标签
            if remapped_label is None:
                continue

            # 加载并转换图像
            sub_image = Image.open(sub_img_path).convert('RGB')
            if self.transform:
                sub_image = self.transform(sub_image)

            sub_images.append(sub_image)
            sub_labels.append(remapped_label)
        # 如果没有有效的子图像，跳过该样本
        if not sub_images:
            raise ValueError(f"没有有效的子图像或标签，跳过样本'{img_name}'")

        # 应用变换到完整图像
        if self.transform:
            image = self.transform(image)



        return {
            'pixel_values': image,  # 完整图像Tensor
            'label': torch.tensor(full_label, dtype=torch.long),
            'filename': img_name,
            'sub_pixel_values': torch.stack(sub_images),  # 子图像Tensor堆叠
            'sub_labels': torch.tensor(sub_labels, dtype=torch.long),
            'sub_filenames': [f"{base_name}{suffix}.png" for suffix in self.sub_image_suffixes]
        }

transform = pth_transforms.Compose([
        pth_transforms.Resize((512, 512)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
# 使用示例
# train_dataset = TextFileDataset(
#     txt_file='/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/filename_txt/txt_5fold/fold1_test.txt',  # 替换为你的txt文件路径
#     img_dir='/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/guizhou_sick_health/',  # 替换为你的图像文件夹路径
#     transform=transform  # 替换为你的图像变换
# )

# train_dataset = TextFileDataset_sub_lung(
#         txt_file='/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/filename_txt/txt_5fold/fold1_test.txt',
#         # 替换为你的txt文件路径
#         img_dir='/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/guizhou_sick_health/',  # 替换为你的图像文件夹路径
#         sub_lung_dir='/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/guizhou_sub_lung/',
#         label_file= '/data1/jingyi/cjy/data/huggingface/guizhou_feiqu_complete/filename_txt/all_label.txt',
#         transform=None  # 替换为你的图像变换
#     )
#
# dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#
# # 4. 测试：打印一个 batch 的数据
# batch = next(iter(dataloader))
# print(batch)
