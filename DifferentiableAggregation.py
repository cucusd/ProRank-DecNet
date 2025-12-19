import torch
from torch import nn
import torch.nn.functional as F
# class DifferentiableAggregation(nn.Module):
#     def __init__(self, k_init=1.0):
#         super().__init__()
#         self.k = k_init  # 初始斜率
#         self.eps = 0.1   # 防止梯度消失的小常数
#
#     def forward(self, sub_logits):
#         sub_probs = F.softmax(sub_logits, dim=-1)
#         c = sub_probs[:, :, 1] + sub_probs[:, :, 2]
#         S = c.sum(dim=1)  # 软计数
#
#         # 改进1：非线性变换压缩输入范围
#         S_transformed = torch.tanh(S - 1)  # 将S-1压缩到[-1, 1]
#
#         # 改进2：动态调整k（示例，实际可按epoch调整）
#         k = max(1.0, self.k * 1.01)  # 缓慢增大斜率
#
#         # 改进3：带epsilon的Sigmoid
#         P_original_1 = torch.sigmoid(k * S_transformed + self.eps)
#         return P_original_1


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DifferentiableAggregation_test(nn.Module):
    def __init__(self, k):
        super().__init__()
        # self.k_init = k_init
        # self.k_max = k_max
        # self.growth = growth
        #
        # self.register_buffer('k', torch.tensor(k_init))
        self.k =k

    def forward(self, sub_logits, original_indices):
        """
        sub_logits: [total_subimages, 3]
        original_indices: [total_subimages], 表示每个子图属于哪个原图
        """
        # Step 1: 计算子图各类别概率
        sub_probs = F.softmax(sub_logits, dim=-1)  # [total_subimages, 3]

        # Step 2: 计算每个原图的两种统计量
        unique_original_indices = torch.unique(original_indices)
        batch_size = len(unique_original_indices)
        S_1 = torch.zeros(batch_size, device=sub_logits.device)  # 类别1+2的概率和
        S_0 = torch.zeros(batch_size, device=sub_logits.device)  # 类别0的概率和
        N = torch.zeros(batch_size, device=sub_logits.device)  # 子图数量

        for idx in unique_original_indices:
            mask = (original_indices == idx)
            S_1[idx] = sub_logits[mask, 1].sum() + sub_logits[mask, 2].sum()
            S_0[idx] = sub_logits[mask, 0].sum()
            N[idx] = mask.sum()  # 该原图的子图数量

        # Step 3: 计算两类决策信号
        # 决策1：类别1+2的概率和是否足够大
        # P_decision_1 = torch.sigmoid(self.k * (S_1 - 1))
        P_decision_1 = torch.sigmoid(self.k * (1-S_1))
        # 决策2：类别0的概率和是否足够大（考虑子图数量）
        # 这里用 (S_0 - threshold) 表示是否超过阈值
        # P_decision_0 = torch.sigmoid(self.k * (S_0 - 4))
        P_decision_0 = torch.sigmoid(self.k * (5-S_0))
        # # Step 4: 平衡两类决策（使用softmax或直接归一化）
        logits_1 = torch.log(P_decision_1 + 1e-10)
        logits_0 = torch.log(P_decision_0 + 1e-10)

        # 拼接成二维logits [batch_size, 2]
        # logits_original = torch.stack([logits_0, logits_1], dim=1)
        logits_original = torch.stack([logits_1,logits_0], dim=1)
        return logits_original


import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableAggregation_avg(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def func(self, sub_logits,current_sub_logits_full):
        """
        通用函数，处理每个原图的子图 logits，输出形状为 [2, 1]
        sub_logits: [num_subimages, 3], 某个原图的所有子图 logits
        Returns: [2, 1], 该原图的胸片 logits
        """
        # 空函数，待用户实现
        # pass
        if sub_logits.size(0) < 6:
            if not isinstance(current_sub_logits_full, np.ndarray):
                current_sub_logits_full = current_sub_logits_full.cpu().numpy()
            count_label_4 = np.sum(current_sub_logits_full == 4)
            count_label_1 = np.sum(current_sub_logits_full == 1)
        else:
            count_label_4 = 0
            count_label_1 = 0
        # 初始化输出张量
        logits_d = torch.zeros(2, 1, device=sub_logits.device)
        # 对每个子图取最大 logits 值的平均值
        max_logits_per_subimage = torch.max(sub_logits, dim=1)[0]  # [num_subimages]
        avg_max_logits = max_logits_per_subimage.mean()  # 标量

        # 提取每个子图的类别 0 的 logits 值的平均值
        class_0_logits = sub_logits[:, 0]  # [num_subimages]
        all_class_0_logits = torch.sum(class_0_logits)
        avg_class_0_logits = torch.mean(class_0_logits)  # 标量

        # 提取每个子图的类别 1 的 logits 值的平均值
        class_1_logits = sub_logits[:, 1]  # [num_subimages]
        avg_class_1_logits = torch.mean(class_1_logits)  # 标量
        all_class_1_logits = torch.sum(class_1_logits)

        # 提取每个子图的类别 2 的 logits 值的平均值
        class_2_logits = sub_logits[:, 2]  # [num_subimages]
        avg_class_2_logits = torch.mean(class_2_logits)  # 标量
        all_class_2_logits = torch.sum(class_2_logits)

        class_0_Judge= torch.sigmoid(self.k * ((all_class_0_logits+count_label_1*avg_max_logits)-5 * avg_max_logits))
        class_1_Judge = torch.sigmoid(self.k * ((all_class_1_logits+all_class_2_logits+count_label_4*avg_max_logits)-1 * avg_max_logits ))
        # class_2_Judge = torch.sigmoid(self.k * ( 6 * avg_class_2_logits-2 * avg_max_logits))

        logits_d[0][0] = class_0_Judge
        logits_d[1][0] = class_1_Judge

        return logits_d.squeeze(dim=1)




    def forward(self, sub_logits, original_indices,full_sub_labels,full_original_indices):
        """
        sub_logits: [total_subimages, 3], 所有子图的 logits
        original_indices: [total_subimages], 表示每个子图属于哪个原图
        Returns: [2, 1], 所有原图胸片 logits 的总和
        """
        # sub_logits.size
        # 获取唯一的原图索引
        unique_original_indices = torch.unique(original_indices)
        batch_size = len(unique_original_indices)

        # 初始化输出张量
        total_logits = torch.zeros(batch_size, 2, device=sub_logits.device)

        # 对每个原图的子图 logits 进行处理
        for idx in unique_original_indices:
            # 提取属于当前原图的子图 logits
            mask = (original_indices == idx)
            current_sub_logits = sub_logits[mask]  # [num_subimages, 3]
            mask_full = (full_original_indices == idx)
            current_sub_logits_full = full_sub_labels[mask_full]  # [num_subimages, 3]
            # 调用通用函数 func 处理当前原图的子图 logits
            original_logits = self.func(current_sub_logits,current_sub_logits_full)  # 期望输出 [2, 1]

            # # 验证 func 输出形状
            # if original_logits.shape != (2, 1):
            #     raise ValueError(f"func output shape must be [2, 1], got {original_logits.shape}")

            # 累加到总和
            total_logits[idx.item()] = original_logits

        return total_logits


class DifferentiableAggregation_more(nn.Module):
    def __init__(self, k,s_function):
        super().__init__()
        self.k = k
        self.s_function = s_function

    """Error function: f(x) = 0.5 * (1 + erf((x-a)/(sigma*sqrt(2))))"""
    def error_function(self,x, sigma=0.2):
        return 0.5 * (1 + torch.erf((x) / (sigma * torch.sqrt(torch.tensor(2.0)))))

    # Hyperbolic tangent (tanh) using torch.tanh
    def tanh_function(self,x):
        return 0.5 * (1 + torch.tanh (x))

    def func(self, sub_logits,current_sub_logits_full,s_function):
        """
        通用函数，处理每个原图的子图 logits，输出形状为 [2, 1]
        sub_logits: [num_subimages, 3], 某个原图的所有子图 logits
        Returns: [2, 1], 该原图的胸片 logits
        """
        # 空函数，待用户实现
        # pass
        if sub_logits.size(0) < 6:
            if not isinstance(current_sub_logits_full, np.ndarray):
                current_sub_logits_full = current_sub_logits_full.cpu().numpy()
            count_label_4 = np.sum(current_sub_logits_full == 4)
            count_label_1 = np.sum(current_sub_logits_full == 1)
        else:
            count_label_4 = 0
            count_label_1 = 0
        # 初始化输出张量
        logits_d = torch.zeros(2, 1, device=sub_logits.device)
        # 对每个子图取最大 logits 值的平均值
        max_logits_per_subimage = torch.max(sub_logits, dim=1)[0]  # [num_subimages]
        avg_max_logits = max_logits_per_subimage.mean()  # 标量

        # 提取每个子图的类别 0 的 logits 值的平均值
        class_0_logits = sub_logits[:, 0]  # [num_subimages]
        all_class_0_logits = torch.sum(class_0_logits)
        avg_class_0_logits = torch.mean(class_0_logits)  # 标量

        # 提取每个子图的类别 1 的 logits 值的平均值
        class_1_logits = sub_logits[:, 1]  # [num_subimages]
        avg_class_1_logits = torch.mean(class_1_logits)  # 标量
        all_class_1_logits = torch.sum(class_1_logits)

        # 提取每个子图的类别 2 的 logits 值的平均值
        class_2_logits = sub_logits[:, 2]  # [num_subimages]
        avg_class_2_logits = torch.mean(class_2_logits)  # 标量
        all_class_2_logits = torch.sum(class_2_logits)

        # y_erf = self.error_function(x)
        # y_tanh = self.tanh_function(x)
        if s_function == 'sigmoid':
            class_0_Judge= torch.sigmoid(self.k * ((all_class_0_logits+count_label_1*avg_max_logits)-5 * avg_max_logits))
            class_1_Judge = torch.sigmoid(self.k * ((all_class_1_logits+all_class_2_logits+count_label_4*avg_max_logits)-1 * avg_max_logits ))
            # class_2_Judge = torch.sigmoid(self.k * ( 6 * avg_class_2_logits-2 * avg_max_logits))
        if s_function == 'tanh':
            class_0_Judge = self.tanh_function(self.k * ((all_class_0_logits + count_label_1 * avg_max_logits) - 5 * avg_max_logits))
            class_1_Judge = self.tanh_function(self.k * ((all_class_1_logits + all_class_2_logits + count_label_4 * avg_max_logits) - 1 * avg_max_logits))

        if s_function == 'Error':
            class_0_Judge = self.error_function(
                ((all_class_0_logits + count_label_1 * avg_max_logits) - 5 * avg_max_logits),self.k)
            class_1_Judge = self.error_function(((all_class_1_logits + all_class_2_logits + count_label_4 * avg_max_logits) - 1 * avg_max_logits),self.k)

        logits_d[0][0] = class_0_Judge
        logits_d[1][0] = class_1_Judge

        return logits_d.squeeze(dim=1)




    def forward(self, sub_logits, original_indices,full_sub_labels,full_original_indices):
        """
        sub_logits: [total_subimages, 3], 所有子图的 logits
        original_indices: [total_subimages], 表示每个子图属于哪个原图
        Returns: [2, 1], 所有原图胸片 logits 的总和
        """
        # sub_logits.size
        # 获取唯一的原图索引
        unique_original_indices = torch.unique(original_indices)
        batch_size = len(unique_original_indices)

        # 初始化输出张量
        total_logits = torch.zeros(batch_size, 2, device=sub_logits.device)

        # 对每个原图的子图 logits 进行处理
        for idx in unique_original_indices:
            # 提取属于当前原图的子图 logits
            mask = (original_indices == idx)
            current_sub_logits = sub_logits[mask]  # [num_subimages, 3]
            mask_full = (full_original_indices == idx)
            current_sub_logits_full = full_sub_labels[mask_full]  # [num_subimages, 3]
            # 调用通用函数 func 处理当前原图的子图 logits
            original_logits = self.func(current_sub_logits,current_sub_logits_full,self.s_function)  # 期望输出 [2, 1]

            # # 验证 func 输出形状
            # if original_logits.shape != (2, 1):
            #     raise ValueError(f"func output shape must be [2, 1], got {original_logits.shape}")

            # 累加到总和
            total_logits[idx.item()] = original_logits

        return total_logits



