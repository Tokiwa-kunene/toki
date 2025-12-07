# ========================================
# 文件7: losses.py - 损失函数
# ========================================
"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比学习损失
    用于拉近锚点与正例的距离，推远负例
    """

    def __init__(self, temperature=0.07):
        """
        初始化损失函数

        Args:
            temperature: 温度参数，控制分布的平滑度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive1, positive2):
        """
        计算InfoNCE损失

        Args:
            anchor: 锚点特征 (原始英文) [batch_size, dim]
            positive1: 正例1 (掩码英文) [batch_size, dim]
            positive2: 正例2 (翻译日文) [batch_size, dim]

        Returns:
            loss: InfoNCE损失值
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # 计算相似度矩阵
        # anchor vs positive1
        sim_anchor_pos1 = torch.matmul(anchor, positive1.T) / self.temperature
        # [batch_size, batch_size]

        # anchor vs positive2
        sim_anchor_pos2 = torch.matmul(anchor, positive2.T) / self.temperature
        # [batch_size, batch_size]

        # 对角线元素是正样本对
        labels = torch.arange(batch_size, device=device)

        # 计算两个正例的损失并平均
        loss1 = F.cross_entropy(sim_anchor_pos1, labels)
        loss2 = F.cross_entropy(sim_anchor_pos2, labels)

        return (loss1 + loss2) / 2
