# ========================================
# 文件6: model.py - 模型架构
# ========================================
"""
模型架构定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel


class XLMSentimentModel(nn.Module):
    """
    双分支XLM-RoBERTa模型
    Branch A: 情感分类
    Branch B: 对比学习
    """

    def __init__(self, model_path, num_classes=2, projection_dim=128):
        """
        初始化模型

        Args:
            model_path: XLM-RoBERTa模型路径
            num_classes: 分类类别数
            projection_dim: 对比学习投影维度
        """
        super(XLMSentimentModel, self).__init__()

        # Backbone: XLM-RoBERTa
        print(f"加载XLM-RoBERTa模型: {model_path}")
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_path)
        self.hidden_size = self.xlm_roberta.config.hidden_size

        # Branch A: 情感分类器
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # Branch B: 对比学习投影头 (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )

        print("模型初始化完成")

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            logits: 分类logits [batch_size, num_classes]
            features: 对比学习特征 [batch_size, projection_dim]
        """
        # 获取XLM-RoBERTa的输出
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS] token的表示（第一个token）
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Branch A: 分类
        logits = self.classifier(cls_output)

        # Branch B: 对比学习
        features = self.projection_head(cls_output)
        features = F.normalize(features, dim=1)  # L2归一化

        return logits, features

