"""
模型架构定义 - SimCSE兼容版本
核心：保持 Attention Pooling 和投影头结构
关键：确保训练时 Dropout 开启（model.train()）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel


class AttentionPooling(nn.Module):
    """
    注意力池化层
    使用可学习的注意力机制来聚合序列信息
    """

    def __init__(self, hidden_size):
        """
        初始化注意力池化层

        Args:
            hidden_size: 隐藏层维度（例如XLM-RoBERTa的768）
        """
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, sequence_output, attention_mask=None):
        """
        前向传播

        Args:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            pooled_output: 池化后的输出 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        attention_scores = self.attention(sequence_output)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                -1e9
            )

        attention_weights = F.softmax(attention_scores, dim=1)
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)

        return pooled_output, attention_weights


class SpecificFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.1):
        super(SpecificFeatureExtractor, self).__init__()

        self.attn_pooling = AttentionPooling(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        # === 修改开始: 将单层 Linear 改为 MLP ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 保持维度进行特征变换
            nn.Tanh(),  # 非线性激活函数 (也可以试 ReLU)
            nn.Dropout(p=dropout_rate),  # 防止过拟合
            nn.Linear(hidden_size, num_classes)  # 最终分类层
        )
        # === 修改结束 ===

    def forward(self, sequence_output, attention_mask):
        pooled_output, attention_weights = self.attn_pooling(
            sequence_output,
            attention_mask
        )

        # 注意：这里如果用了 Sequential，前向传播可以直接传
        # features = self.dropout(pooled_output) # 这行可以去掉了，因为Sequential里已经包含了Dropout
        logits = self.classifier(pooled_output)

        return logits, pooled_output, attention_weights
'''
class SpecificFeatureExtractor(nn.Module):
    """特定特征提取子网络"""

    def __init__(self, hidden_size, num_classes):
        """
        初始化特征提取器

        Args:
            hidden_size: 隐藏层维度
            num_classes: 分类类别数
        """
        super(SpecificFeatureExtractor, self).__init__()

        self.attn_pooling = AttentionPooling(hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence_output, attention_mask):
        """
        前向传播

        Args:
            sequence_output: 序列输出 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            logits: 分类logits [batch_size, num_classes]
            pooled_output: 池化后的特征 [batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, seq_len, 1]
        """
        pooled_output, attention_weights = self.attn_pooling(
            sequence_output,
            attention_mask
        )

        features = self.dropout(pooled_output)
        logits = self.classifier(features)

        return logits, pooled_output, attention_weights
'''

class XLMSentimentModel(nn.Module):
    """
    SimCSE-ABSA模型

    架构：
    - Branch A: 情感分类（Attention Pooling）
    - Branch B: SimCSE 对比学习（基于 Dropout）

    SimCSE 核心思想：
    同一输入两次 forward（利用 Dropout 噪声），生成两个不同的特征向量作为正样本对
    """

    def __init__(self, model_path, num_classes=2, projection_dim=128, dropout_rate=0.1):
        """
        初始化 SimCSE-ABSA 模型

        Args:
            model_path: XLM-RoBERTa模型路径
            num_classes: 分类类别数
            projection_dim: 对比学习投影维度
        """
        super(XLMSentimentModel, self).__init__()

        print("\n" + "=" * 70)
        print("初始化 SimCSE-ABSA 模型")
        print("=" * 70)

        # Backbone: XLM-RoBERTa
        print(f"\n[1/3] 加载 XLM-RoBERTa Backbone")
        print(f"      模型路径: {model_path}")
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            model_path,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )

        self.hidden_size = self.xlm_roberta.config.hidden_size
        print(f"      ✓ Backbone加载成功 (hidden_size={self.hidden_size})")
        print(f"      ✓ Dropout Rate: {self.xlm_roberta.config.hidden_dropout_prob}")

        # Branch A: 情感分类器
        print(f"\n[2/3] 初始化 Branch A: 情感分类分支")
        self.specific_feature_extractor = SpecificFeatureExtractor(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        print(f"      ✓ 使用 Attention Pooling")
        print(f"      ✓ 分类器: {self.hidden_size} -> {num_classes}")

        # Branch B: 对比学习投影头
        print(f"\n[3/3] 初始化 Branch B: SimCSE 对比学习分支")
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )
        print(f"      ✓ 投影头: {self.hidden_size} -> {projection_dim}")
        print(f"      ✓ SimCSE: 利用 Dropout 生成正样本对")

        print("\n" + "=" * 70)
        print("SimCSE-ABSA 模型初始化完成")
        print("=" * 70)
        print("\n架构总览:")
        print("  XLM-RoBERTa (Backbone, Dropout=0.1)")
        print("    ├── Branch A: 情感分类")
        print("    │     └── Attention Pooling -> Classifier")
        print("    │")
        print("    └── Branch B: SimCSE 对比学习")
        print("          └── [CLS] -> Projection Head -> L2 Norm")
        print("          └── 同一输入两次Forward，Dropout生成正样本对")
        print("=" * 70 + "\n")

    def forward(self, input_ids, attention_mask, return_attention=False):
        """
        前向传播

        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_attention: 是否返回注意力权重

        Returns:
            logits: 分类logits [batch_size, num_classes]
            features: 对比学习特征 [batch_size, projection_dim]
            attention_weights: (可选) 注意力权重
        """
        # XLM-RoBERTa 输出
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        cls_output = outputs.pooler_output

        # Branch A: 情感分类
        logits, pooled_features, attention_weights = self.specific_feature_extractor(
            sequence_output,
            attention_mask
        )

        # Branch B: SimCSE 对比学习
        features = self.projection_head(cls_output)
        features = F.normalize(features, dim=1)

        if return_attention:
            return logits, features, attention_weights
        else:
            return logits, features