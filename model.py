"""
模型架构定义 - SimCSE兼容版本
核心：保持 Attention Pooling 和投影头结构
关键：确保训练时 Dropout 开启（model.train()）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


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
            nn.Linear(hidden_size, hidden_size),        #全连接层：hidden_size --> hidden_size
            nn.Tanh(),                                  #激活函数Tahn
            nn.Linear(hidden_size, 1)        #全连接层：hidden_size --> 1 即注意力分数
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
        #将输入向量的数据扔进上面构造好的attention层里，计算分数
        #nn.Sequential()函数在调用的时候，会直接执行其内置的forward函数，功能为把()内部的数据扔进该函数构建好的网络中跑一遍

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)       #在末尾强制增加了一个维度
            attention_scores = attention_scores.masked_fill(    #masked_fill：替换数值，mask==0定位，替换为-1e9
                attention_mask == 0,
                -1e9
            )

        attention_weights = F.softmax(attention_scores, dim=1)
        #对每个token的注意力分数进行softmax归一化处理，一个句子内部所有token分数之和为1（即权重）
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)
        #每个token的原始向量*归一化后的权重，之后每个处理后的token向量互相相加，得到一个句子向量（1*768）

        return pooled_output, attention_weights


class SpecificFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.1):
        super(SpecificFeatureExtractor, self).__init__()

        self.attn_pooling = AttentionPooling(hidden_size)   #backbone后的数据先经历的池化层

        self.classifier = nn.Sequential(                    #经过池化层之后的数据在这之后要经历的结构
            nn.Linear(hidden_size, hidden_size),  # 保持维度进行特征变换
            nn.Tanh(),  # 非线性激活函数 (也可以试 ReLU)
            nn.Dropout(p=dropout_rate),  # 防止过拟合
            nn.Linear(hidden_size, num_classes)  # 最终分类层
        )
        # === 修改结束 ===

    def forward(self, sequence_output, attention_mask):
        pooled_output, attention_weights = self.attn_pooling(   #池化层操作
            sequence_output,
            attention_mask
        )

        # 注意：这里如果用了 Sequential，前向传播可以直接传
        # features = self.dropout(pooled_output)    # 这行可以去掉了，因为Sequential里已经包含了Dropout
        logits = self.classifier(pooled_output)     #池化层之后的分类层操作，注意力权重不变，仅输入了pool_output

        return logits, pooled_output, attention_weights #返回最终值，池化值和注意力权重，Branch A 结束


class XLMSentimentModel(nn.Module):
    """
    轻量化 SimCSE-ABSA 模型

    Branch A: 纯 [CLS] 分类 (等效于 Baseline 的直接投射)
    Branch B: SimCSE 对比学习
    """

    def __init__(self, model_path, num_classes=2, projection_dim=128, dropout_rate=0.1):
        super(XLMSentimentModel, self).__init__()

        print("\n" + "=" * 70)
        print("初始化 精简版 SimCSE-ABSA 模型 (No Pooling)")
        print("=" * 70)

        # 1. 加载 Backbone
        print(f"\n[1/3] 加载 Backbone (AutoModel)")
        self.backbone = AutoModel.from_pretrained(
            model_path,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        self.hidden_size = self.backbone.config.hidden_size
        print(f"      ✓ Backbone加载成功 (hidden_size={self.hidden_size})")

        # 2. Branch A: 情感分类分支 (等效于 Baseline)
        print(f"\n[2/3] 初始化 Branch A: 极简分类头")
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_size, num_classes)
        )
        print(f"      ✓ 删除了 Attention Pooling，直接使用 [CLS]")
        print(f"      ✓ 分类器: {self.hidden_size} -> {num_classes}")

        # 3. Branch B: 对比学习投影头
        print(f"\n[3/3] 初始化 Branch B: SimCSE 对比学习分支")
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, projection_dim)
        )
        print(f"      ✓ 投影头: {self.hidden_size} -> {projection_dim}")

        print("\n" + "=" * 70)
        print("模型初始化完成")
        print("=" * 70 + "\n")

    def forward(self, input_ids, attention_mask, return_attention=False):
        # 1. Backbone 输出
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 直接提取 [CLS] 的最终向量作为全局表达
        cls_output = outputs.pooler_output

        # 2. Branch A: 情感分类 (极简)
        logits = self.classifier(cls_output)

        # 3. Branch B: SimCSE 对比学习
        features = self.projection_head(cls_output)
        features = F.normalize(features, dim=1)

        # 保持与 trainer.py 潜在的解包习惯兼容
        if return_attention:
            # 返回 None 因为没有注意力权重可提供
            return logits, features, None
        else:
            return logits, features