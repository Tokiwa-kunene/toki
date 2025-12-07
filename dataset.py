# ========================================
# 文件5: dataset.py - PyTorch数据集
# ========================================
"""
PyTorch数据集定义
"""
import torch
from torch.utils.data import Dataset


class CrossLingualDataset(Dataset):
    """跨语言情感分析数据集"""

    def __init__(self, data, tokenizer, max_length, is_train=True):
        """
        初始化数据集

        Args:
            data: 数据字典
            tokenizer: XLM-RoBERTa tokenizer
            max_length: 最大序列长度
            is_train: 是否为训练集
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        """返回数据集大小"""
        return len(self.data['label'])

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            样本字典
        """
        label = self.data['label'][idx]

        if self.is_train:
            # 训练模式：返回三组文本（原文、掩码、翻译）
            original_text = self.data['original_text'][idx]
            masked_text = self.data['masked_text'][idx]
            translated_text = self.data['translated_text'][idx]

            # Tokenize三个文本
            original_encoding = self._tokenize(original_text)
            masked_encoding = self._tokenize(masked_text)
            translated_encoding = self._tokenize(translated_text)

            return {
                'original_input_ids': original_encoding['input_ids'],
                'original_attention_mask': original_encoding['attention_mask'],
                'masked_input_ids': masked_encoding['input_ids'],
                'masked_attention_mask': masked_encoding['attention_mask'],
                'translated_input_ids': translated_encoding['input_ids'],
                'translated_attention_mask': translated_encoding['attention_mask'],
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # 测试模式：只返回一组文本
            text = self.data['text'][idx]
            encoding = self._tokenize(text)

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'label': torch.tensor(label, dtype=torch.long)
            }

    def _tokenize(self, text):
        """
        对文本进行tokenize

        Args:
            text: 输入文本

        Returns:
            编码后的字典
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }