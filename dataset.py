"""
PyTorch数据集定义 - SimCSE版本
核心变更：不再返回翻译文本，仅返回原始文本
SimCSE 在 Trainer 中通过两次 forward 生成正样本对
"""
import torch
from torch.utils.data import Dataset


class CrossLingualDataset(Dataset):
    """跨语言情感分析数据集 - SimCSE版本"""

    def __init__(self, data, tokenizer, max_length):
        """
        初始化数据集

        Args:
            data: 数据字典，包含 'text' 和 'label'
            tokenizer: XLM-RoBERTa tokenizer
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回数据集大小"""
        return len(self.data['label'])

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            样本字典，包含 input_ids, attention_mask, label
        """
        text = self.data['text'][idx]
        label = self.data['label'][idx]

        # Tokenize文本
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