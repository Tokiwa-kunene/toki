"""
主要作用：
将在preprocessing中处理好的数据进行tokenizer化
修改中括号[]的定义，使其能读取某个为特定id句子的信息（input_ids, attention_mask,label)

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
        self.data = data                    #一个大字典，包含两个列表，即text和label
        self.tokenizer = tokenizer          #从外部引入的tokenizer
        self.max_length = max_length        #设定最大读取长度

    def __len__(self):
        """返回数据集大小"""
        return len(self.data['label'])

    def __getitem__(self, idx):

        text = self.data['text'][idx]       #中括号[]表示读取，这里即读取data字典里面的test列表，再读取列表中的idx
        label = self.data['label'][idx]     #此处idx为形参，取决于调用时[]内部的数字是多少，即读取对应位置的数据

        encoding = self._tokenize(text)     #将在上面刚刚读取的text进行tokenizer化，生成encoding字典
        '''
        备注：
        idx: 样本索引，即再使用[]时的内部数字
            例如，若[5]，则idx=5，也就提取了id为5的那句话的信息
            其实也就是抓一句话然后给它tokenizer了
            
        返回一个字典 encoding，里面包含两个纯粹的一维 PyTorch 张量：input_ids 和 attention_mask（Shape 均为 [max_length]）。
        input_idx:即为由文本转化而来的一维向量
        attention_mask:即标注真实文本和填充符的向量，真实文本位置为1，填充符位置为0，使自注意力计算中占位符不影响计算结果
        attention_mask向量由.tokenizer()功能生成
        下面的return：改变[]的功能，使其变为读取一个包含三个项目的字典
        '''

        return {
            'input_ids': encoding['input_ids'],             #关于一句话的一个list
            'attention_mask': encoding['attention_mask'],   #同样关于一句话的list
            'label': torch.tensor(label, dtype=torch.long)  #单纯的一个值，表示这句话的label
        }

    def _tokenize(self, text):
        """
        对文本进行tokenize
        文本形式的句子 --> 分割成许多token并加如[CLS]等分隔符 --> 根据字典将token和分隔符转化为数字
        Args:
            text: 输入文本

        Returns:
            编码后的字典
        """
        encoding = self.tokenizer(
            text,                           #原始句子文本
            max_length=self.max_length,     #最大长度
            padding='max_length',           #填充空白部分，一般设置为1
            truncation=True,                #对超过设定长度的部分进行截断
            return_tensors='pt'             #返回格式为PyTorch的tensor
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  #squeeze使强行在二维空间中表示的一维向量还原成单纯的一维表示
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }