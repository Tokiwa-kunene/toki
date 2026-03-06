# toki
个人小作文试水用代码存档，大佬就不要看了，没啥参考价值
Personal recording on the Github, no any valuable things inside, please igrone.
個人的なコード記録です、ただ何もないつまらないものしかないから、ご無視してください、ご協力ありがとうございます。

第一版，模型结构如下：
=======================================================================
                        [ 输入层 (Input Layer) ]
                  input_ids         attention_mask
                       |                  |
=======================================================================
                                  |
                       [ 主干网络 (Backbone) ]
                   XLM-RoBERTa (Pre-trained Model)
                                  |
         ---------------------------------------------------
         |                                                 |
  sequence_output                                     cls_output
 (包含所有Token的特征)                                (仅代表句子的全局特征)
 [batch, seq_len, hidden_size]                       [batch, hidden_size]
         |                                                 |
======================================    ======================================
| Branch A: 特定情感特征提取与分类   |    | Branch B: SimCSE 句子表征与对比学习|
======================================    ======================================
         |                                                 |
[ Attention Pooling (注意力池化) ]                [ Projection Head (投影头) ]
| 1. nn.Linear -> Tanh -> nn.Linear  |        | 1. nn.Linear(hidden, proj_dim) |
| 2. Masking 屏蔽填充符 (替换为-1e9) |        | 2. Tanh() 激活函数             |
| 3. Softmax 归一化 (attention_weights)|      | 3. nn.Linear(proj_dim, proj_dim)|
| 4. 矩阵加权求和                    |                 |
         |                                                 |
   pooled_output                                           |
   [batch, hidden_size]                                    |
         |                                                 |
[ Classifier (分类器) ]                           [ L2 Normalization (归一化) ]
| 1. nn.Linear(hidden, hidden)       |        | F.normalize(dim=1)             |
| 2. Tanh()                          |                 |
| 3. nn.Dropout(dropout_rate)        |                 |
| 4. nn.Linear(hidden, num_classes)  |                 |
         |                                                 |
======================================    ======================================
      logits                                            features
  [batch, num_classes]                               [batch, proj_dim]
  (用于计算交叉熵分类 Loss)                           (用于计算无监督对比学习 Loss)
======================================    ======================================
