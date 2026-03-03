"""
训练器模块 - SimCSE版本
核心变更：
1. 实现 SimCSE (同一输入两次forward，利用Dropout)
2. 优化 Checkpoint 保存策略
3. 动态语言显示 (修复打印硬编码问题)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os
import time
from datetime import timedelta


class Trainer:
    """SimCSE-ABSA 训练器"""

    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        """
        初始化训练器
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE
        )

        # 学习率调度器
        total_steps = len(train_loader) * config.NUM_EPOCHS
        warmup_steps = int(config.WARMUP_RATIO * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 损失函数
        self.cls_criterion = nn.CrossEntropyLoss()
        self.temperature = config.TEMPERATURE
        self.alpha = config.ALPHA

        print(f"\n损失权重配置:")
        print(f"  Alpha = {self.alpha}")
        print(f"  分类损失权重 = {1 - self.alpha:.3f}")
        print(f"  SimCSE损失权重 = {self.alpha:.3f}")

        # 初始化统一的最佳指标记录器
        self.best_stats = {
            'valid': {'acc': 0.0, 'acc_epoch': 0, 'f1': 0.0, 'f1_epoch': 0},
            'test':  {'acc': 0.0, 'acc_epoch': 0, 'f1': 0.0, 'f1_epoch': 0}
        }
        # 保留 best_valid_f1 变量用于 Checkpoint 判断逻辑
        self.best_valid_f1 = 0.0

        # 时间统计
        self.epoch_times = []
        self.total_start_time = None

    def train_epoch(self, epoch):
        """
        训练一个epoch（SimCSE版本）
        """
        self.model.train()  # 确保 Dropout 开启！
        #.train() 方法其实是继承自 PyTorch 的底层基类 nn.Module，作用为保证该模型为训练开启的模式，switch on!みたいな感じ~
        total_loss = 0
        total_cls_loss = 0
        total_cl_loss = 0

        epoch_start_time = time.time()

        #进度条，tqdm是一个进度条功能：
        progress_bar = tqdm(
            self.train_loader,
            desc=f"训练 Epoch {epoch + 1}/{self.config.NUM_EPOCHS}"
        )

        #将数据运送进GPU中：
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.config.DEVICE)
            attention_mask = batch['attention_mask'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)

            # ===== SimCSE 核心：同一输入两次 forward =====
            """
            说明：
            两次forward输入的token数值（即input_ids）完全一样，仅仅是dropout的值不一样
            这就会导致一个样本却诞生了高度相似但又不一致的样本对
            即，正样本对，也为后面的对比学习打下了基础
            
            logits 为Branch A（分类头）的结果
            feathers 为Branch B（对比头）的结果
            """
            # 第一次 forward（Dropout 状态1）
            logits_1, features_1 = self.model(input_ids, attention_mask)
            #在此时对实例model进行参数的注入，随后才开始运行XLMSentimentModel类中的forward函数，并返回两个值
            #logits是Branch A的最终结果，features则是只进行了投影头处理的Branch B的结果

            # 第二次 forward（Dropout 状态2，不同的随机掩码）
            logits_2, features_2 = self.model(input_ids, attention_mask)

            # ===== 计算分类损失（使用第一次的logits） =====
            loss_cls = self.cls_criterion(logits_1, labels) #对正样本对中的1号样本进行交叉熵损失计算
            #相当于在Branch A的最后补充了一个交叉熵损失函数

            # ===== 计算 SimCSE 对比损失 =====
            # features_1 和 features_2 互为正样本对
            loss_cl = self._compute_simcse_loss(features_1, features_2) #计算对比损失，def在下面

            # ===== 归一化总损失 =====
            loss = (1.0 - self.alpha) * loss_cls + self.alpha * loss_cl
            #可以发现其实第二次forward产生的Branch A的logits并没有什么用...

            # ===== 反向传播 =====
            self.optimizer.zero_grad()          #优化器，清除了（重置为零）模型中所有参数在上一轮计算中残留的梯度数据
            loss.backward()                     #反向传播算法
            torch.nn.utils.clip_grad_norm_(     #“梯度裁剪”操作，强制将模型所有参数的总体梯度数值限制在一个设定的最大阈值内。
                self.model.parameters(),
                max_norm=self.config.MAX_GRAD_NORM
            )
            self.optimizer.step()               #正式执行了模型参数的更新与修改
            self.scheduler.step()               #动态调整了优化器下一步将要使用的学习率数值

            # 记录损失
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_cl_loss += loss_cl.item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'simcse': f'{loss_cl.item():.4f}'
            })

        epoch_time = time.time() - epoch_start_time

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_cl_loss = total_cl_loss / len(self.train_loader)

        return avg_loss, avg_cls_loss, avg_cl_loss, epoch_time

    def _compute_simcse_loss(self, z1, z2):
        """计算 SimCSE 对比损失（InfoNCE）"""
        """
        张量运算： z1 (形状 [32, 128]) 与转置后的 z2.T (形状 [128, 32]) 进行矩阵相乘。
        物理结果： 得到一个极其关键的二维张量 sim_matrix，形状为 [32, 32]。
        交叉点 (i, j)的数值： 代表第i句话的“影子1”与第j句话的“影子2”有多像（相似度得分）。
        在这个矩阵里，只有主对角线上的元素（例如 (0,0), (1,1), ..., (31,31)）互为真正的正样本对（Positive Pairs），
        因为它们来源于同一句话的不同 Dropout 状态。其余所有非对角线的位置，全都是不同句子拼凑的负样本（Negative Pairs）
        最后除以温度系数（一般数值很小），放大相似度差异
        """
        batch_size = z1.size(0) #size()：pytorch自带函数，提取tensor的维度，0表示获取第0维的长度
        device = z1.device

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=device)    #arange函数，生成数字list，device：指定生成设备
        #生成tensor([ 0,  1,  2,  3,  4, ..., 30, 31])
        """
        关于F.cross_entropy(input,target)损失函数的说明：
        首先将input（即logits值）进行归一化处理，一般为softmax
        之后根据PyTorch官方文档，该函数有两个功能，一个是熟悉的logits和ground_truth labels的匹配度
        另一个是indices索引功能，识别方法主要靠target的类型（float为概率预测 int为索引）
        在进行索引功能时，函数会根据indices的数值进行查找
        比如，若indices为[4,0,1],则在logits的矩阵中，第0行查找第4个，第1行找第0个，第2行找第1个值
        所以，若想找矩阵上对角线的值，则需生成等差数列[0,1,2...]，即使用torch.arange()函数
        随后，将根据indices查找出来的数值，进行负对数处理，之后求和相加，即为损失值
        """
        loss = F.cross_entropy(sim_matrix, labels)  #计算交叉熵损失函数

        return loss

    def evaluate(self, data_loader, dataset_name="验证集"):
        """评估模型性能"""
        self.model.eval()       #全局永久关闭所有的 Dropout 层
        all_preds = []          #建两个空list
        all_labels = []         #在下面的for循环中，一次只能处理batch_size数量的数据，多了会覆盖，所以需要储存的地方

        with torch.no_grad():   #所有缩进在这个 with 语句块内部的代码，都不会被记录到动态计算图中
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}", leave=False):
                input_ids = batch['input_ids'].to(self.config.DEVICE)   #将数据从计算机的CPU强制转移到了GPU中，下同
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)

                logits, _ = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())       #把 GPU 里算出的 PyTorch 张量，转移并拼接到了普通的 Python 列表里
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)    #计算准确率

        # 根据类别数选择 F1 计算模式
        if self.config.NUM_CLASSES > 2:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = f1_score(all_labels, all_preds, average='binary')

        return accuracy, f1

    def save_checkpoint(self, epoch, valid_acc, valid_f1, test_acc, test_f1, is_best=False, is_last=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'valid_accuracy': valid_acc,
            'valid_f1': valid_f1,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'best_valid_f1': self.best_valid_f1,
            'best_stats': self.best_stats,
            'alpha': self.alpha,
            'temperature': self.temperature
        }

        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存: {best_path}")

        if is_last and self.config.SAVE_LAST_EPOCH:
            last_path = os.path.join(self.config.CHECKPOINT_DIR, 'last_model.pt')
            torch.save(checkpoint, last_path)
            print(f"  ✓ 最后模型已保存: {last_path}")

    def train(self):
        """完整训练流程"""
        # --- [关键修改] 获取当前配置的语言名称 ---
        valid_lang_name = self.config.get_lang_name(self.config.VALID_LANG)
        test_lang_name = self.config.get_lang_name(self.config.TEST_LANG)

        print("\n" + "=" * 70)
        print("开始 SimCSE-ABSA 训练")
        print("=" * 70)
        print(f"SimCSE 策略: Dropout-based Unsupervised Contrastive Learning")
        print(f"损失权重配置 (Alpha={self.alpha}):")
        print(f"  分类损失: {(1 - self.alpha) * 100:.1f}%")
        print(f"  SimCSE损失: {self.alpha * 100:.1f}%")
        print("=" * 70)

        self.total_start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print('=' * 70)

            # 1. 训练
            avg_loss, avg_cls_loss, avg_cl_loss, epoch_time = self.train_epoch(epoch)
            self.epoch_times.append(epoch_time)

            # 2. 评估 (使用动态语言名称)
            print(f"\n在验证集({valid_lang_name})上评估...")
            eval_start = time.time()
            valid_acc, valid_f1 = self.evaluate(self.valid_loader, f"验证集({valid_lang_name})")

            print(f"在测试集({test_lang_name})上评估...")
            test_acc, test_f1 = self.evaluate(self.test_loader, f"测试集({test_lang_name})")
            eval_time = time.time() - eval_start

            # 3. 更新最佳指标
            # --- 验证集 ---
            if valid_acc > self.best_stats['valid']['acc']:
                self.best_stats['valid']['acc'] = valid_acc
                self.best_stats['valid']['acc_epoch'] = epoch + 1

            is_best_valid = False
            if valid_f1 > self.best_stats['valid']['f1']:
                self.best_stats['valid']['f1'] = valid_f1
                self.best_stats['valid']['f1_epoch'] = epoch + 1
                self.best_valid_f1 = valid_f1
                is_best_valid = True

            # --- 测试集 ---
            if test_acc > self.best_stats['test']['acc']:
                self.best_stats['test']['acc'] = test_acc
                self.best_stats['test']['acc_epoch'] = epoch + 1

            if test_f1 > self.best_stats['test']['f1']:
                self.best_stats['test']['f1'] = test_f1
                self.best_stats['test']['f1_epoch'] = epoch + 1

            # 4. 打印当前 Epoch 结果 (使用动态语言名称)
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1} 结果")
            print('=' * 70)
            print(f"训练损失: {avg_loss:.4f}")
            print(f"  分类损失: {avg_cls_loss:.4f}")
            print(f"  SimCSE损失: {avg_cl_loss:.4f}")

            print(f"\n验证集({valid_lang_name}):")
            print(f"  准确率: {valid_acc:.4f}")
            print(f"  F1分数: {valid_f1:.4f}")

            print(f"\n测试集({test_lang_name} - Zero-shot):")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  F1分数: {test_f1:.4f}")

            # 打印高光时刻
            markers = []
            if is_best_valid:
                markers.append("🏆 新的最佳验证F1")
            if test_acc == self.best_stats['test']['acc']:
                markers.append("⭐ 测试集Acc新高")
            if test_f1 == self.best_stats['test']['f1']:
                markers.append("⭐ 测试集F1新高")

            if markers:
                print(f"\n" + " | ".join(markers))

            # 5. 保存模型
            is_last_epoch = (epoch + 1) == self.config.NUM_EPOCHS
            if is_best_valid or is_last_epoch:
                print()
                self.save_checkpoint(
                    epoch, valid_acc, valid_f1, test_acc, test_f1,
                    is_best=is_best_valid,
                    is_last=is_last_epoch
                )

            print('=' * 70)

        # 训练结束后的总结报告 (使用动态语言名称)
        total_time = time.time() - self.total_start_time

        print("\n" + "=" * 70)
        print("SimCSE-ABSA 训练完成！全过程最佳指标报告")
        print("=" * 70)

        print(f"【验证集 ({valid_lang_name})】")
        print(f"  最高 Accuracy : {self.best_stats['valid']['acc']:.4f} (出现在 Epoch {self.best_stats['valid']['acc_epoch']})")
        print(f"  最高 F1 Score : {self.best_stats['valid']['f1']:.4f} (出现在 Epoch {self.best_stats['valid']['f1_epoch']})")

        print(f"\n【测试集 ({test_lang_name} - Zero-shot)】")
        print(f"  最高 Accuracy : {self.best_stats['test']['acc']:.4f} (出现在 Epoch {self.best_stats['test']['acc_epoch']})")
        print(f"  最高 F1 Score : {self.best_stats['test']['f1']:.4f} (出现在 Epoch {self.best_stats['test']['f1_epoch']})")

        print("-" * 70)
        print(f"  总训练时间: {timedelta(seconds=int(total_time))}")
        print(f"  模型保存路径: {self.config.CHECKPOINT_DIR}")
        print("=" * 70 + "\n")