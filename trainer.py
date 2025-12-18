"""
训练器模块 - SimCSE版本
核心变更：
1. 实现 SimCSE (同一输入两次forward，利用Dropout)
2. 优化 Checkpoint 保存策略（防止磁盘爆满）
3. 追踪所有历史最佳指标（验证集F1、测试集Acc、测试集F1）
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

        Args:
            model: SimCSE-ABSA 模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
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

        # 最佳指标追踪（完整版）
        self.best_valid_f1 = 0.0  # 最佳验证集F1
        self.best_valid_f1_test_acc = 0.0  # 最佳验证F1时的测试准确率
        self.best_valid_f1_test_f1 = 0.0  # 最佳验证F1时的测试F1

        # 新增：追踪测试集本身的历史最高分
        self.best_test_acc = 0.0  # 测试集历史最高准确率
        self.best_test_f1 = 0.0  # 测试集历史最高F1
        self.best_test_acc_epoch = 0  # 达到最高准确率的epoch
        self.best_test_f1_epoch = 0  # 达到最高F1的epoch

        # 时间统计
        self.epoch_times = []
        self.total_start_time = None

    def train_epoch(self, epoch):
        """
        训练一个epoch（SimCSE版本）

        核心：对每个样本进行两次forward，利用Dropout生成正样本对

        Args:
            epoch: 当前epoch编号

        Returns:
            (avg_loss, avg_cls_loss, avg_cl_loss, epoch_time)
        """
        self.model.train()  # 确保 Dropout 开启！
        total_loss = 0
        total_cls_loss = 0
        total_cl_loss = 0

        epoch_start_time = time.time()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"训练 Epoch {epoch + 1}/{self.config.NUM_EPOCHS}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.config.DEVICE)
            attention_mask = batch['attention_mask'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)

            # ===== SimCSE 核心：同一输入两次 forward =====
            # 第一次 forward（Dropout 状态1）
            logits_1, features_1 = self.model(input_ids, attention_mask)

            # 第二次 forward（Dropout 状态2，不同的随机掩码）
            logits_2, features_2 = self.model(input_ids, attention_mask)

            # ===== 计算分类损失（使用第一次的logits） =====
            loss_cls = self.cls_criterion(logits_1, labels)

            # ===== 计算 SimCSE 对比损失 =====
            # features_1 和 features_2 互为正样本对
            loss_cl = self._compute_simcse_loss(features_1, features_2)

            # ===== 归一化总损失 =====
            loss = (1.0 - self.alpha) * loss_cls + self.alpha * loss_cl

            # ===== 反向传播 =====
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.MAX_GRAD_NORM
            )
            self.optimizer.step()
            self.scheduler.step()

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
        """
        计算 SimCSE 对比损失（InfoNCE）

        对于同一个batch：
        - 正样本：(z1[i], z2[i]) - 同一个输入的两次forward
        - 负样本：batch内的其他样本

        Args:
            z1: 第一次forward的特征 [batch_size, projection_dim]
            z2: 第二次forward的特征 [batch_size, projection_dim]

        Returns:
            loss: SimCSE 对比损失
        """
        batch_size = z1.size(0)
        device = z1.device

        # 计算相似度矩阵
        # sim[i,j] = cosine_similarity(z1[i], z2[j]) / temperature
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 对角线元素是正样本对（z1[i] 和 z2[i] 来自同一输入）
        labels = torch.arange(batch_size, device=device)

        # InfoNCE 损失：让对角线元素的相似度最大
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def evaluate(self, data_loader, dataset_name="验证集"):
        """
        评估模型性能

        Args:
            data_loader: 数据加载器
            dataset_name: 数据集名称

        Returns:
            (accuracy, f1): 准确率和F1分数
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}", leave=False):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)

                # 只需要一次forward（评估时不需要Dropout多样性）
                logits, _ = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')

        return accuracy, f1

    def save_checkpoint(self, epoch, valid_acc, valid_f1, test_acc, test_f1, is_best=False, is_last=False):
        """
        保存模型检查点（优化的保存策略）

        Args:
            epoch: 当前epoch
            valid_acc: 验证集准确率
            valid_f1: 验证集F1
            test_acc: 测试集准确率
            test_f1: 测试集F1
            is_best: 是否为最佳验证F1模型
            is_last: 是否为最后一个epoch
        """
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
            'best_test_acc': self.best_test_acc,
            'best_test_f1': self.best_test_f1,
            'alpha': self.alpha,
            'temperature': self.temperature
        }

        # 只保存最佳模型（防止磁盘爆满）
        if is_best:
            best_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存: {best_path}")

        # 可选：保存最后一个epoch
        if is_last and self.config.SAVE_LAST_EPOCH:
            last_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                'last_model.pt'
            )
            torch.save(checkpoint, last_path)
            print(f"  ✓ 最后模型已保存: {last_path}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 70)
        print("开始 SimCSE-ABSA 训练")
        print("=" * 70)
        print(f"SimCSE 策略: Dropout-based Unsupervised Contrastive Learning")
        print(f"损失权重配置 (Alpha={self.alpha}):")
        print(f"  分类损失: {(1 - self.alpha) * 100:.1f}%")
        print(f"  SimCSE损失: {self.alpha * 100:.1f}%")
        print("=" * 70)

        # 开始计时
        self.total_start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print('=' * 70)

            # 训练一个epoch
            avg_loss, avg_cls_loss, avg_cl_loss, epoch_time = self.train_epoch(epoch)
            self.epoch_times.append(epoch_time)

            # 在验证集上评估
            print("\n在验证集(英语)上评估...")
            eval_start = time.time()
            valid_acc, valid_f1 = self.evaluate(self.valid_loader, "验证集")

            # 在测试集上评估
            print("在测试集(日语)上评估...")
            test_acc, test_f1 = self.evaluate(self.test_loader, "测试集")
            eval_time = time.time() - eval_start

            # ===== 更新所有历史最佳指标 =====
            # 1. 检查验证集F1是否为最佳
            is_best_valid = valid_f1 > self.best_valid_f1
            if is_best_valid:
                self.best_valid_f1 = valid_f1
                self.best_valid_f1_test_acc = test_acc
                self.best_valid_f1_test_f1 = test_f1

            # 2. 更新测试集历史最高准确率
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_test_acc_epoch = epoch + 1

            # 3. 更新测试集历史最高F1
            if test_f1 > self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_test_f1_epoch = epoch + 1

            # 时间统计
            elapsed_time = time.time() - self.total_start_time
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.config.NUM_EPOCHS - (epoch + 1)
            estimated_remaining = remaining_epochs * (avg_epoch_time + eval_time)

            # 打印结果
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1} 结果")
            print('=' * 70)
            print(f"训练损失: {avg_loss:.4f}")
            print(f"  分类损失: {avg_cls_loss:.4f} (权重: {1 - self.alpha:.3f})")
            print(f"  SimCSE损失: {avg_cl_loss:.4f} (权重: {self.alpha:.3f})")
            print(f"\n验证集(英语):")
            print(f"  准确率: {valid_acc:.4f}")
            print(f"  F1分数: {valid_f1:.4f}")
            print(f"\n测试集(日语 - Zero-shot):")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  F1分数: {test_f1:.4f}")

            # 标记新记录
            markers = []
            if is_best_valid:
                markers.append("🏆 新的最佳验证F1")
            if test_acc == self.best_test_acc:
                markers.append("⭐ 测试集历史最高准确率")
            if test_f1 == self.best_test_f1:
                markers.append("⭐ 测试集历史最高F1")

            if markers:
                print(f"\n" + " | ".join(markers))

            # 时间信息
            print(f"\n⏱️  时间统计:")
            print(f"  本轮训练: {timedelta(seconds=int(epoch_time))}")
            print(f"  本轮评估: {timedelta(seconds=int(eval_time))}")
            print(f"  已用时间: {timedelta(seconds=int(elapsed_time))}")
            print(f"  预计剩余: {timedelta(seconds=int(estimated_remaining))}")

            # 保存检查点
            is_last_epoch = (epoch + 1) == self.config.NUM_EPOCHS
            if is_best_valid or is_last_epoch:
                print()
                self.save_checkpoint(
                    epoch, valid_acc, valid_f1,
                    test_acc, test_f1,
                    is_best=is_best_valid,
                    is_last=is_last_epoch
                )

            print('=' * 70)

        # 总时间统计
        total_time = time.time() - self.total_start_time

        # ===== 完整的训练总结报告 =====
        print("\n" + "=" * 70)
        print("SimCSE-ABSA 训练完成！")
        print("=" * 70)

        print(f"\n📊 完整训练报告:")
        print("=" * 70)

        # 1. 验证集最佳F1及其对应的测试集性能
        print(f"\n[1] 基于验证集F1选择的最佳模型:")
        print(f"    最佳验证F1: {self.best_valid_f1:.4f}")
        print(f"    对应测试准确率: {self.best_valid_f1_test_acc:.4f}")
        print(f"    对应测试F1: {self.best_valid_f1_test_f1:.4f}")

        # 2. 测试集历史最高准确率
        print(f"\n[2] 测试集历史最高准确率:")
        print(f"    Best Test Accuracy: {self.best_test_acc:.4f}")
        print(f"    达到于 Epoch {self.best_test_acc_epoch}")

        # 3. 测试集历史最高F1
        print(f"\n[3] 测试集历史最高F1:")
        print(f"    Best Test F1: {self.best_test_f1:.4f}")
        print(f"    达到于 Epoch {self.best_test_f1_epoch}")

        # 4. 时间统计
        print(f"\n⏱️  训练时间统计:")
        print(f"    总训练时间: {timedelta(seconds=int(total_time))}")
        print(f"    平均每轮: {timedelta(seconds=int(total_time / self.config.NUM_EPOCHS))}")

        print("=" * 70)

        # 提示信息
        print(f"\n💡 说明:")
        print(f"  - [1] 是保存的 best_model.pt 的性能")
        print(f"  - [2][3] 是测试集在整个训练过程中达到的峰值")
        print(f"  - 如果 [2][3] 明显高于 [1]，说明存在过拟合")
        print("=" * 70 + "\n")