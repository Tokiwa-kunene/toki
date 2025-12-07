"""
训练器模块
负责模型训练和评估
修改版：支持训练集、验证集、测试集
实现三元对比学习：原始英语 vs TF-IDF掩码 vs 日语翻译
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os


class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        """
        初始化训练器

        Args:
            model: 模型
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
        # 注意：不再使用单独的InfoNCE类，直接在训练中实现三元对比
        self.temperature = config.TEMPERATURE

        # 最佳指标
        self.best_valid_f1 = 0.0
        self.best_test_f1 = 0.0

    def train_epoch(self, epoch):
        """
        训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            平均损失元组 (总损失, 分类损失, 对比损失)
        """
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_cl_loss = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"训练 Epoch {epoch + 1}/{self.config.NUM_EPOCHS}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            original_input_ids = batch['original_input_ids'].to(self.config.DEVICE)
            original_attention_mask = batch['original_attention_mask'].to(self.config.DEVICE)
            masked_input_ids = batch['masked_input_ids'].to(self.config.DEVICE)
            masked_attention_mask = batch['masked_attention_mask'].to(self.config.DEVICE)
            translated_input_ids = batch['translated_input_ids'].to(self.config.DEVICE)
            translated_attention_mask = batch['translated_attention_mask'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)

            # 前向传播 - 三个分支
            logits_orig, features_orig = self.model(
                original_input_ids,
                original_attention_mask
            )
            _, features_masked = self.model(
                masked_input_ids,
                masked_attention_mask
            )
            _, features_translated = self.model(
                translated_input_ids,
                translated_attention_mask
            )

            # 计算分类损失
            loss_cls = self.cls_criterion(logits_orig, labels)

            # 计算三元对比学习损失
            # 实现真正的三元对比：anchor (原始) vs positive1 (掩码) vs positive2 (翻译)
            loss_cl = self._compute_triplet_contrastive_loss(
                features_orig,  # anchor: 原始英语
                features_masked,  # positive1: TF-IDF掩码英语
                features_translated  # positive2: 日语翻译
            )

            # 总损失
            loss = loss_cls + self.config.LAMBDA_CL * loss_cl

            # 反向传播
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
                'cl': f'{loss_cl.item():.4f}'
            })

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_cl_loss = total_cl_loss / len(self.train_loader)

        return avg_loss, avg_cls_loss, avg_cl_loss

    def _compute_triplet_contrastive_loss(self, anchor, positive1, positive2):
        """
        计算三元对比学习损失

        核心思想：
        - anchor: 原始英语样本的特征表示
        - positive1: TF-IDF掩码增强的英语样本特征（augmented positive）
        - positive2: 翻译成日语的样本特征（cross-lingual positive）

        目标：
        1. 拉近 anchor 与 positive1 的距离（同语言增强）
        2. 拉近 anchor 与 positive2 的距离（跨语言对齐）
        3. 推远 anchor 与其他批次样本的距离（负样本）

        Args:
            anchor: [batch_size, projection_dim] 原始英语特征
            positive1: [batch_size, projection_dim] 掩码英语特征
            positive2: [batch_size, projection_dim] 日语翻译特征

        Returns:
            contrastive_loss: 对比学习损失值
        """
        batch_size = anchor.size(0)
        device = anchor.device

        # ========== 方法1: 分别计算两个正例的对比损失（原实现） ==========
        # 这种方法将两种正例分开处理

        # anchor vs positive1 (TF-IDF掩码)
        # 计算相似度矩阵: [batch_size, batch_size]
        # sim[i,j] 表示第i个anchor与第j个positive1的相似度
        sim_anchor_pos1 = torch.matmul(anchor, positive1.T) / self.temperature

        # anchor vs positive2 (日语翻译)
        sim_anchor_pos2 = torch.matmul(anchor, positive2.T) / self.temperature

        # 对角线元素是正样本对，非对角线是负样本对
        # 例如：sim[0,0]是anchor[0]和positive1[0]的相似度（正样本）
        #      sim[0,1]是anchor[0]和positive1[1]的相似度（负样本）
        labels = torch.arange(batch_size, device=device)

        # 使用交叉熵损失，目标是让对角线元素的相似度最大
        loss1 = F.cross_entropy(sim_anchor_pos1, labels)
        loss2 = F.cross_entropy(sim_anchor_pos2, labels)

        # 方法1总损失
        loss_method1 = (loss1 + loss2) / 2

        # ========== 方法2: 真正的三元对比（推荐） ==========
        # 这种方法在同一个对比空间中处理两种正例

        # 构建增强的正例矩阵：将positive1和positive2拼接
        # positives: [batch_size, 2*projection_dim]
        # 对于第i个样本，它有两个正例：positive1[i]和positive2[i]
        positives = torch.cat([positive1, positive2], dim=0)  # [2*batch_size, projection_dim]

        # 计算anchor与所有正例的相似度
        # sim_matrix: [batch_size, 2*batch_size]
        # sim_matrix[i, j] 表示anchor[i]与第j个正例的相似度
        sim_matrix = torch.matmul(anchor, positives.T) / self.temperature

        # 构建标签：每个anchor有两个正例
        # 对于anchor[i]，它的正例是positive1[i](索引i)和positive2[i](索引i+batch_size)
        pos_mask = torch.zeros(batch_size, 2 * batch_size, device=device)
        for i in range(batch_size):
            pos_mask[i, i] = 1  # positive1[i]是正例
            pos_mask[i, i + batch_size] = 1  # positive2[i]是正例

        # 计算三元对比损失
        # 对每个anchor，计算它与所有样本的对比损失
        # 正例(positive1和positive2)的相似度应该高，负例应该低

        # exp(sim)用于softmax计算
        exp_sim = torch.exp(sim_matrix)

        # 分子：正例的相似度之和
        pos_sim = (exp_sim * pos_mask).sum(dim=1)

        # 分母：所有样本的相似度之和
        all_sim = exp_sim.sum(dim=1)

        # 对比损失：-log(正例相似度/所有相似度)
        # 目标是最大化正例相似度占比
        loss_method2 = -torch.log(pos_sim / all_sim).mean()

        # ========== 选择使用哪种方法 ==========
        # 可以通过配置选择，这里默认使用方法2（真正的三元对比）
        use_method2 = True  # 改为False可以使用原来的方法1

        if use_method2:
            return loss_method2
        else:
            return loss_method1

    def evaluate_with_contrastive(self, data_loader, dataset_name="验证集"):
        """
        在有对比学习数据的数据集上评估（训练集/验证集）

        Args:
            data_loader: 数据加载器
            dataset_name: 数据集名称

        Returns:
            准确率和F1分数
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}"):
                # 只使用原始文本进行预测
                original_input_ids = batch['original_input_ids'].to(self.config.DEVICE)
                original_attention_mask = batch['original_attention_mask'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)

                # 前向传播
                logits, _ = self.model(original_input_ids, original_attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')

        return accuracy, f1

    def evaluate_test(self):
        """
        在测试集（日语）上评估

        Returns:
            准确率和F1分数
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="评估测试集(日语)"):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)

                # 前向传播
                logits, _ = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')

        return accuracy, f1

    def save_checkpoint(self, epoch, valid_acc, valid_f1, test_acc, test_f1, is_best=False):
        """
        保存模型检查点

        Args:
            epoch: 当前epoch
            valid_acc: 验证集准确率
            valid_f1: 验证集F1
            test_acc: 测试集准确率
            test_f1: 测试集F1
            is_best: 是否为最佳模型
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
            'best_test_f1': self.best_test_f1
        }

        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'checkpoint_epoch_{epoch + 1}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"  检查点已保存: {checkpoint_path}")

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存: {best_path}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("开始训练...")
        print("=" * 50)

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print('=' * 50)

            # 训练一个epoch
            avg_loss, avg_cls_loss, avg_cl_loss = self.train_epoch(epoch)

            # 在验证集上评估
            print("\n在验证集(英语)上评估...")
            valid_acc, valid_f1 = self.evaluate_with_contrastive(
                self.valid_loader,
                "验证集"
            )

            # 在测试集上评估
            print("\n在测试集(日语)上评估...")
            test_acc, test_f1 = self.evaluate_test()

            # 打印结果
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1} 结果")
            print('=' * 50)
            print(f"训练损失: {avg_loss:.4f} "
                  f"(分类: {avg_cls_loss:.4f}, 对比: {avg_cl_loss:.4f})")
            print(f"\n验证集(英语):")
            print(f"  准确率: {valid_acc:.4f}")
            print(f"  F1分数: {valid_f1:.4f}")
            print(f"\n测试集(日语 - Zero-shot):")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  F1分数: {test_f1:.4f}")

            # 检查是否为最佳模型（基于验证集F1）
            is_best = valid_f1 > self.best_valid_f1
            if is_best:
                self.best_valid_f1 = valid_f1
                self.best_test_f1 = test_f1
                print(f"\n✓ 新的最佳验证集F1: {self.best_valid_f1:.4f}")

            # 保存检查点
            if (epoch + 1) % self.config.SAVE_FREQ == 0 or is_best:
                print()
                self.save_checkpoint(
                    epoch, valid_acc, valid_f1,
                    test_acc, test_f1, is_best
                )

            print('=' * 50)

        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        print(f"最佳验证集F1: {self.best_valid_f1:.4f}")
        print(f"对应的测试集F1: {self.best_test_f1:.4f}")
        print("=" * 50 + "\n")