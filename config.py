"""
配置文件 - SimCSE版本
存储所有超参数和路径配置
核心变更：移除翻译相关配置，切换到 SimCSE (Dropout-based Contrastive Learning)
"""
import torch
import os


class Config:
    """全局配置类"""

    # ============ 路径配置 ============
    # 数据路径
    TRAIN_DATA_PATH = './dataset/Amazon/train.csv'
    VALID_DATA_PATH = './dataset/Amazon/validation.csv'
    TEST_DATA_PATH = './dataset/Amazon/test.csv'

    # 模型路径
    MODEL_PATH = './Xlm-roberta-base'

    # 输出路径
    OUTPUT_DIR = '/kaggle/working/'
    #/kaggle/working/
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

    # ============ 数据采样配置 ============
    # 是否使用小规模数据进行快速测试
    USE_SMALL_DATASET = True

    # 每个数据集使用的样本数量
    TRAIN_SAMPLE_SIZE = 10000
    VALID_SAMPLE_SIZE = 2000
    TEST_SAMPLE_SIZE = 2000

    # 是否进行分层采样
    STRATIFIED_SAMPLING = True

    # ============ 数据处理参数 ============
    MAX_LENGTH = 128

    # ============ 模型参数 ============
    NUM_CLASSES = 2  # 二分类：正面/负面
    PROJECTION_DIM = 128  # 对比学习投影维度

    # ============ SimCSE 关键配置 ============
    # Dropout 是 SimCSE 的核心！利用 Dropout 生成正样本对
    DROPOUT_RATE = 0.1
    # XLM-RoBERTa 默认 dropout，SimCSE 依赖此设置

    # ============ 训练参数 ============
    BATCH_SIZE = 32
    NUM_EPOCHS = 5  # SimCSE 通常收敛较快
    LEARNING_RATE = 2e-5

    # ============ 损失函数权重配置 ============
    # 对比学习损失的权重系数 alpha ∈ [0, 1]
    ALPHA = 0.3

    # 温度参数（用于对比学习）
    TEMPERATURE = 0.07

    # 梯度裁剪
    MAX_GRAD_NORM = 1.0

    # 学习率预热比例
    WARMUP_RATIO = 0.1

    # ============ 性能优化配置 ============
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False

    # ============ Checkpoint 保存策略 ============
    # 重要变更：防止磁盘爆满
    SAVE_FREQ = 999  # 设置为很大的数，实际上只在 is_best 时保存
    SAVE_ONLY_BEST = True  # 只保存最佳模型
    SAVE_LAST_EPOCH = True  # 是否保存最后一个 epoch

    # ============ 其他配置 ============
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG_FREQ = 10

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "=" * 70)
        print("SimCSE-ABSA 配置")
        print("=" * 70)

        # 数据集配置
        if cls.USE_SMALL_DATASET:
            print("⚠️  使用小规模数据集测试模式")
            print(f"   训练集: {cls.TRAIN_SAMPLE_SIZE} 条")
            print(f"   验证集: {cls.VALID_SAMPLE_SIZE} 条")
            print(f"   测试集: {cls.TEST_SAMPLE_SIZE} 条")
        else:
            print("✓ 使用完整数据集")

        # 训练配置
        print(f"\n训练参数:")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  训练轮数: {cls.NUM_EPOCHS}")
        print(f"  学习率: {cls.LEARNING_RATE}")

        # SimCSE 特定配置
        print(f"\nSimCSE 配置:")
        print(f"  策略: Unsupervised Contrastive Learning with Dropout")
        print(f"  Dropout Rate: {cls.DROPOUT_RATE}")
        print(f"  → 同一输入两次Forward，利用Dropout生成正样本对")
        print(f"  → 不需要翻译数据，避免翻译噪声")

        # 损失权重
        print(f"\n损失权重 Alpha: {cls.ALPHA}")
        print(f"  → 分类损失: {(1-cls.ALPHA)*100:.1f}%")
        print(f"  → SimCSE Loss: {cls.ALPHA*100:.1f}%")

        # Checkpoint 策略
        print(f"\nCheckpoint 保存策略:")
        if cls.SAVE_ONLY_BEST:
            print(f"  → 仅保存最佳模型 (best_model.pt)")
        else:
            print(f"  → 每 {cls.SAVE_FREQ} 个epoch保存一次")
        if cls.SAVE_LAST_EPOCH:
            print(f"  → 保存最后一个epoch (last_model.pt)")

        # 性能配置
        print(f"\n性能配置:")
        print(f"  设备: {cls.DEVICE}")
        print(f"  num_workers: {cls.NUM_WORKERS}")
        print(f"  pin_memory: {cls.PIN_MEMORY}")

        print("=" * 70 + "\n")

    @classmethod
    def get_performance_tips(cls):
        """获取性能优化建议"""
        tips = []

        if cls.NUM_WORKERS == 0:
            tips.append("🚀 将 NUM_WORKERS 设置为 4 可以提速 50%+")

        if cls.BATCH_SIZE < 32:
            tips.append("🚀 将 BATCH_SIZE 增大到 32 可以提速 30%+")

        if cls.USE_SMALL_DATASET and cls.TRAIN_SAMPLE_SIZE > 300:
            tips.append("⚡ 测试时可将样本数减少到 200 以加快速度")

        if cls.PIN_MEMORY and cls.DEVICE.type == 'cpu':
            tips.append("⚠️  CPU训练时建议将 PIN_MEMORY 设为 False")

        return tips