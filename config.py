"""
配置文件 - 存储所有超参数和路径配置
修改版：支持使用小规模数据集进行快速测试
"""
import torch
import os


class Config:
    """全局配置类"""

    # ============ 路径配置 ============
    # 数据路径 - 修改为三个文件
    TRAIN_DATA_PATH = './dataset/Amazon/train_original.csv'
    VALID_DATA_PATH = './dataset/Amazon/validation_original.csv'
    TEST_DATA_PATH = './dataset/Amazon/test_original.csv'

    # 模型路径（请修改为你的本地路径）
    MODEL_PATH = './Xlm-roberta-base'

    # 输出路径
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

    # 翻译模型配置
    TRANSLATION_MODEL_PATH = './dataset/translation_module/Helsinki-NLPopus-mt-en-jap'
    TRANSLATION_BATCH_SIZE = 4  # 翻译时的批次大小
    TRANSLATION_MAX_LENGTH = 512  # 翻译的最大长度

    # 翻译缓存（如果有预翻译的数据）
    USE_CACHED_TRANSLATION = False
    CACHED_TRANSLATION_FILE = None  # 例如: 'translated_data.csv'

    # ============ 数据采样配置（新增！） ============
    # 是否使用小规模数据进行快速测试
    USE_SMALL_DATASET = True  # 改为 False 使用全部数据

    # 每个数据集使用的样本数量（仅当 USE_SMALL_DATASET=True 时生效）
    TRAIN_SAMPLE_SIZE = 500  # 训练集使用200条
    VALID_SAMPLE_SIZE = 100  # 验证集使用20条
    TEST_SAMPLE_SIZE = 100  # 测试集使用20条

    # 是否进行分层采样（保持正负样本比例）
    STRATIFIED_SAMPLING = True

    # ============ 数据处理参数 ============
    # TF-IDF掩码比例
    TOP_TFIDF_RATIO = 0.15  # 保留Top 15%的高TF-IDF词

    # 序列最大长度
    MAX_LENGTH = 128

    # ============ 模型参数 ============
    # 分类类别数
    NUM_CLASSES = 2  # 二分类：正面/负面

    # 对比学习投影维度
    PROJECTION_DIM = 128

    # ============ 训练参数 ============
    # 批次大小
    BATCH_SIZE = 16

    # 训练轮数
    NUM_EPOCHS = 5  # 如果使用小数据集，可以减少到2-3轮

    # 学习率
    LEARNING_RATE = 2e-5

    # 对比学习损失权重
    LAMBDA_CL = 0.5

    # 温度参数（用于InfoNCE）
    TEMPERATURE = 0.07

    # 梯度裁剪
    MAX_GRAD_NORM = 1.0

    # 学习率预热比例
    WARMUP_RATIO = 0.1

    # ============ 其他配置 ============
    # 随机种子
    SEED = 42

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存模型的频率（每N个epoch）
    SAVE_FREQ = 1

    # 日志打印频率（每N个batch）
    LOG_FREQ = 10  # 小数据集时减少打印频率

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "=" * 50)
        print("当前配置")
        print("=" * 50)
        if cls.USE_SMALL_DATASET:
            print("⚠️  使用小规模数据集测试模式")
            print(f"   训练集: {cls.TRAIN_SAMPLE_SIZE} 条")
            print(f"   验证集: {cls.VALID_SAMPLE_SIZE} 条")
            print(f"   测试集: {cls.TEST_SAMPLE_SIZE} 条")
        else:
            print("✓ 使用完整数据集")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"设备: {cls.DEVICE}")
        print("=" * 50 + "\n")