"""
主程序入口 - SimCSE版本
运行方式: python main.py

SimCSE-ABSA: SimCSE-based Cross-lingual ABSA
核心改进：
1. SimCSE (Dropout-based Unsupervised Contrastive Learning)
2. 无需翻译数据，避免翻译噪声
3. Attention Pooling 提取情感特征
"""
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from config import Config
from utils import set_seed, save_config, print_model_info, get_timestamp
from data_preprocessing import DataPreprocessor
from dataset import CrossLingualDataset
from model import XLMSentimentModel
from trainer import Trainer


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("SimCSE-ABSA: SimCSE-based Cross-lingual ABSA")
    print("跨语言情感分析项目 - 英语 -> 日语情感迁移")
    print("=" * 70)
    print("\n核心改进:")
    print("  [1] SimCSE - Dropout-based Contrastive Learning")
    print("  [2] 无需翻译数据 - 避免翻译噪声")
    print("  [3] Attention Pooling - 捕捉情感关键词")
    print("=" * 70)

    # 创建必要的目录
    config = Config()
    config.create_dirs()

    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n随机种子: {config.SEED}")
    print(f"使用设备: {config.DEVICE}")

    # 打印配置信息
    config.print_config()

    # 打印性能优化建议
    tips = config.get_performance_tips()
    if tips:
        print("=" * 70)
        print("💡 性能优化建议:")
        print("=" * 70)
        for tip in tips:
            print(f"  {tip}")
        print("=" * 70 + "\n")

    # 保存配置
    timestamp = get_timestamp()
    config_save_path = f"{config.LOG_DIR}/config_{timestamp}.json"
    save_config(config, config_save_path)

    # ========== 步骤1: 数据预处理 ==========
    print("\n" + "=" * 70)
    print("步骤1: 数据预处理 (SimCSE模式)")
    print("=" * 70)

    preprocessor = DataPreprocessor(config)
    train_data, valid_data, test_data = preprocessor.prepare_datasets()

    # ========== 步骤2: 加载Tokenizer ==========
    print("=" * 70)
    print("步骤2: 加载Tokenizer")
    print("=" * 70)

    tokenizer = XLMRobertaTokenizer.from_pretrained(config.MODEL_PATH)
    print("✓ Tokenizer加载完成\n")

    # ========== 步骤3: 创建数据集和数据加载器 ==========
    print("=" * 70)
    print("步骤3: 创建数据加载器")
    print("=" * 70)

    # 训练集
    train_dataset = CrossLingualDataset(
        train_data,
        tokenizer,
        config.MAX_LENGTH
    )

    # DataLoader配置
    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY and config.DEVICE.type == 'cuda',
    }

    if config.NUM_WORKERS > 0 and config.PERSISTENT_WORKERS:
        dataloader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )

    # 验证集
    valid_dataset = CrossLingualDataset(
        valid_data,
        tokenizer,
        config.MAX_LENGTH
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    # 测试集（日语）
    test_dataset = CrossLingualDataset(
        test_data,
        tokenizer,
        config.MAX_LENGTH
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    print(f"✓ 训练批次数: {len(train_loader)}")
    print(f"✓ 验证批次数: {len(valid_loader)}")
    print(f"✓ 测试批次数: {len(test_loader)}")
    print(f"✓ DataLoader配置: workers={config.NUM_WORKERS}, "
          f"pin_memory={config.PIN_MEMORY and config.DEVICE.type == 'cuda'}")
    print()

    # ========== 步骤4: 初始化SimCSE-ABSA模型 ==========
    print("=" * 70)
    print("步骤4: 初始化 SimCSE-ABSA 模型")
    print("=" * 70)

    model = XLMSentimentModel(
        model_path=config.MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        projection_dim=config.PROJECTION_DIM,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    print_model_info(model)

    # ========== 步骤5: 训练 ==========
    print("=" * 70)
    print("步骤5: 开始训练")
    print("=" * 70)

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        test_loader,
        config
    )
    trainer.train()

    print("\n" + "=" * 70)
    print("SimCSE-ABSA 项目运行完成！")
    print("=" * 70)
    print(f"\n检查点保存位置: {config.CHECKPOINT_DIR}")
    print(f"日志保存位置: {config.LOG_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()