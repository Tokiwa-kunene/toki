"""
主程序入口
运行方式: python main.py
修改版：支持训练集、验证集、测试集三个数据集
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
    print("\n" + "=" * 50)
    print("跨语言情感分析项目")
    print("英语 -> 日语情感迁移")
    print("=" * 50)

    # 创建必要的目录
    config = Config()
    config.create_dirs()

    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n随机种子: {config.SEED}")
    print(f"使用设备: {config.DEVICE}")

    # 打印配置信息
    config.print_config()

    # 保存配置
    timestamp = get_timestamp()
    config_save_path = f"{config.LOG_DIR}/config_{timestamp}.json"
    save_config(config, config_save_path)

    # ========== 步骤1: 数据预处理 ==========
    print("\n" + "=" * 50)
    print("开始数据预处理...")
    print("=" * 50)

    preprocessor = DataPreprocessor(config)
    train_data, valid_data, test_data = preprocessor.prepare_datasets()

    # ========== 步骤2: 加载Tokenizer ==========
    print("=" * 50)
    print("加载Tokenizer...")
    print("=" * 50)

    tokenizer = XLMRobertaTokenizer.from_pretrained(config.MODEL_PATH)
    print("Tokenizer加载完成\n")

    # ========== 步骤3: 创建数据集和数据加载器 ==========
    print("=" * 50)
    print("创建数据加载器...")
    print("=" * 50)

    # 训练集
    train_dataset = CrossLingualDataset(
        train_data,
        tokenizer,
        config.MAX_LENGTH,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # 验证集
    valid_dataset = CrossLingualDataset(
        valid_data,
        tokenizer,
        config.MAX_LENGTH,
        is_train=True  # 验证集也需要三元组数据
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 测试集（日语）
    test_dataset = CrossLingualDataset(
        test_data,
        tokenizer,
        config.MAX_LENGTH,
        is_train=False  # 测试集只需要单个文本
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(valid_loader)}")
    print(f"测试批次数: {len(test_loader)}\n")

    # ========== 步骤4: 初始化模型 ==========
    print("=" * 50)
    print("初始化模型...")
    print("=" * 50)

    model = XLMSentimentModel(
        model_path=config.MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        projection_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)

    print_model_info(model)

    # ========== 步骤5: 训练 ==========
    trainer = Trainer(model, train_loader, valid_loader, test_loader, config)
    trainer.train()

    print("\n" + "=" * 50)
    print("项目运行完成！")
    print("=" * 50)
    print(f"\n检查点保存位置: {config.CHECKPOINT_DIR}")
    print(f"日志保存位置: {config.LOG_DIR}")


if __name__ == "__main__":
    main()