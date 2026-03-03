"""
主要作用：
将数据从csv导出到python中（转化为dataframe（df）格式）
根据需求对数据进行筛选和删除
打包成tensor格式的文件，以便其他部分读取和处理

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """数据预处理类 - SimCSE版本"""

    def __init__(self, config):
        """
        初始化数据预处理器

        Args:
            config: 配置对象
        """
        self.config = config

    def load_and_filter_data(self):
        """
        加载并筛选数据（从train/valid/test三个文件）

        Returns:
            train_en_df: 训练集英语数据
            valid_en_df: 验证集英语数据
            test_ja_df: 测试集日语数据
        """
        print("=" * 50)
        print("步骤1: 加载数据")
        print("=" * 50)

        # 读取训练集，        注：df为DataFrame格式，类似一个python内部的大型Excel
        print(f"\n读取训练集: {self.config.TRAIN_DATA_PATH}")
        if not os.path.exists(self.config.TRAIN_DATA_PATH):
            raise FileNotFoundError(f"训练集文件不存在: {self.config.TRAIN_DATA_PATH}")
        train_df = pd.read_csv(self.config.TRAIN_DATA_PATH) #通过panda库将csv文件转换为py可读的df文件
        print(f"  ✓ 原始训练数据: {len(train_df)} 条")

        # 读取验证集
        print(f"\n读取验证集: {self.config.VALID_DATA_PATH}")
        if not os.path.exists(self.config.VALID_DATA_PATH):
            raise FileNotFoundError(f"验证集文件不存在: {self.config.VALID_DATA_PATH}")
        valid_df = pd.read_csv(self.config.VALID_DATA_PATH)
        print(f"  ✓ 原始验证数据: {len(valid_df)} 条")

        # 读取测试集
        print(f"\n读取测试集: {self.config.TEST_DATA_PATH}")
        if not os.path.exists(self.config.TEST_DATA_PATH):
            raise FileNotFoundError(f"测试集文件不存在: {self.config.TEST_DATA_PATH}")
        test_df = pd.read_csv(self.config.TEST_DATA_PATH)
        print(f"  ✓ 原始测试数据: {len(test_df)} 条")

        # 检查必需的列是否存在
        required_columns = ['stars', 'review_body', 'language']
        for df_name, df in [('训练集', train_df), ('验证集', valid_df), ('测试集', test_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{df_name}缺少必需的列: {missing_cols}")

        if self.config.USE_THREE_CLASSES:
            # 三分类逻辑
            print(">>> 启用三分类模式: 保留3星数据 (Label: 0=负面, 1=中性, 2=正面)")

            #将star转换为数字
            def convert_label(stars):
                if stars in [1, 2]:
                    return 0  # Negative
                elif stars == 3:
                    return 1  # Neutral (新增)
                elif stars in [4, 5]:
                    return 2  # Positive (原为1，现改为2)
                else:
                    return -1
        else:
            # 二分类逻辑 (原有逻辑)
            print(">>> 启用二分类模式: 丢弃3星数据 (Label: 0=负面, 1=正面)")
            def convert_label(stars):
                if stars in [1, 2]:
                    return 0  # Negative
                elif stars in [4, 5]:
                    return 1  # Positive
                else:
                    return -1  # 3星及其他丢弃

        # 获取语言显示名称
        train_lang_name = self.config.get_lang_name(self.config.TRAIN_LANG)
        valid_lang_name = self.config.get_lang_name(self.config.VALID_LANG)
        test_lang_name = self.config.get_lang_name(self.config.TEST_LANG)

        print("\n" + "=" * 50)
        print(f"步骤1.1: 处理训练集 (筛选{train_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        train_df = train_df[train_df['language'] == self.config.TRAIN_LANG].copy()
        #[train_df['language'] == self.config.TRAIN_LANG]：判断df中lang列中的值是否与config设置相同，同为True不同为False
        #train_df[上述代码]:保留[]内为True的行，删去False的行
        #.copy()的作用：开辟新的储存空间，储存处理后的df文件，并删掉原先的df文件

        train_df['label'] = train_df['stars'].apply(convert_label)
        train_df = train_df[train_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{train_lang_name}并转换标签后: {len(train_df)} 条")

        print("\n" + "=" * 50)
        print(f"步骤1.2: 处理验证集 (筛选{valid_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        valid_df = valid_df[valid_df['language'] == self.config.VALID_LANG].copy()

        valid_df['label'] = valid_df['stars'].apply(convert_label)
        valid_df = valid_df[valid_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{valid_lang_name}并转换标签后: {len(valid_df)} 条")

        print("\n" + "=" * 50)
        print(f"步骤1.3: 处理测试集 (筛选{test_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        test_df = test_df[test_df['language'] == self.config.TEST_LANG].copy()

        test_df['label'] = test_df['stars'].apply(convert_label)
        test_df = test_df[test_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{test_lang_name}并转换标签后: {len(test_df)} 条")

        # 如果使用小规模数据集，进行采样
        if self.config.USE_SMALL_DATASET:
            print("\n" + "=" * 50)
            print("⚠️  启用小规模数据集模式")
            print("=" * 50)

            train_df = self._sample_data(
                train_df,
                self.config.TRAIN_SAMPLE_SIZE,
                "训练集"
            )
            valid_df = self._sample_data(
                valid_df,
                self.config.VALID_SAMPLE_SIZE,
                "验证集"
            )
            test_df = self._sample_data(
                test_df,
                self.config.TEST_SAMPLE_SIZE,
                "测试集"
            )

        print("\n" + "=" * 50)
        print("数据加载完成！")
        print("=" * 50)
        print(f"✓ 最终训练集: {len(train_df)} 条")
        print(f"✓ 最终验证集: {len(valid_df)} 条")
        print(f"✓ 最终测试集: {len(test_df)} 条")
        print("=" * 50 + "\n")

        return train_df, valid_df, test_df

    def _sample_data(self, df, sample_size, dataset_name):
        """
        从数据框中采样指定数量的数据

        Args:
            df: 数据框
            sample_size: 采样数量
            dataset_name: 数据集名称（用于日志）

        Returns:
            采样后的数据框
        """
        if len(df) <= sample_size:
            print(f"\n{dataset_name}:")
            print(f"  数据量({len(df)})小于采样量({sample_size})，使用全部数据")
            label_counts = df['label'].value_counts()
            print(f"  标签分布 - 负面: {label_counts.get(0, 0)}, 正面: {label_counts.get(1, 0)}")
            return df

        if self.config.STRATIFIED_SAMPLING:
            # 分层采样，保持正负样本比例
            try:
                sampled_df, _ = train_test_split(
                    df,
                    train_size=sample_size,
                    stratify=df['label'],
                    random_state=self.config.SEED
                )
                print(f"\n{dataset_name}:")
                print(f"  ✓ 分层采样 {sample_size} 条（保持正负比例）")
            except Exception as e:
                # 如果分层采样失败（比如某类样本太少），使用随机采样
                sampled_df = df.sample(
                    n=sample_size,
                    random_state=self.config.SEED
                ).reset_index(drop=True)
                print(f"\n{dataset_name}:")
                print(f"  ⚠ 分层采样失败，使用随机采样 {sample_size} 条")
                print(f"  失败原因: {e}")
        else:
            # 随机采样
            sampled_df = df.sample(
                n=sample_size,
                random_state=self.config.SEED
            ).reset_index(drop=True)
            print(f"\n{dataset_name}:")
            print(f"  ✓ 随机采样 {sample_size} 条")

        # 打印标签分布
        label_counts = sampled_df['label'].value_counts()
        print(f"  标签分布 - 负面: {label_counts.get(0, 0)}, 正面: {label_counts.get(1, 0)}")

        return sampled_df

    def prepare_datasets(self):
        """
        准备训练集、验证集和测试集（SimCSE版本：无需翻译）

        Returns:
            train_data: 训练数据字典
            valid_data: 验证数据字典
            test_data: 测试数据字典
        """
        # 加载和筛选数据
        train_df, valid_df, test_df = self.load_and_filter_data()

        print("\n" + "=" * 50)
        print("步骤2: 准备数据集 (SimCSE模式)")
        print("=" * 50)
        print("✓ SimCSE 不需要翻译数据")
        print("✓ 仅使用原始文本 + Dropout 生成正样本对")
        print("=" * 50 + "\n")

        # 构建训练数据（SimCSE：只需要原始文本）
        train_data = {
            'text': train_df['review_body'].tolist(),   #tolist()使df格式中的某一列强行提取出来变为list格式，并保存在字典中
            'label': train_df['label'].tolist()
        }
        #于是根据df文件生成了一个包含两个list的字典

        # 构建验证数据
        valid_data = {
            'text': valid_df['review_body'].tolist(),
            'label': valid_df['label'].tolist()
        }

        # 构建测试数据（日语数据 - Zero-shot测试）
        test_data = {
            'text': test_df['review_body'].tolist(),
            'label': test_df['label'].tolist()
        }

        print("=" * 50)
        print("数据准备完成！")
        print("=" * 50)
        print(f"训练集大小: {len(train_data['label'])} 条")
        print(f"验证集大小: {len(valid_data['label'])} 条")
        print(f"测试集大小: {len(test_data['label'])} 条")
        print("=" * 50 + "\n")

        return train_data, valid_data, test_data