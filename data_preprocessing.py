"""
数据预处理模块
负责数据加载、筛选、TF-IDF掩码和翻译
完全修正版：正确使用三个数据文件路径
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, config):
        """
        初始化数据预处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.mask_token = '<mask>'
        self.tfidf_vectorizer = None

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

        # 读取训练集
        print(f"\n读取训练集: {self.config.TRAIN_DATA_PATH}")
        if not os.path.exists(self.config.TRAIN_DATA_PATH):
            raise FileNotFoundError(f"训练集文件不存在: {self.config.TRAIN_DATA_PATH}")
        train_df = pd.read_csv(self.config.TRAIN_DATA_PATH)
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

        # 定义标签转换函数
        def convert_label(stars):
            if stars in [1, 2]:
                return 0  # Negative
            elif stars in [4, 5]:
                return 1  # Positive
            else:
                return -1  # 需要丢弃

        print("\n" + "=" * 50)
        print("步骤1.1: 处理训练集")
        print("=" * 50)
        # 处理训练集（筛选英语）
        train_df = train_df[train_df['language'] == 'en'].copy()
        train_df['label'] = train_df['stars'].apply(convert_label)
        train_df = train_df[train_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选英语并转换标签后: {len(train_df)} 条")

        print("\n" + "=" * 50)
        print("步骤1.2: 处理验证集")
        print("=" * 50)
        # 处理验证集（筛选英语）
        valid_df = valid_df[valid_df['language'] == 'en'].copy()
        valid_df['label'] = valid_df['stars'].apply(convert_label)
        valid_df = valid_df[valid_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选英语并转换标签后: {len(valid_df)} 条")

        print("\n" + "=" * 50)
        print("步骤1.3: 处理测试集")
        print("=" * 50)
        # 处理测试集（筛选日语）
        test_df = test_df[test_df['language'] == 'ja'].copy()
        test_df['label'] = test_df['stars'].apply(convert_label)
        test_df = test_df[test_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选日语并转换标签后: {len(test_df)} 条")

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

    def create_tfidf_mask(self, texts):
        """
        使用TF-IDF创建掩码增强样本

        Args:
            texts: 文本列表

        Returns:
            masked_texts: 掩码后的文本列表
        """
        print("\n" + "=" * 50)
        print("步骤2: 计算TF-IDF并创建掩码样本")
        print("=" * 50)

        # 计算TF-IDF
        print("正在计算TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"  ✓ TF-IDF词汇表大小: {len(feature_names)}")

        masked_texts = []

        for idx, text in enumerate(tqdm(texts, desc="创建TF-IDF掩码")):
            # 获取当前文本的TF-IDF向量
            tfidf_scores = tfidf_matrix[idx].toarray()[0]

            # 找出TF-IDF分数最高的词
            word_scores = {
                feature_names[i]: tfidf_scores[i]
                for i in range(len(feature_names))
                if tfidf_scores[i] > 0
            }

            if len(word_scores) == 0:
                masked_texts.append(text)
                continue

            # 排序并获取Top K的词
            sorted_words = sorted(
                word_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_k = max(1, int(len(sorted_words) * self.config.TOP_TFIDF_RATIO))
            top_words = [word for word, score in sorted_words[:top_k]]

            # 替换为<mask>
            masked_text = text
            for word in top_words:
                # 使用单词边界进行替换，避免部分匹配
                masked_text = masked_text.replace(f" {word} ", f" {self.mask_token} ")
                masked_text = masked_text.replace(f" {word},", f" {self.mask_token},")
                masked_text = masked_text.replace(f" {word}.", f" {self.mask_token}.")

            masked_texts.append(masked_text)

        print(f"✓ 成功创建 {len(masked_texts)} 条掩码样本")
        print("=" * 50 + "\n")

        return masked_texts

    def translate_texts(self, texts, dataset_name=""):
        """
        使用Helsinki-NLP模型翻译英文文本到日语

        Args:
            texts: 英文文本列表
            dataset_name: 数据集名称（用于缓存文件命名）

        Returns:
            translated_texts: 日语翻译列表
        """
        print("=" * 50)
        print(f"步骤3: 翻译英文到日语 ({dataset_name})")
        print("=" * 50)

        # 确保输出目录存在
        self.config.create_dirs()

        # 检查是否有缓存
        cache_file = os.path.join(
            self.config.OUTPUT_DIR,
            f'translated_cache_{dataset_name}.csv'
        )

        # 首先检查用户配置的缓存文件
        if self.config.USE_CACHED_TRANSLATION and self.config.CACHED_TRANSLATION_FILE:
            if os.path.exists(self.config.CACHED_TRANSLATION_FILE):
                print(f"加载用户指定的缓存翻译: {self.config.CACHED_TRANSLATION_FILE}")
                try:
                    cached_df = pd.read_csv(self.config.CACHED_TRANSLATION_FILE)
                    if 'review_body_translated' in cached_df.columns:
                        if len(cached_df) >= len(texts):
                            translated_texts = cached_df['review_body_translated'].tolist()[:len(texts)]
                            print(f"✓ 成功加载 {len(translated_texts)} 条翻译")
                            print("=" * 50 + "\n")
                            return translated_texts
                except Exception as e:
                    print(f"⚠ 加载缓存失败: {e}")

        # 检查自动生成的缓存文件
        if os.path.exists(cache_file):
            print(f"加载自动缓存: {cache_file}")
            try:
                cached_df = pd.read_csv(cache_file)
                if 'review_body_translated' in cached_df.columns:
                    if len(cached_df) >= len(texts):
                        translated_texts = cached_df['review_body_translated'].tolist()[:len(texts)]
                        print(f"✓ 成功加载 {len(translated_texts)} 条翻译")
                        print("=" * 50 + "\n")
                        return translated_texts
                    else:
                        print(f"⚠ 缓存数量({len(cached_df)})不足，需要翻译{len(texts)}条")
            except Exception as e:
                print(f"⚠ 加载缓存失败: {e}")

        # 使用Helsinki-NLP模型进行翻译
        print("\n开始使用Helsinki-NLP模型翻译...")
        print(f"翻译模型路径: {self.config.TRANSLATION_MODEL_PATH}")
        print(f"需要翻译 {len(texts)} 条数据\n")

        try:
            from transformers import MarianMTModel, MarianTokenizer
            import torch

            # 加载翻译模型和tokenizer
            print("加载翻译模型...")
            if not os.path.exists(self.config.TRANSLATION_MODEL_PATH):
                raise FileNotFoundError(
                    f"翻译模型不存在: {self.config.TRANSLATION_MODEL_PATH}\n"
                    f"请确认模型路径是否正确"
                )

            tokenizer = MarianTokenizer.from_pretrained(self.config.TRANSLATION_MODEL_PATH)
            model = MarianMTModel.from_pretrained(self.config.TRANSLATION_MODEL_PATH)

            # 将模型移到GPU（如果可用）
            device = self.config.DEVICE
            model = model.to(device)
            model.eval()

            print(f"✓ 翻译模型加载成功 (设备: {device})")
            print(f"✓ 批次大小: {self.config.TRANSLATION_BATCH_SIZE}\n")

            translated_texts = []
            batch_size = self.config.TRANSLATION_BATCH_SIZE

            # 批量翻译
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="翻译进度"):
                    batch_texts = texts[i:i + batch_size]

                    try:
                        # Tokenize
                        inputs = tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config.TRANSLATION_MAX_LENGTH
                        ).to(device)

                        # 生成翻译
                        translated = model.generate(
                            **inputs,
                            max_length=self.config.TRANSLATION_MAX_LENGTH,
                            num_beams=4,
                            early_stopping=True
                        )

                        # 解码
                        batch_translations = tokenizer.batch_decode(
                            translated,
                            skip_special_tokens=True
                        )

                        translated_texts.extend(batch_translations)

                    except Exception as e:
                        # 如果批次翻译失败，使用原文
                        print(f"\n⚠ 批次 {i//batch_size + 1} 翻译失败: {str(e)[:100]}")
                        translated_texts.extend(batch_texts)

            print(f"\n✓ 翻译完成: {len(translated_texts)} 条")

            # 释放GPU内存
            del model
            del tokenizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ 翻译模型加载/运行失败: {e}")
            print("⚠ 使用原文代替翻译\n")
            print("=" * 50 + "\n")
            return texts

        # 保存翻译结果到缓存
        try:
            pd.DataFrame({
                'review_body_translated': translated_texts
            }).to_csv(cache_file, index=False, encoding='utf-8')
            print(f"✓ 翻译结果已保存到: {cache_file}")
            print("  下次运行将自动使用此缓存")
        except Exception as e:
            print(f"⚠ 保存翻译缓存失败: {e}")

        print("=" * 50 + "\n")

        return translated_texts

    def prepare_datasets(self):
        """
        准备训练集、验证集和测试集

        Returns:
            train_data: 训练数据字典
            valid_data: 验证数据字典
            test_data: 测试数据字典
        """
        # 加载和筛选数据
        train_df, valid_df, test_df = self.load_and_filter_data()

        # ===== 处理训练集 =====
        print("\n" + "=" * 50)
        print("处理训练集数据")
        print("=" * 50)

        # 创建TF-IDF掩码（仅在训练集上计算TF-IDF）
        train_masked_texts = self.create_tfidf_mask(train_df['review_body'].tolist())
        train_df['masked_text'] = train_masked_texts

        # 翻译训练集
        train_translated_texts = self.translate_texts(
            train_df['review_body'].tolist(),
            dataset_name="train"
        )
        train_df['translated_text'] = train_translated_texts

        # ===== 处理验证集 =====
        print("\n" + "=" * 50)
        print("处理验证集数据")
        print("=" * 50)

        # 验证集也需要掩码和翻译（使用训练集的TF-IDF）
        if self.tfidf_vectorizer is not None:
            print("使用训练集的TF-IDF对验证集进行掩码...")
            valid_masked_texts = self._apply_tfidf_mask(valid_df['review_body'].tolist())
            print(f"✓ 验证集掩码完成\n")
        else:
            print("⚠ TF-IDF未初始化，验证集使用原文")
            valid_masked_texts = valid_df['review_body'].tolist()

        valid_df['masked_text'] = valid_masked_texts

        valid_translated_texts = self.translate_texts(
            valid_df['review_body'].tolist(),
            dataset_name="valid"
        )
        valid_df['translated_text'] = valid_translated_texts

        # 构建训练数据
        train_data = {
            'original_text': train_df['review_body'].tolist(),
            'masked_text': train_df['masked_text'].tolist(),
            'translated_text': train_df['translated_text'].tolist(),
            'label': train_df['label'].tolist()
        }

        # 构建验证数据
        valid_data = {
            'original_text': valid_df['review_body'].tolist(),
            'masked_text': valid_df['masked_text'].tolist(),
            'translated_text': valid_df['translated_text'].tolist(),
            'label': valid_df['label'].tolist()
        }

        # 构建测试数据（日语数据 - Zero-shot测试）
        test_data = {
            'text': test_df['review_body'].tolist(),
            'label': test_df['label'].tolist()
        }

        print("\n" + "=" * 50)
        print("数据准备完成！")
        print("=" * 50)
        print(f"训练集大小: {len(train_data['label'])} 条")
        print(f"验证集大小: {len(valid_data['label'])} 条")
        print(f"测试集大小: {len(test_data['label'])} 条")
        print("=" * 50 + "\n")

        return train_data, valid_data, test_data

    def _apply_tfidf_mask(self, texts):
        """
        应用已经训练好的TF-IDF来掩码文本（用于验证集）

        Args:
            texts: 文本列表

        Returns:
            masked_texts: 掩码后的文本列表
        """
        if self.tfidf_vectorizer is None:
            return texts

        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        masked_texts = []

        for idx, text in enumerate(tqdm(texts, desc="应用TF-IDF掩码")):
            tfidf_scores = tfidf_matrix[idx].toarray()[0]
            word_scores = {
                feature_names[i]: tfidf_scores[i]
                for i in range(len(feature_names))
                if tfidf_scores[i] > 0
            }

            if len(word_scores) == 0:
                masked_texts.append(text)
                continue

            sorted_words = sorted(
                word_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_k = max(1, int(len(sorted_words) * self.config.TOP_TFIDF_RATIO))
            top_words = [word for word, score in sorted_words[:top_k]]

            masked_text = text
            for word in top_words:
                masked_text = masked_text.replace(f" {word} ", f" {self.mask_token} ")
                masked_text = masked_text.replace(f" {word},", f" {self.mask_token},")
                masked_text = masked_text.replace(f" {word}.", f" {self.mask_token}.")

            masked_texts.append(masked_text)

        return masked_texts