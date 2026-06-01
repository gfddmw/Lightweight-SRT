"""
build_csl_daily_vocab.py
=========================
Task C · 任务点一：构建 Gloss 与 中文词表

从 CSL-Daily 数据集构建：
1. 手语 Gloss 词表（基于 csl-daily.json 中的 gloss 列）
2. 中文词表（使用 jieba 对中文句子进行分词）
3. 统一特殊 Token：<PAD>/<BOS>/<EOS>/<UNK>

输出：
- configs/csl_daily/gloss_vocab.json
- configs/csl_daily/text_vocab.json
- configs/csl_daily/dataset_splits.json

Usage:
python scripts/data_prep/build_csl_daily_vocab.py --csl_root D:/nju/2/SLT/dataset/CSL-Daily/
"""

import json
import os
import argparse
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set
import logging
import jieba
import random

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 特殊 Token 定义
SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "BOS": "<BOS>",
    "EOS": "<EOS>",
    "UNK": "<UNK>"
}

def load_csl_daily_data(csl_daily_dir: Path) -> tuple[List[Dict], Dict]:
    """加载 CSL-Daily 数据，返回每条样本的元数据和分割信息"""
    # 读取主 JSON 文件
    json_path = csl_daily_dir / "csl-daily.json"

    if not json_path.exists():
        raise FileNotFoundError(f"未找到 csl-daily.json: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为列表格式
    samples = []
    for video_id, metadata in data.items():
        # 安全获取字段，提供默认值
        sample = {
            'index': metadata.get('index', ''),
            'name': video_id,  # video_id 作为 name
            'length': metadata.get('length', 0),
            'gloss': metadata.get('gloss', ''),  # 手语 gloss（空格分隔的词序列）
            'chars': metadata.get('char', ''),    # 汉字序列
            'words': metadata.get('word', ''),    # 中文词序列
            'postag': metadata.get('postag', ''),
            'split': metadata.get('split', 'train')  # 分割信息
        }
        # 验证必要字段
        if not sample['index']:
            logger.warning(f"样本 {video_id} 缺少 index，跳过")
            continue
        if not sample['gloss']:
            logger.warning(f"样本 {video_id} 缺少 gloss，跳过")
            continue
        samples.append(sample)

    # 按分割分组
    split_data = {'train': [], 'dev': [], 'test': []}
    for sample in samples:
        split_name = sample.get('split', 'train')
        if split_name in split_data:
            split_data[split_name].append(sample)

    logger.info(f"加载了 {len(samples)} 条 CSL-Daily 数据")
    logger.info(f"训练集: {len(split_data['train'])} 条")
    logger.info(f"验证集: {len(split_data['dev'])} 条")
    logger.info(f"测试集: {len(split_data['test'])} 条")

    return samples, split_data

def build_gloss_vocab(data: List[Dict], min_freq: int = 1) -> Dict:
    """构建手语 Gloss 词表"""
    # 收集所有 gloss 词
    gloss_counter = Counter()
    for i, sample in enumerate(data):
        # 调试信息
        if i == 0:
            logger.debug(f"第一个样本类型: {type(sample)}")
            logger.debug(f"第一个样本内容: {sample}")

        # 处理 gloss 字段
        if isinstance(sample, dict):
            gloss = sample.get('gloss', '')
            if not isinstance(gloss, str):
                gloss = str(gloss) if gloss is not None else ''
                logger.warning(f"样本 {i} 的 gloss 不是字符串类型: {type(gloss)}")
        else:
            logger.error(f"样本 {i} 不是字典类型: {type(sample)}")
            continue

        # 确保是字符串类型
        if gloss is None:
            gloss = ''

        # gloss 是空格分隔的词序列，如 "你 们 好"
        gloss_words = gloss.split()
        for word in gloss_words:
            if word.strip():  # 过滤空字符串
                gloss_counter[word] += 1
            else:
                logger.warning(f"样本 {i} 发现空 gloss 词")

    # 按频率排序
    sorted_glosses = [word for word, count in gloss_counter.most_common()
                     if count >= min_freq]

    # 构建 vocab {word: idx}
    vocab = {}
    # 添加特殊 Token
    for idx, token in enumerate(SPECIAL_TOKENS.values()):
        vocab[token] = idx

    # 添加 gloss 词
    for idx, word in enumerate(sorted_glosses, start=len(SPECIAL_TOKENS)):
        vocab[word] = idx

    # 创建反向映射
    idx2word = {idx: word for word, idx in vocab.items()}

    # 统计信息
    logger.info(f"Gloss 词表统计:")
    logger.info(f"  总词汇量: {len(vocab)}")
    logger.info(f"  特殊 Token: {len(SPECIAL_TOKENS)}")
    logger.info(f"  Gloss 词: {len(sorted_glosses)}")
    logger.info(f"  最频繁的 10 个词: {sorted_glosses[:10]}")

    return {
        'vocab': vocab,
        'idx2word': idx2word,
        'counter': dict(gloss_counter)
    }

def build_text_vocab(data: List[Dict], min_freq: int = 1) -> Dict:
    """构建中文词表（使用 jieba 分词）"""
    # 收集所有中文句子
    word_counter = Counter()
    char_counter = Counter()
    sentence_list = []

    for sample in data:
        # 处理 words 字段
        words_text = sample.get('words', '')
        if not isinstance(words_text, str):
            words_text = str(words_text) if words_text is not None else ''

        # 获取中文句子（去除标点）
        chinese_text = re.sub(r'[^一-龥]', ' ', words_text)
        chinese_text = ' '.join(chinese_text.split())  # 标准化空格

        if chinese_text:
            sentence_list.append(chinese_text)

            # 使用 jieba 分词
            words = jieba.lcut(chinese_text, cut_all=False)  # 精确模式
            for word in words:
                if len(word.strip()) > 0:  # 过滤空词
                    word_counter[word] += 1

            # 统计汉字
            chars = list(chinese_text)
            for char in chars:
                if char.strip():  # 过滤空格
                    char_counter[char] += 1

    # 按频率排序
    sorted_words = [word for word, count in word_counter.most_common()
                   if count >= min_freq]
    sorted_chars = [char for char, count in char_counter.most_common()
                    if count >= min_freq]

    # 构建 word vocab
    word_vocab = {}
    for idx, token in enumerate(SPECIAL_TOKENS.values()):
        word_vocab[token] = idx

    for idx, word in enumerate(sorted_words, start=len(SPECIAL_TOKENS)):
        word_vocab[word] = idx

    # 构建 char vocab
    char_vocab = {}
    for idx, token in enumerate(SPECIAL_TOKENS.values()):
        char_vocab[token] = idx

    for idx, char in enumerate(sorted_chars, start=len(SPECIAL_TOKENS)):
        char_vocab[char] = idx

    # 创建反向映射
    word_idx2word = {idx: word for word, idx in word_vocab.items()}
    char_idx2char = {idx: char for char, idx in char_vocab.items()}

    # 统计信息
    logger.info(f"中文词表统计:")
    logger.info(f"  句子数: {len(sentence_list)}")
    logger.info(f"  词词汇量: {len(word_vocab)}")
    logger.info(f"  字词汇量: {len(char_vocab)}")
    logger.info(f"  最频繁的 10 个词: {sorted_words[:10]}")
    logger.info(f"  最频繁的 10 个字: {sorted_chars[:10]}")

    return {
        'word_vocab': word_vocab,
        'word_idx2word': word_idx2word,
        'char_vocab': char_vocab,
        'char_idx2char': char_idx2char,
        'word_counter': dict(word_counter),
        'char_counter': dict(char_counter),
        'sentences': sentence_list[:100]  # 保存前100个句子作为示例
    }

def build_dataset_splits(split_data: Dict) -> Dict:
    """构建数据集分割索引"""
    splits = {}

    for split_name, samples in split_data.items():
        splits[split_name] = []
        for sample in samples:
            splits[split_name].append({
                'index': sample['index'],
                'name': sample['name'],
                'length': sample['length'],
                'gloss': sample['gloss'],
                'chars': sample['chars'],
                'words': sample['words']
            })

    # 保存统计信息
    stats = {
        'train_size': len(splits['train']),
        'dev_size': len(splits['dev']),
        'test_size': len(splits['test']),
        'total_size': len(splits['train']) + len(splits['dev']) + len(splits['test'])
    }

    logger.info(f"数据集分割统计:")
    logger.info(f"  训练集: {stats['train_size']}")
    logger.info(f"  验证集: {stats['dev_size']}")
    logger.info(f"  测试集: {stats['test_size']}")
    logger.info(f"  总计: {stats['total_size']}")

    return {
        'splits': splits,
        'stats': stats
    }

def save_vocab(vocab: Dict, output_path: Path):
    """保存词表到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"词表已保存到: {output_path}")

def create_vocabularies_dir():
    """创建词汇表目录"""
    vocab_dir = Path("data/csl_daily/vocabularies")
    splits_dir = Path("data/csl_daily/splits")

    vocab_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    return vocab_dir, splits_dir

def generate_sample_mappings(data: List[Dict], output_dir: Path):
    """生成样本映射示例（供后续使用）"""
    # 创建一个映射字典的模板
    mapping_template = {}

    # 随机选择一些样本作为示例
    sample_indices = random.sample(range(len(data)), min(10, len(data)))

    for idx in sample_indices:
        sample = data[idx]
        mapping_template[sample['name']] = {
            "skeleton": f"processed/csl_daily/skeletons/{sample['name']}.npy",
            "teacher_feat": f"processed/csl_daily/teacher_features/{sample['name']}.npy",
            "teacher_logits": f"processed/csl_daily/teacher_logits/{sample['name']}.npy",
            "gloss": sample['gloss'],
            "sentence": sample['words'],
            "subset": "train"  # 这里需要根据实际分割确定
        }

    # 保存映射模板
    template_path = output_dir / "sample_mapping.json"
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_template, f, ensure_ascii=False, indent=2)

    logger.info(f"样本映射模板已保存到: {template_path}")

def main():
    parser = argparse.ArgumentParser(description="构建 CSL-Daily 词表")
    parser.add_argument("--csl_root", type=str,
                        default="D:/nju/2/SLT/dataset/CSL-Daily/",
                        help="CSL-Daily 数据根目录")
    parser.add_argument("--gloss_output", type=str,
                        default="data/csl_daily/vocabularies/gloss_vocab.json",
                        help="Gloss 词表输出路径")
    parser.add_argument("--text_output", type=str,
                        default="data/csl_daily/vocabularies/text_vocab.json",
                        help="中文词表输出路径")
    parser.add_argument("--splits_output", type=str,
                        default="data/csl_daily/splits/dataset_splits.json",
                        help="数据集分割输出路径")
    parser.add_argument("--min_freq", type=int, default=1,
                        help="最小词频")
    parser.add_argument("--jieba_dict", type=str, default=None,
                        help="自定义 jieba 词典路径")
    args = parser.parse_args()

    # 创建目录结构
    create_vocabularies_dir()

    # 初始化 jieba
    if args.jieba_dict:
        jieba.load_userdict(args.jieba_dict)
        logger.info(f"加载了自定义 jieba 词典: {args.jieba_dict}")

    # 确保分词词典已加载
    jieba.initialize()

    # 转换为 Path 对象
    csl_root = Path(args.csl_root)
    gloss_output = Path(args.gloss_output)
    text_output = Path(args.text_output)
    splits_output = Path(args.splits_output)

    # 加载数据
    data, split_data = load_csl_daily_data(csl_root)

    # 构建词表
    gloss_vocab = build_gloss_vocab(data, args.min_freq)
    text_vocab = build_text_vocab(data, args.min_freq)
    dataset_splits = build_dataset_splits(split_data)

    # 保存词表
    save_vocab(gloss_vocab, gloss_output)
    save_vocab(text_vocab, text_output)
    save_vocab(dataset_splits, splits_output)

    # 生成样本映射示例
    vocab_dir = Path("data/csl_daily/vocabularies")
    generate_sample_mappings(data, vocab_dir)

    # 生成统计报告
    stats = {
        'total_samples': len(data),
        'gloss_vocab_size': len(gloss_vocab['vocab']),
        'text_word_vocab_size': len(text_vocab['word_vocab']),
        'text_char_vocab_size': len(text_vocab['char_vocab']),
        'train_size': dataset_splits['stats']['train_size'],
        'dev_size': dataset_splits['stats']['dev_size'],
        'test_size': dataset_splits['stats']['test_size']
    }

    stats_output = vocab_dir / "vocab_stats.json"
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("词表构建完成！")
    logger.info(f"统计信息: {stats}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()