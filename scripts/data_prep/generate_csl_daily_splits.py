"""
generate_csl_daily_splits.py
============================
Task C · 任务点二：建立全局数据索引

生成一个包含所有模态路径的 JSON 文件，作为后续数据集的全局索引。

输出格式：
{
  "vid123": {
    "skeleton": "processed/csl_daily/skeletons/vid123.npy",
    "teacher_feat": "processed/csl_daily/teacher_features/vid123.npy",
    "teacher_logits": "processed/csl_daily/teacher_logits/vid123.npy",
    "gloss": "手语 谢谢 你",
    "sentence": "谢谢你",
    "subset": "train"
  }
}

Usage:
python scripts/data_prep/generate_csl_daily_splits.py --csl_root D:/nju/2/SLT/dataset/CSL-Daily/ --vocab_root configs/csl_daily/
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_splits(splits_file: Path) -> Dict:
    """加载数据集分割信息"""
    if not splits_file.exists():
        raise FileNotFoundError(f"未找到数据集分割文件: {splits_file}")

    with open(splits_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_vocabs(vocab_dir: Path) -> Dict:
    """加载词表"""
    vocabs = {}

    # 加载 gloss 词表
    gloss_vocab_file = vocab_dir / "gloss_vocab.json"
    if gloss_vocab_file.exists():
        with open(gloss_vocab_file, 'r', encoding='utf-8') as f:
            gloss_data = json.load(f)
            vocabs['gloss_vocab'] = gloss_data.get('vocab', {})

    # 加载文本词表
    text_vocab_file = vocab_dir / "text_vocab.json"
    if text_vocab_file.exists():
        with open(text_vocab_file, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            vocabs['text_vocab'] = {
                'word_vocab': text_data.get('word_vocab', {}),
                'char_vocab': text_data.get('char_vocab', {})
            }

    return vocabs

def text_to_ids(text: str, vocab: Dict, mode: str = 'word') -> List[int]:
    """将文本转换为 ID 序列"""
    if mode == 'word':
        # 使用空格分词
        words = text.split()
        return [vocab.get(word, vocab.get('<UNK>')) for word in words]
    elif mode == 'char':
        # 按字符分割
        chars = list(text)
        return [vocab.get(char, vocab.get('<UNK>')) for char in chars]
    else:
        raise ValueError(f"不支持的模式: {mode}")

def gloss_to_ids(gloss: str, vocab: Dict) -> List[int]:
    """将 gloss 序列转换为 ID 序列"""
    words = gloss.split()
    return [vocab.get(word, vocab.get('<UNK>')) for word in words]

def build_global_index(splits_data: Dict, vocab_dir: Path, output_path: Path,
                       skeleton_dir: str = "processed/csl_daily/skeletons/",
                       teacher_feat_dir: str = "processed/csl_daily/teacher_features/",
                       teacher_logits_dir: str = "processed/csl_daily/teacher_logits/") -> Dict:
    """构建全局数据索引"""

    # 加载词表
    vocabs = load_vocabs(vocab_dir)
    gloss_vocab = vocabs.get('gloss_vocab', {})
    text_word_vocab = vocabs.get('text_vocab', {}).get('word_vocab', {})
    text_char_vocab = vocabs.get('text_vocab', {}).get('char_vocab', {})

    # 构建全局索引
    global_index = {}
    stats = {
        'total_samples': 0,
        'missing_skeleton': 0,
        'missing_teacher_feat': 0,
        'missing_teacher_logits': 0
    }

    # 遍历所有分割
    for split_name, samples in splits_data['splits'].items():
        for sample in samples:
            vid = sample['name']

            # 检查文件是否存在（仅记录，不作为过滤条件）
            skeleton_path = f"{skeleton_dir}{vid}.npy"
            teacher_feat_path = f"{teacher_feat_dir}{vid}.npy"
            teacher_logits_path = f"{teacher_logits_dir}{vid}.npy"

            # 转换文本为 IDs
            sentence_ids_word = text_to_ids(sample['words'], text_word_vocab, 'word')
            sentence_ids_char = text_to_ids(sample['words'], text_char_vocab, 'char')

            # 转换 gloss 为 IDs
            gloss_ids = gloss_to_ids(sample['gloss'], gloss_vocab)

            # 构建条目
            global_index[vid] = {
                "skeleton": skeleton_path,
                "teacher_feat": teacher_feat_path,
                "teacher_logits": teacher_logits_path,
                "gloss_text": sample['gloss'],
                "sentence_text": sample['words'],
                "gloss_ids": gloss_ids,
                "sentence_ids_word": sentence_ids_word,
                "sentence_ids_char": sentence_ids_char,
                "length": sample['length'],
                "subset": split_name,
                "has_skeleton": Path(skeleton_path).exists(),
                "has_teacher_feat": Path(teacher_feat_path).exists(),
                "has_teacher_logits": Path(teacher_logits_path).exists()
            }

            stats['total_samples'] += 1
            if not Path(skeleton_path).exists():
                stats['missing_skeleton'] += 1
            if not Path(teacher_feat_path).exists():
                stats['missing_teacher_feat'] += 1
            if not Path(teacher_logits_path).exists():
                stats['missing_teacher_logits'] += 1

    # 打印统计信息
    logger.info(f"全局索引构建完成:")
    logger.info(f"  总样本数: {stats['total_samples']}")
    logger.info(f"  缺少骨架文件: {stats['missing_skeleton']} ({stats['missing_skeleton']/stats['total_samples']*100:.1f}%)")
    logger.info(f"  缺少教师特征: {stats['missing_teacher_feat']} ({stats['missing_teacher_feat']/stats['total_samples']*100:.1f}%)")
    logger.info(f"  缺少教师logits: {stats['missing_teacher_logits']} ({stats['missing_teacher_logits']/stats['total_samples']*100:.1f}%)")

    return global_index

def verify_index_alignment(global_index: Dict) -> Dict:
    """验证索引对齐情况"""
    verification = {
        'aligned_samples': 0,
        'misaligned_samples': 0,
        'missing_info': 0
    }

    for vid, data in global_index.items():
        # 检查是否有缺失的信息
        missing_files = []
        if not data['has_skeleton']:
            missing_files.append('skeleton')
        if not data['has_teacher_feat']:
            missing_files.append('teacher_feat')
        if not data['has_teacher_logits']:
            missing_files.append('teacher_logits')

        if missing_files:
            verification['missing_info'] += 1
            logger.debug(f"样本 {vid} 缺少文件: {missing_files}")
        else:
            verification['aligned_samples'] += 1

    verification['misaligned_samples'] = verification['missing_info']

    logger.info(f"索引对齐验证:")
    logger.info(f"  完全对齐: {verification['aligned_samples']}")
    logger.info(f"  缺少文件: {verification['misaligned_samples']}")

    return verification

def save_global_index(global_index: Dict, output_path: Path):
    """保存全局索引"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(global_index, f, ensure_ascii=False, indent=2)

    logger.info(f"全局索引已保存到: {output_path}")

def generate_summary_report(global_index: Dict, output_path: Path):
    """生成数据集摘要报告"""
    report = {
        'dataset_info': {
            'total_videos': len(global_index),
            'train_videos': sum(1 for d in global_index.values() if d['subset'] == 'train'),
            'dev_videos': sum(1 for d in global_index.values() if d['subset'] == 'dev'),
            'test_videos': sum(1 for d in global_index.values() if d['subset'] == 'test'),
        },
        'gloss_stats': {
            'total_glosses': len(set(d['gloss_text'] for d in global_index.values())),
            'avg_gloss_length': sum(len(d['gloss_text'].split()) for d in global_index.values()) / len(global_index)
        },
        'sentence_stats': {
            'total_sentences': len(set(d['sentence_text'] for d in global_index.values())),
            'avg_sentence_length': sum(len(d['sentence_text'].split()) for d in global_index.values()) / len(global_index)
        },
        'file_availability': {
            'skeleton_available': sum(1 for d in global_index.values() if d['has_skeleton']),
            'teacher_feat_available': sum(1 for d in global_index.values() if d['has_teacher_feat']),
            'teacher_logits_available': sum(1 for d in global_index.values() if d['has_teacher_logits'])
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"数据集摘要报告已保存到: {output_path}")

    # 打印关键统计
    logger.info("=" * 60)
    logger.info("数据集摘要:")
    logger.info(f"  总视频数: {report['dataset_info']['total_videos']}")
    logger.info(f"  训练集: {report['dataset_info']['train_videos']}")
    logger.info(f"  验证集: {report['dataset_info']['dev_videos']}")
    logger.info(f"  测试集: {report['dataset_info']['test_videos']}")
    logger.info(f"  平均 gloss 长度: {report['gloss_stats']['avg_gloss_length']:.1f} 词")
    logger.info(f"  平均句子长度: {report['sentence_stats']['avg_sentence_length']:.1f} 词")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="构建 CSL-Daily 全局数据索引")
    parser.add_argument("--csl_root", type=str,
                        default="D:/nju/2/SLT/dataset/CSL-Daily/",
                        help="CSL-Daily 数据根目录")
    parser.add_argument("--vocab_root", type=str,
                        default="data/csl_daily/vocabularies/",
                        help="词表根目录")
    parser.add_argument("--splits_file", type=str,
                        default="data/csl_daily/splits/dataset_splits.json",
                        help="数据集分割文件路径")
    parser.add_argument("--output", type=str,
                        default="data/csl_daily/vocabularies/global_index.json",
                        help="全局索引输出路径")
    parser.add_argument("--report_output", type=str,
                        default="data/csl_daily/vocabularies/dataset_summary.json",
                        help="数据集摘要报告输出路径")
    parser.add_argument("--skeleton_dir", type=str,
                        default="processed/csl_daily/skeletons/",
                        help="骨架文件目录")
    parser.add_argument("--teacher_feat_dir", type=str,
                        default="processed/csl_daily/teacher_features/",
                        help="教师特征目录")
    parser.add_argument("--teacher_logits_dir", type=str,
                        default="processed/csl_daily/teacher_logits/",
                        help="教师logits目录")
    args = parser.parse_args()

    # 转换为 Path 对象
    csl_root = Path(args.csl_root)
    vocab_root = Path(args.vocab_root)
    splits_file = Path(args.splits_file)
    output_path = Path(args.output)
    report_output = Path(args.report_output)

    # 加载数据集分割
    splits_data = load_splits(splits_file)

    # 构建全局索引
    global_index = build_global_index(
        splits_data,
        vocab_root,
        output_path,
        args.skeleton_dir,
        args.teacher_feat_dir,
        args.teacher_logits_dir
    )

    # 验证对齐情况
    verify_index_alignment(global_index)

    # 保存文件
    save_global_index(global_index, output_path)
    generate_summary_report(global_index, report_output)

if __name__ == "__main__":
    main()