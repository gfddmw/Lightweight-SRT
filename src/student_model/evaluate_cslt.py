#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 路径修复
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.datasets.cslt_dataset import CSLTDataset, cslt_collate_fn
from src.student_model.architecture.multi_stream_stgcn import MultiStreamSTGCN
from src.student_model.architecture.cslt_model import CSLTModel
from src.student_model.train_cslt import prepare_input


def calculate_edit_distance(reference, hypothesis):
    """
    计算两个序列之间的 Levenshtein 距离（编辑距离）。
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j
        
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                dp[i][j] = min(substitution, deletion, insertion)
                
    return dp[ref_len][hyp_len]


def ctc_greedy_decode(ctc_logits, input_lengths, blank_id=2004):
    """
    CTC 贪婪解码算法。
    ctc_logits: [B, T_out, num_classes] 的 Tensor。
    input_lengths: [B] 的 Tensor，下采样之前的时序长度。
    """
    # 按照 CTC Argmax 原理，直接取每个时间步概率最大的类别
    predictions = torch.argmax(ctc_logits, dim=-1)  # [B, T_out]
    
    decoded_sentences = []
    
    for i in range(predictions.size(0)):
        seq_len = (input_lengths[i].item() - 1) // 4 + 1
        pred_seq = predictions[i, :seq_len].tolist()
        
        # 去除连续重复的元素并去除 blank
        decoded = []
        prev = None
        for val in pred_seq:
            if val != prev:
                if val != blank_id:
                    decoded.append(val)
                prev = val
        decoded_sentences.append(decoded)
        
    return decoded_sentences


def main():
    parser = argparse.ArgumentParser(description="CSLT Model Evaluation Script (WER)")
    
    # 路径与控制参数
    parser.add_argument('--global_index_path', type=str, default='data/csl_daily/vocabularies/global_index.json',
                        help='Data global index path')
    parser.add_argument('--work_dir', type=str, default='work_dir/CSL',
                        help='Directory where the best model and logs are saved')
    parser.add_argument('--subset', type=str, default='dev', choices=['train', 'dev', 'test'],
                        help='Dataset subset to evaluate on (default: dev)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable mixed precision inference (AMP)')
    parser.add_argument('--random_init', action='store_true', default=False,
                        help='Evaluate on randomly initialized model weights (no pretrained or trained weights)')
    
    args = parser.parse_args()
    
    # 1. 实例化评估集
    dataset = CSLTDataset(
        global_index_path=args.global_index_path,
        subset=args.subset,
        in_channels=3
    )
    print(f"成功加载评估集 '{args.subset}'，共有 {len(dataset)} 个样本")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cslt_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 2. 构建模型
    print("正在构建 CSLTModel 与 MultiStreamSTGCN encoder...")
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    encoder = MultiStreamSTGCN(
        num_class=10, 
        in_channels=3, 
        graph_args=graph_args,
        edge_importance_weighting=True
    )
    
    model = CSLTModel(
        encoder=encoder,
        in_channels=768,
        out_channels=1024,
        num_classes=2005
    )
    
    # 加载权重
    if args.random_init:
        print("警告：已开启 --random_init 模式！模型将使用随机初始化的权重进行评估，不加载任何训练过的参数。")
    else:
        best_path = Path(args.work_dir) / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"未找到最优权重文件: {best_path}。若想评估随机初始化模型，请传入 --random_init")
            
        state_dict = torch.load(best_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"成功加载最优权重: {best_path}")
    
    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    model.eval()
    
    total_distance = 0
    total_words = 0
    
    # 4. 开始评估循环
    print(f"开始遍历 '{args.subset}' 进行测试计算...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 数据送入设备并转换形状
            joints, bones, motion, input_lengths, gloss_ids, gloss_lengths, teacher_logits, teacher_lengths = prepare_input(batch, device)
            
            # 前向计算
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(joints=joints, bones=bones, motion=motion)
                ctc_logits = outputs["ctc_logits"]
            
            # 解码
            # blank_id 为词表大小 2004
            decoded_preds = ctc_greedy_decode(ctc_logits, input_lengths, blank_id=2004)
            
            # 对比并累加编辑距离
            for idx in range(len(decoded_preds)):
                pred_seq = decoded_preds[idx]
                
                # 获取 True Sequence，并去除 padding
                true_len = gloss_lengths[idx].item()
                true_seq = gloss_ids[idx, :true_len].tolist()
                
                # 计算编辑距离
                distance = calculate_edit_distance(true_seq, pred_seq)
                
                total_distance += distance
                total_words += true_len
                
    # 5. 计算并打印总错误率
    if total_words == 0:
        print("错误：评估集总词数为 0！")
        return
        
    wer = (total_distance / total_words) * 100
    
    print("=" * 60)
    print(f"评估完成！评估子集: {args.subset}")
    print(f"总编辑距离 (S + I + D): {total_distance}")
    print(f"总参考词数 (N): {total_words}")
    print(f"模型错词率 (WER): {wer:.4f} %")
    print("=" * 60)
    
    # 将测试结果追加写入工作目录下的 train.log
    log_path = Path(args.work_dir) / "train.log"
    if log_path.parent.exists():
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"模型评估报告 (自动测试)\n")
            f.write(f"评估子集: {args.subset}\n")
            if args.random_init:
                f.write(f"权重模式: 随机初始化 (无训练)\n")
            else:
                f.write(f"最优权重路径: {best_path}\n")
            f.write(f"总编辑距离 (S + I + D): {total_distance}\n")
            f.write(f"总参考词数 (N): {total_words}\n")
            f.write(f"错词率 (WER): {wer:.4f} %\n")
            f.write("=" * 50 + "\n")
        print(f"已将评估报告追加写入至: {log_path}")


if __name__ == '__main__':
    main()
