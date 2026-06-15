#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 路径修复：确保项目根目录在 sys.path 中，以便直接运行和引用
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows 编码修复，防止在 GBK 控制台打印 UTF-8 时报错，并开启行缓冲确保实时输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.common.datasets.cslt_dataset import CSLTDataset, cslt_collate_fn
from src.student_model.architecture.multi_stream_stgcn import MultiStreamSTGCN
from src.student_model.architecture.cslt_model import CSLTModel
from src.student_model.feature_loss import masked_feature_mse_loss


class DualLogger(object):
    """
    双向日志记录器，同时将输出流输出到控制台和指定日志文件，并实时刷新。
    """
    def __init__(self, terminal, file_path):
        self.terminal = terminal
        self.log_file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()



def load_pretrained_weights(model, weights_path):
    """
    加载 WLASL2000 孤立词 SOTA 预训练权重。
    实现权重名称映射，加上 "encoder." 前缀，并丢弃 fcn 分类头。
    """
    print(f"==================================================")
    print(f"开始加载预训练权重: {weights_path}")
    print(f"==================================================")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到预训练权重文件: {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location='cpu')
    model_state = model.state_dict()
    
    new_state_dict = {}
    ignored_keys = []
    mapped_count = 0
    
    for k, v in checkpoint.items():
        # 丢弃 classification fcn 分类头
        if "fcn" in k:
            ignored_keys.append(k)
            continue
            
        # 映射名称：添加 "encoder." 前缀
        new_k = "encoder." + k
        
        if new_k in model_state:
            # 校验 shape 是否匹配
            if model_state[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                mapped_count += 1
            else:
                print(f"维度不匹配: {new_k} | 目标模型形状 {model_state[new_k].shape} | 权重文件形状 {v.shape}")
        else:
            print(f"键名在目标模型中不存在: {new_k}")
            
    print(f"成功映射权重键值对数量: {mapped_count} 个")
    print(f"已过滤 (包含 'fcn' 的分类头) 的键值对数量: {len(ignored_keys)} 个")
    
    # 允许不完全加载，因为 TemporalAdapter 和 CTCHead 不在 WLASL2000 backbone 中
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"非严格加载报告:")
    print(f"  - 缺失参数 (未加载) 数量: {len(missing_keys)} (大部分属于 temporal_adapter 和 ctc_head，此为预期表现)")
    if unexpected_keys:
        print(f"  - 冗余参数 数量: {len(unexpected_keys)}")
        
    print(f"==================================================\n")
    return model


def prepare_input(batch, device):
    """
    处理输入数据，将 42 个关节点 (包含左右手各 21 个点)
    拆分为左右手两个 instance (M=2)，拼接成形状为 [B, C, T, 21, 2] 的张量。
    """
    joints = batch["joints"]  # [B, C, T, 42, 1]
    bones = batch["bones"]    # [B, C, T, 42, 1]
    motion = batch["motion"]  # [B, C, T, 42, 1]
    
    # 左右手各 21 个点，并在最后一个维度 M 上进行 concat，从而得到 M=2 维
    joints_split = torch.cat([joints[..., :21, :], joints[..., 21:, :]], dim=-1).to(device)
    bones_split = torch.cat([bones[..., :21, :], bones[..., 21:, :]], dim=-1).to(device)
    motion_split = torch.cat([motion[..., :21, :], motion[..., 21:, :]], dim=-1).to(device)
    
    input_lengths = batch["input_lengths"].to(device)
    gloss_ids = batch["gloss_ids"].to(device)
    gloss_lengths = batch["gloss_lengths"].to(device)
    
    return joints_split, bones_split, motion_split, input_lengths, gloss_ids, gloss_lengths


def scale_lengths_to_time(lengths, source_time, target_time):
    """
    将输入序列长度按实际模型输出时间维缩放，用于时序特征 mask。
    """
    if source_time <= 0:
        return torch.zeros_like(lengths)
    scaled = torch.ceil(lengths.float() * float(target_time) / float(source_time))
    return scaled.long().clamp(min=0, max=target_time)


def main():
    parser = argparse.ArgumentParser(description="CSLT Temporal fine-tuning script with CTC Loss")
    
    # 数据相关参数
    parser.add_argument('--global_index_path', type=str, default='data/csl_daily/vocabularies/global_index.json',
                        help='Data global index path')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    
    # 优化与训练控制
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--debug_overfit', action='store_true', default=False,
                        help='Debug overfitting mode: use 20 samples to train on a single batch for 50 epochs')
    parser.add_argument('--work_dir', type=str, default='work_dir/CSL',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading (default: 2)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable mixed precision training (AMP) to speed up')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='Path to pretrained weights to load before training (default: None, trains from scratch)')
    parser.add_argument('--feature_loss_weight', type=float, default=1.0,
                        help='Weight for masked temporal feature MSE loss. Set to 0 to train with CTC only.')
    
    args = parser.parse_args()
    
    # 建立模型参数和日志保存路径
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    # 启动双向日志记录（控制台 + 文件）
    log_file = work_dir / "train.log"
    sys.stdout = DualLogger(sys.stdout, log_file)
    sys.stderr = DualLogger(sys.stderr, log_file)
    
    # 1. 实例化 Dataset
    dataset = CSLTDataset(
        global_index_path=args.global_index_path,
        subset="train",
        in_channels=3
    )
    print(f"成功加载训练集，共有 {len(dataset)} 个样本")
    
    # Debug 过拟合配置
    if args.debug_overfit:
        print("开启 debug_overfit 模式！")
        dataset.sample_ids = dataset.sample_ids[:20]
        print(f"已将训练集样本限制为前 {len(dataset.sample_ids)} 条。")
        args.epochs = 50
        batch_size = args.batch_size  # 使用较小 batch_size 防止 OOM，之后仅对第一个 batch 进行过拟合
    else:
        batch_size = args.batch_size
        
    # 2. 配置 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not args.debug_overfit),  # overfit 时固定顺序
        collate_fn=cslt_collate_fn,
        num_workers=0 if args.debug_overfit else args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0 and not args.debug_overfit)
    )
    
    # 3. 初始化模型
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
        num_classes=7388
    )
    
    # 4. 加载预训练权重
    if args.load_weights:
        model = load_pretrained_weights(model, args.load_weights)
    else:
        print("提示：默认不加载权重，模型将从头（随机初始化）开始训练。")
    
    # 5. 配置硬件设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的训练设备: {device}")
    model.to(device)
    
    # 6. 损失函数与优化器
    # 词表大小 7387，分类层 num_classes=7388，blank=7387
    ctc_loss = nn.CTCLoss(blank=7387, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 混合精度 GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    model.train()
    
    # 7. 开始训练循环
    if args.debug_overfit:
        # 在 debug 过拟合模式下，我们仅对固定的单个 batch 进行 50 次循环迭代
        print("开始单批次过拟合训练循环 (50 Epochs)...")
        debug_batch = next(iter(dataloader))
        
        # 将数据送入 GPU/CPU 并按模型通道需求处理
        joints, bones, motion, input_lengths, gloss_ids, gloss_lengths = prepare_input(debug_batch, device)
        teacher_feats = debug_batch["teacher_feats"].to(device) if args.feature_loss_weight > 0 else None
        teacher_lengths = debug_batch["teacher_lengths"].to(device) if args.feature_loss_weight > 0 else None
        
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            # 开启混合精度前向计算
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(joints=joints, bones=bones, motion=motion)
                ctc_logits = outputs["ctc_logits"]
                adapted_feat = outputs["adapted_feat"]
            
            # 转换为 FP32 进行 softmax/CTC 计算以防溢出
            ctc_logits_fp32 = ctc_logits.float()
            log_probs = F.log_softmax(ctc_logits_fp32, dim=-1).transpose(0, 1)
            
            # 计算 targets_lengths (下采样后的序列长度)
            targets_lengths = (input_lengths - 1) // 4 + 1
            
            # 禁用混合精度计算 CTC Loss 以保证数值稳定
            with torch.cuda.amp.autocast(enabled=False):
                ctc_loss_value = ctc_loss(log_probs, gloss_ids, targets_lengths, gloss_lengths)
                if args.feature_loss_weight > 0:
                    student_feature_lengths = scale_lengths_to_time(
                        input_lengths,
                        source_time=joints.size(2),
                        target_time=adapted_feat.size(1),
                    )
                    feature_loss_value = masked_feature_mse_loss(
                        adapted_feat.float(),
                        teacher_feats.float(),
                        student_lengths=student_feature_lengths,
                        teacher_lengths=teacher_lengths,
                    )
                else:
                    feature_loss_value = ctc_loss_value.new_zeros(())
                loss = ctc_loss_value + args.feature_loss_weight * feature_loss_value
            
            # 反向传播与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(
                f"Epoch [{epoch+1:02d}/{args.epochs:02d}] - "
                f"Total Loss: {loss.item():.6f} | "
                f"CTC Loss: {ctc_loss_value.item():.6f} | "
                f"Feature Loss: {feature_loss_value.item():.6f}"
            )
            
        print("Debug 过拟合训练完成！")
        overfit_path = work_dir / "overfit_model.pt"
        torch.save(model.state_dict(), overfit_path)
        print(f"已保存过拟合权重至: {overfit_path}")
    else:
        print(f"开始正式训练循环 ({args.epochs} Epochs)...")
        for epoch in range(args.epochs):
            epoch_losses = []
            epoch_ctc_losses = []
            epoch_feature_losses = []
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                # 数据送入设备并处理
                joints, bones, motion, input_lengths, gloss_ids, gloss_lengths = prepare_input(batch, device)
                
                # 开启混合精度前向计算
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    outputs = model(joints=joints, bones=bones, motion=motion)
                    ctc_logits = outputs["ctc_logits"]
                    adapted_feat = outputs["adapted_feat"]
                
                # 转换为 FP32 进行 softmax/CTC 计算以防溢出
                ctc_logits_fp32 = ctc_logits.float()
                log_probs = F.log_softmax(ctc_logits_fp32, dim=-1).transpose(0, 1)
                targets_lengths = (input_lengths - 1) // 4 + 1
                
                # 禁用混合精度计算 CTC Loss
                with torch.cuda.amp.autocast(enabled=False):
                    ctc_loss_value = ctc_loss(log_probs, gloss_ids, targets_lengths, gloss_lengths)
                    if args.feature_loss_weight > 0:
                        teacher_feats = batch["teacher_feats"].to(device)
                        teacher_lengths = batch["teacher_lengths"].to(device)
                        student_feature_lengths = scale_lengths_to_time(
                            input_lengths,
                            source_time=joints.size(2),
                            target_time=adapted_feat.size(1),
                        )
                        feature_loss_value = masked_feature_mse_loss(
                            adapted_feat.float(),
                            teacher_feats.float(),
                            student_lengths=student_feature_lengths,
                            teacher_lengths=teacher_lengths,
                        )
                    else:
                        feature_loss_value = ctc_loss_value.new_zeros(())
                    loss = ctc_loss_value + args.feature_loss_weight * feature_loss_value
                
                # 反向传播与优化
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(loss.item())
                epoch_ctc_losses.append(ctc_loss_value.item())
                epoch_feature_losses.append(feature_loss_value.item())
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(
                        f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
                        f"Batch [{batch_idx+1:03d}/{len(dataloader):03d}] - "
                        f"Total Loss: {loss.item():.6f} | "
                        f"CTC Loss: {ctc_loss_value.item():.6f} | "
                        f"Feature Loss: {feature_loss_value.item():.6f}"
                    )
            
            mean_loss = np.mean(epoch_losses)
            mean_ctc_loss = np.mean(epoch_ctc_losses)
            mean_feature_loss = np.mean(epoch_feature_losses)
            print(
                f"Epoch [{epoch+1:02d}/{args.epochs:02d}] Finished | "
                f"Average Total Loss: {mean_loss:.6f} | "
                f"Average CTC Loss: {mean_ctc_loss:.6f} | "
                f"Average Feature Loss: {mean_feature_loss:.6f}"
            )
            
            # 保存每 Epoch 结束的 checkpoint
            checkpoint_path = work_dir / f"epoch_{epoch+1}_loss_{mean_loss:.4f}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
            }, checkpoint_path)
            print(f"已保存 Checkpoint 至: {checkpoint_path}")
            
            # 保存 Best 模型
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_path = work_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                print(f"更新最佳模型 (Best Loss: {best_loss:.6f}) 并保存至: {best_path}")


if __name__ == '__main__':
    main()
