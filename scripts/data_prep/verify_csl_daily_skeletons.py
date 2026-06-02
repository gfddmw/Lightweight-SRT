import os
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def verify_skeletons(mapping_path, skeleton_dir):
    mapping_path = Path(mapping_path)
    skeleton_dir = Path(skeleton_dir)
    
    if not mapping_path.exists():
        print(f"❌ 找不到映射文件: {mapping_path}")
        return
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    print(f"📋 映射表中共有 {len(mapping)} 个视频")
    
    missing = []
    corrupted = []
    empty_hands = []
    valid = 0
    
    all_ratios = []
    
    for vid, meta in tqdm(mapping.items(), desc="Checking"):
        npy_path = skeleton_dir / f"{vid}.npy"
        if not npy_path.exists():
            missing.append(vid)
            continue
            
        try:
            data = np.load(npy_path)
            # 校验 shape [T, 42, 9]
            if data.ndim != 3 or data.shape[1] != 42 or data.shape[2] != 9:
                corrupted.append((vid, f"错误维度: {data.shape}，期望 (T, 42, 9)"))
                continue
                
            T = data.shape[0]
            if T == 0:
                corrupted.append((vid, "帧数为 0"))
                continue
                
            # 检查是否有手部关节点数据
            joints = data[:, :, :3]
            # 计算至少有一只手被检测到的帧的比例
            any_hand_frame = np.any(joints != 0, axis=(1, 2))
            valid_ratio = float(any_hand_frame.mean())
            all_ratios.append(valid_ratio)
            
            if valid_ratio == 0:
                empty_hands.append(vid)
            else:
                valid += 1
                
        except Exception as e:
            corrupted.append((vid, f"加载失败: {str(e)}"))
            
    print("=" * 60)
    print("📊 骨骼点数据集完整性自检报告：")
    print("-" * 60)
    print(f"✅ 完美可用 (有效手部数据): {valid} / {len(mapping)} ({valid/len(mapping):.1%})")
    print(f"❓ 虽有文件但无手部坐标 (空帧): {len(empty_hands)} / {len(mapping)}")
    print(f"❌ 损坏或维度错误: {len(corrupted)} / {len(mapping)}")
    print(f"🔍 缺失（未提取）: {len(missing)} / {len(mapping)}")
    
    if all_ratios:
        print(f"📈 平均有效检测帧率 (视频中至少一只手可见的帧数占比): {np.mean(all_ratios):.1%}")
        
    print("=" * 60)
    
    # 打印详细细节
    if missing:
        print(f"⚠️ 缺失的视频（前 5 个）：{missing[:5]}")
    if corrupted:
        print(f"⚠️ 损坏的视频（前 5 个）：{corrupted[:5]}")
    if empty_hands:
        print(f"⚠️ 无任何手部检测的视频（前 5 个）：{empty_hands[:5]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="configs/csl_daily/video_mapping.json")
    parser.add_argument("--skeleton_dir", default="processed/csl_daily/skeletons")
    args = parser.parse_args()
    
    verify_skeletons(args.mapping, args.skeleton_dir)
