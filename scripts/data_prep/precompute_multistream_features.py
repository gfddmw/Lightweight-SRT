import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

def precompute_features(data_dir, index_paths, output_bones, output_motion, topology, in_channels=3):
    """
    预计算 Bones 和 Motion 特征并保存为 .npy 文件
    """
    data_dir = Path(data_dir)
    output_bones = Path(output_bones)
    output_motion = Path(output_motion)
    
    os.makedirs(output_bones, exist_ok=True)
    os.makedirs(output_motion, exist_ok=True)
    
    # 汇总所有需要处理的 sample_id
    sample_ids = set()
    for idx_path in index_paths:
        if os.path.exists(idx_path):
            with open(idx_path, 'r') as f:
                sample_ids.update(json.load(f))
    
    print(f"🚀 开始预计算 {len(sample_ids)} 个样本的特征 (in_channels={in_channels})...")
    
    for vid_id in tqdm(sample_ids):
        npy_path = data_dir / f"{vid_id}.npy"
        if not npy_path.exists():
            continue
            
        # 1. 加载原始数据 (T, V, C)
        data = np.load(npy_path, allow_pickle=True).astype(np.float32)
        if data.ndim == 2:
            data = data.reshape(data.shape[0], -1, 3)
        elif data.ndim == 3 and data.shape[-1] == 2:
            data = np.dstack([data, np.zeros(data.shape[:-1], dtype=np.float32)])
        
        # 动态取坐标 (与 dataset 逻辑一致)
        joints = data[:, :, :in_channels]
        
        # 2. 计算 Bones
        bones = np.zeros_like(joints)
        for src, tgt in topology:
            if src < joints.shape[1] and tgt < joints.shape[1]:
                bones[:, tgt] = joints[:, tgt] - joints[:, src]
        
        # 3. 计算 Motion
        motion = np.zeros_like(joints)
        if joints.shape[0] > 1:
            motion[:-1] = joints[1:] - joints[:-1]
            
        # 4. 保存
        np.save(output_bones / f"{vid_id}.npy", bones)
        np.save(output_motion / f"{vid_id}.npy", motion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='processed/skeletons')
    parser.add_argument('--output_bones', default='processed/bones')
    parser.add_argument('--output_motion', default='processed/motion')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of channels to use (e.g. 2 for xy, 3 for xyz/conf)')
    args = parser.parse_args()
    
    # 默认 MediaPipe 手部拓扑
    topology = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    
    index_files = ['processed/train_indices.json', 'processed/test_indices.json']
    
    precompute_features(args.data_dir, index_files, args.output_bones, args.output_motion, topology, in_channels=args.in_channels)
    print("✨ 预计算完成！请在配置文件中更新对应的特征目录。")
