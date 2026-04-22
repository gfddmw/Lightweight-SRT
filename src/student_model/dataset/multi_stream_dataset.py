#动态计算 Bones 与 Motion
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import os
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.common.transforms.skeleton_transforms import SkeletonCompose, TemporalCropOrPad

class MultiStreamSkeletonDataset(Dataset):
    """
    多流骨架数据集 (Joints / Bones / Motion)
    适配 MediaPipe 单手 21 点 .npy 格式 & ST-GCN (N, C, T, V, M) 输入规范
    """
    def __init__(self, index_path, data_dir, topology, label_map_path=None, 
                 target_len=64, is_train=True, normalize_wrist=True, **kwargs):
        
        if not os.path.isabs(index_path):
            index_path = PROJECT_ROOT / index_path
        else:
            index_path = Path(index_path)
            
        if not os.path.isabs(data_dir):
            data_dir = PROJECT_ROOT / data_dir
        else:
            data_dir = Path(data_dir)
            
        if label_map_path and not os.path.isabs(label_map_path):
            label_map_path = PROJECT_ROOT / label_map_path

        # 1. 加载清洗后的索引
        with open(index_path, 'r') as f:
            self.sample_ids = json.load(f)
        self.data_dir = data_dir
        self.topology = topology  # list of tuples: [(src_idx, tgt_idx), ...]
        self.target_len = target_len
        self.normalize_wrist = normalize_wrist
        
        # 2. 加载标签映射 (WLASL 强烈建议使用独立 JSON 映射表)
        self.label_map = None
        if label_map_path:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        else:
            print("⚠️ 未提供 label_map_path，将尝试从文件名解析标签 (不推荐用于生产)")

        # 3. 构建变换管道
        base_transforms = [TemporalCropOrPad(target_len=target_len, padding_mode='zero')]
        if is_train:
            # 训练集：增强 + 时间对齐
            from src.common.transforms.skeleton_transforms import RandomRotateSkeleton
            base_transforms.insert(0, RandomRotateSkeleton(max_angle=15, center_on_root=True))
            
        self.transforms = SkeletonCompose(base_transforms)

    def __len__(self):
        return len(self.sample_ids)

    def _load_sample(self, vid_id):
        """加载 .npy 骨架文件，确保形状为 (T, 21, 3)"""
        npy_path = self.data_dir / f"{vid_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"骨架文件缺失: {npy_path}")
            
        data = np.load(npy_path, allow_pickle=True).astype(np.float32)
        
        # 兼容处理：确保为 (T, K, 3)
        if data.ndim == 2:  # (T, K*3) 扁平格式
            if data.shape[1] % 3 == 0:
                data = data.reshape(data.shape[0], -1, 3)
        elif data.ndim == 3 and data.shape[-1] == 2:  # 缺置信度列
            data = np.dstack([data, np.zeros(data.shape[:-1], dtype=np.float32)])
            
        return data

    def _get_label(self, vid_id):
        """安全获取标签"""
        # 🚀 兜底机制：如果外部传参失败，直接在这里强行加载绝对路径
        if self.label_map is None:
            map_path = PROJECT_ROOT / "processed" / "label_map.json"
            if map_path.exists():
                with open(map_path, 'r') as f:
                    self.label_map = json.load(f)

        # 正常逻辑
        if self.label_map:
            return self.label_map.get(str(vid_id), self.label_map.get(vid_id, -1))
        
        # 降级解析：假设文件名包含数字标签
        try:
            parts = vid_id.replace('-', '_').split('_')
            for p in parts:
                if p.isdigit():
                    return int(p)
            return -1
        except:
            return -1

    def _compute_bones(self, joints):
        """基于拓扑计算骨骼向量 (T, V, C)"""
        bones = np.zeros_like(joints)
        max_v = joints.shape[1]
        for src, tgt in self.topology:
            if src < max_v and tgt < max_v:
                bones[:, tgt] = joints[:, tgt] - joints[:, src]
        return bones

    def _compute_motion(self, joints):
        """计算时间差分运动特征 (T, V, C)"""
        motion = np.zeros_like(joints)
        T = joints.shape[0]
        if T > 1:
            motion[:-1] = joints[1:] - joints[:-1]
            # 最后一帧运动置零 (ST-GCN 标准实践，避免引入虚假边界信号)
            motion[-1] = 0.0
        return motion

    def __getitem__(self, idx):
        vid_id = self.sample_ids[idx]
        raw_data = self._load_sample(vid_id)
        
        # 仅取 x, y 坐标 (MediaPipe 的 z 为相对深度，手语识别中 2D 投影更稳定)
        joints = raw_data[:, :, :2] 
        
        # 可选：以腕部(索引0)为中心归一化，消除位置偏差
        if self.normalize_wrist:
            wrist = joints[:, 0:1, :]  # (T, 1, 2)
            joints = joints - wrist
            
        # 1. 数据增强 & 时间对齐
        joints = self.transforms(joints)
        
        # 2. 计算多流特征 (在增强/对齐后计算，保证物理一致性)
        bones = self._compute_bones(joints)
        motion = self._compute_motion(joints)
        
        # 3. 转换为 ST-GCN 标准张量格式: (C, T, V, M) -> M=1
        def to_stgcn_tensor(x):
            # x: (T, V, C) -> (C, T, V) -> (C, T, V, 1)
            return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(-1).contiguous()

        label = self._get_label(vid_id)
        if label == -1:
            # 🚀 容错：如果实在找不到标签，分配默认类别 0，避免中断训练
            label = 0  
        return {
            "joints": to_stgcn_tensor(joints),
            "bones": to_stgcn_tensor(bones),
            "motion": to_stgcn_tensor(motion)
        }, torch.tensor(label, dtype=torch.long)