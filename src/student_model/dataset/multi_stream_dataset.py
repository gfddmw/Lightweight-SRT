# 深度优化版：支持训练集读取预计算特征 + mmap 快速 IO
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import os
import sys

def _find_project_root():
    """自动查找项目根目录"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "processed").exists():
            return parent
    return Path(__file__).resolve().parents[3]

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.transforms.skeleton_transforms import TemporalCropOrPad

class MultiStreamSkeletonDataset(Dataset):
    def __init__(self, index_path=None, split_json=None, subset='train', data_dir=None, topology=None, label_map_path=None,
                 target_len=64, is_train=True, normalize_wrist=True, in_channels=2,
                 teacher_logits_dir=None, teacher_feature_dir=None, bones_dir=None, motion_dir=None, **kwargs):

        if split_json:
            if not os.path.isabs(split_json):
                split_json = PROJECT_ROOT / split_json
            with open(split_json, 'r', encoding='utf-8') as f:
                self.split_json_data = json.load(f)
            self.sample_ids = [vid for vid, meta in self.split_json_data.items() if meta.get('subset') == subset]
        elif index_path:
            index_path = PROJECT_ROOT / index_path if not os.path.isabs(index_path) else Path(index_path)
            with open(index_path, 'r') as f:
                self.sample_ids = json.load(f)
        else:
            raise ValueError("Must provide 'split_json' or 'index_path'")

        self.data_dir = PROJECT_ROOT / data_dir if not os.path.isabs(data_dir) else Path(data_dir)
        self.topology = topology
        self.target_len = target_len
        self.normalize_wrist = normalize_wrist
        self.in_channels = in_channels
        
        def _resolve_dir(d):
            if not d: return None
            return PROJECT_ROOT / d if not os.path.isabs(d) else Path(d)

        self.teacher_logits_dir = _resolve_dir(teacher_logits_dir)
        self.teacher_feature_dir = _resolve_dir(teacher_feature_dir)
        self.bones_dir = _resolve_dir(bones_dir)
        self.motion_dir = _resolve_dir(motion_dir)

        self.is_train = is_train
        if is_train:
            from src.common.transforms.skeleton_transforms import RandomRotateSkeleton
            self.spatial_transform = RandomRotateSkeleton(max_angle=15, center_on_root=True)
        else:
            self.spatial_transform = None
        
        self.temporal_transform = TemporalCropOrPad(target_len=target_len, padding_mode='zero', random=is_train)

    def __len__(self):
        return len(self.sample_ids)

    def _load_npy(self, path):
        """使用 mmap_mode 提升读取速度"""
        try:
            return np.load(path, allow_pickle=True, mmap_mode='r').astype(np.float32)
        except:
            return None

    def _get_label(self, vid_id):
        if hasattr(self, 'split_json_data') and str(vid_id) in self.split_json_data:
            meta = self.split_json_data[str(vid_id)]
            if 'action' in meta and len(meta['action']) > 0:
                return int(meta['action'][0])
        return 0

    def _compute_bones(self, joints):
        bones = np.zeros_like(joints)
        max_v = joints.shape[1]
        src_indices = [e[0] for e in self.topology if e[0] < max_v and e[1] < max_v]
        tgt_indices = [e[1] for e in self.topology if e[0] < max_v and e[1] < max_v]
        if src_indices:
            bones[:, tgt_indices] = joints[:, tgt_indices] - joints[:, src_indices]
        return bones

    def _compute_motion(self, joints):
        motion = np.zeros_like(joints)
        if joints.shape[0] > 1:
            motion[:-1] = joints[1:] - joints[:-1]
        return motion

    def __getitem__(self, idx):
        vid_id = self.sample_ids[idx]
        
        # 1. 加载 Joints
        joints_path = self.data_dir / f"{vid_id}.npy"
        joints = self._load_npy(joints_path)
        if joints is None: return self.__getitem__(0) # 容错
        
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        joints = joints[:, :, :self.in_channels]

        if self.normalize_wrist:
            joints[:, :, :2] -= joints[:, 0:1, :2]

        if self.spatial_transform:
            joints = self.spatial_transform(joints)

        # 2. 优先加载预计算的 Bones/Motion (不再区分训练测试集)
        bones = None
        motion = None
        
        if self.bones_dir:
            bones = self._load_npy(self.bones_dir / f"{vid_id}.npy")
        if self.motion_dir:
            motion = self._load_npy(self.motion_dir / f"{vid_id}.npy")

        if bones is None or bones.shape[-1] != self.in_channels:
            bones = self._compute_bones(joints)
        if motion is None or motion.shape[-1] != self.in_channels:
            motion = self._compute_motion(joints)

        # 3. 时间对齐与转换
        joints = self.temporal_transform(joints)
        bones = self.temporal_transform(bones)
        motion = self.temporal_transform(motion)

        def to_stgcn_tensor(x):
            return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(-1).contiguous()

        data_dict = {
            "joints": to_stgcn_tensor(joints),
            "bones": to_stgcn_tensor(bones),
            "motion": to_stgcn_tensor(motion)
        }
        
        label_t = torch.tensor(self._get_label(vid_id), dtype=torch.long)

        # 4. 加载蒸馏辅助数据 (mmap)
        teacher_logits = None
        if self.teacher_logits_dir:
            lp = self.teacher_logits_dir / f"{vid_id}.npy"
            if lp.exists():
                teacher_logits = torch.from_numpy(np.load(lp, mmap_mode='r').astype(np.float32))

        teacher_features = None
        if self.teacher_feature_dir:
            fp = self.teacher_feature_dir / f"{vid_id}.npy"
            if fp.exists():
                feat = np.load(fp, mmap_mode='r').astype(np.float32)
                if feat.size % 1024 == 0 and feat.size > 0:
                    feat = feat.reshape(-1, 1024).mean(axis=0, keepdims=True)
                elif feat.ndim == 1:
                    feat = feat[None, :]
                teacher_features = torch.from_numpy(feat.copy())

        if teacher_logits is not None and teacher_features is not None:
            return data_dict, teacher_logits, teacher_features, label_t, vid_id
        elif teacher_logits is not None:
            return data_dict, teacher_logits, label_t, vid_id
        
        return data_dict, label_t
