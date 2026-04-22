import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data as data_utl


class StudentFeatureDatasetNew(data_utl.Dataset):
    """
    Returns:
        (skeleton, teacher_logits, teacher_features, label, video_id)
    """

    def __init__(
        self,
        split_json: str,
        skeleton_dir: str,
        logits_dir: str,
        teacher_feature_dir: str,
        subset: str = "train",
        seq_len: int = 64,
        skeleton_transform=None,
        logits_transform=None,
        teacher_feature_transform=None,
        max_class_id_exclusive: int = None,
        max_samples: int = None,
        **kwargs,
    ):
        self.split_json = split_json
        self.skeleton_dir = skeleton_dir
        self.logits_dir = logits_dir
        self.teacher_feature_dir = teacher_feature_dir
        self.subset = subset
        self.seq_len = seq_len
        self.skeleton_transform = skeleton_transform
        self.logits_transform = logits_transform
        self.teacher_feature_transform = teacher_feature_transform
        self.max_class_id_exclusive = max_class_id_exclusive
        self.max_samples = max_samples

        self.skeleton_index = self.new_build_skeleton_index(self.skeleton_dir)
        self.samples = self.new_build_samples()

    @staticmethod
    def new_build_skeleton_index(skeleton_dir: str) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for root, _, files in os.walk(skeleton_dir):
            for filename in files:
                if filename.endswith(".npy"):
                    index[os.path.splitext(filename)[0]] = os.path.join(root, filename)
        return index

    def new_build_samples(self) -> List[Tuple[str, str, str, str, int]]:
        with open(self.split_json, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("split json must be dict: video_id -> metadata")

        samples: List[Tuple[str, str, str, str, int]] = []
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("subset") != self.subset:
                continue
            if "action" not in meta or len(meta["action"]) == 0:
                continue

            label = int(meta["action"][0])
            if self.max_class_id_exclusive is not None and label >= int(self.max_class_id_exclusive):
                continue
            skeleton_path = self.skeleton_index.get(str(vid))
            logits_path = os.path.join(self.logits_dir, f"{vid}.npy")
            teacher_feature_path = os.path.join(self.teacher_feature_dir, f"{vid}.npy")

            if skeleton_path is None:
                continue
            if not os.path.exists(logits_path):
                continue
            if not os.path.exists(teacher_feature_path):
                continue

            samples.append((str(vid), skeleton_path, logits_path, teacher_feature_path, label))

        if self.max_samples is not None:
            samples = samples[: int(self.max_samples)]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def new_pad_or_trim(self, array: np.ndarray) -> np.ndarray:
        frame_count = array.shape[0]
        if frame_count == self.seq_len:
            return array
        if frame_count > self.seq_len:
            return array[: self.seq_len]

        pad = np.zeros((self.seq_len - frame_count, array.shape[1], array.shape[2]), dtype=array.dtype)
        return np.concatenate([array, pad], axis=0)

    def __getitem__(self, index: int):
        vid, skeleton_path, logits_path, teacher_feature_path, label = self.samples[index]

        skeleton = np.load(skeleton_path).astype(np.float32)
        skeleton = self.new_pad_or_trim(skeleton)
        skeleton = skeleton.reshape(self.seq_len, -1)

        logits = np.load(logits_path).astype(np.float32)
        teacher_features = np.load(teacher_feature_path).astype(np.float32)
        if teacher_features.ndim == 1:
            teacher_features = teacher_features[None, :]

        if self.skeleton_transform is not None:
            skeleton = self.skeleton_transform(skeleton)
        if self.logits_transform is not None:
            logits = self.logits_transform(logits)
        if self.teacher_feature_transform is not None:
            teacher_features = self.teacher_feature_transform(teacher_features)

        skeleton_tensor = torch.from_numpy(skeleton)
        logits_tensor = torch.from_numpy(logits)
        teacher_feature_tensor = torch.from_numpy(teacher_features)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return skeleton_tensor, logits_tensor, teacher_feature_tensor, label_tensor, vid
