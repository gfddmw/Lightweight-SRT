import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data as data_utl


class StudentDataset(data_utl.Dataset):
    """
    Returns (skeleton, teacher_logits, label, video_id) for student training.
    """

    def __init__(
        self,
        split_json: str,
        skeleton_dir: str,
        logits_dir: str,
        subset: str = "train",
        seq_len: int = 64,
        skeleton_transform=None,
        logits_transform=None,
        **kwargs,
    ):
        self.split_json = split_json
        self.skeleton_dir = skeleton_dir
        self.logits_dir = logits_dir
        self.subset = subset
        self.seq_len = seq_len
        self.skeleton_transform = skeleton_transform
        self.logits_transform = logits_transform

        self.skeleton_index = self._build_skeleton_index(self.skeleton_dir)
        self.samples = self._build_samples()

    @staticmethod
    def _build_skeleton_index(skeleton_dir: str) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for root, _, files in os.walk(skeleton_dir):
            for f in files:
                if f.endswith(".npy"):
                    index[os.path.splitext(f)[0]] = os.path.join(root, f)
        return index

    def _build_samples(self) -> List[Tuple[str, str, str, int]]:
        with open(self.split_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("split json must be dict: video_id -> metadata")

        samples: List[Tuple[str, str, str, int]] = []
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("subset") != self.subset:
                continue
            if "action" not in meta or len(meta["action"]) == 0:
                continue
            label = int(meta["action"][0])

            skeleton_path = self.skeleton_index.get(str(vid))
            logits_path = os.path.join(self.logits_dir, f"{vid}.npy")
            if skeleton_path is None or not os.path.exists(logits_path):
                continue

            samples.append((str(vid), skeleton_path, logits_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_or_trim(self, x: np.ndarray) -> np.ndarray:
        t = x.shape[0]
        if t == self.seq_len:
            return x
        if t > self.seq_len:
            return x[: self.seq_len]
        pad = np.zeros((self.seq_len - t, x.shape[1], x.shape[2]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, index: int):
        vid, skeleton_path, logits_path, label = self.samples[index]

        skeleton = np.load(skeleton_path).astype(np.float32)
        skeleton = self._pad_or_trim(skeleton)
        skeleton = skeleton.reshape(self.seq_len, -1)  # (T, F)

        logits = np.load(logits_path).astype(np.float32)

        if self.skeleton_transform is not None:
            skeleton = self.skeleton_transform(skeleton)
        if self.logits_transform is not None:
            logits = self.logits_transform(logits)

        skeleton_t = torch.from_numpy(skeleton)
        logits_t = torch.from_numpy(logits)
        label_t = torch.tensor(label, dtype=torch.long)

        return skeleton_t, logits_t, label_t, vid
