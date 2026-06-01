"""
PyTorch Dataset and collate_fn for CSL-Daily continuous sign language translation.

The loader is schema-tolerant so it can work with either:
  - Task C's global index JSON, containing explicit paths and token IDs.
  - The raw CSL-Daily annotation JSON plus default processed directories.

Returned sample keys are stable; missing optional fields are represented by
empty tensors/strings rather than fabricated data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_records(raw: Any, subset: Optional[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = enumerate(raw)
    else:
        raise ValueError(f"Unsupported index JSON root type: {type(raw)!r}")

    for key, value in iterator:
        if not isinstance(value, dict):
            continue
        item_subset = value.get("subset", value.get("split"))
        if subset is not None and item_subset != subset:
            continue
        vid = value.get("vid", value.get("id", value.get("name", key)))
        record = dict(value)
        record["vid"] = str(vid)
        records.append(record)
    return records


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _ids_from_value(value: Any, vocab: Optional[Mapping[str, int]]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        if vocab is None:
            return []
        return [int(vocab[token]) for token in value.split() if token in vocab]
    return []


def _load_vocab(path: Optional[str | Path]) -> Optional[Dict[str, int]]:
    if not path:
        return None
    raw = _load_json(_resolve(path))
    if isinstance(raw, dict):
        if all(isinstance(v, int) for v in raw.values()):
            return {str(k): int(v) for k, v in raw.items()}
        for key in ("token_to_id", "word2id", "gloss2id", "text2id", "stoi"):
            value = raw.get(key)
            if isinstance(value, dict):
                return {str(k): int(v) for k, v in value.items()}
    raise ValueError(f"Cannot parse vocab file as token->id mapping: {path}")


def _load_npy_float32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)


def _to_stgcn_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert [T,V,C] to [C,T,V,1]."""
    if array.ndim != 3:
        raise ValueError(f"Expected skeleton stream [T,V,C], got {array.shape}")
    return torch.from_numpy(np.array(array.transpose(2, 0, 1), dtype=np.float32, copy=True)).unsqueeze(-1)


class CSLTDataset(Dataset):
    def __init__(
        self,
        index_json: str | Path = "scripts/data_prep/generate_csl_daily_splits.json",
        subset: Optional[str] = None,
        skeleton_dir: str | Path = "processed/csl_daily/skeletons",
        teacher_feature_dir: str | Path = "processed/csl_daily/teacher_features",
        teacher_logits_dir: str | Path = "processed/csl_daily/teacher_logits",
        gloss_vocab_path: Optional[str | Path] = None,
        text_vocab_path: Optional[str | Path] = None,
        require_files: bool = True,
        use_mmap: bool = True,
    ) -> None:
        self.index_json = _resolve(index_json)
        self.skeleton_dir = _resolve(skeleton_dir)
        self.teacher_feature_dir = _resolve(teacher_feature_dir)
        self.teacher_logits_dir = _resolve(teacher_logits_dir)
        self.gloss_vocab = _load_vocab(gloss_vocab_path)
        self.text_vocab = _load_vocab(text_vocab_path)
        self.require_files = require_files
        self.use_mmap = use_mmap

        if not self.index_json.exists():
            raise FileNotFoundError(
                f"Index JSON not found: {self.index_json}. "
                "Download/produce CSL-Daily annotations or Task C's global index first."
            )

        records = _normalize_records(_load_json(self.index_json), subset)
        self.samples: List[Dict[str, Any]] = []
        missing: List[str] = []
        for record in records:
            sample = self._prepare_record(record)
            missing_paths = [str(path) for path in self._required_paths(sample) if not path.exists()]
            if missing_paths:
                missing.append(f"{sample['vid']}: {', '.join(missing_paths)}")
                if require_files:
                    continue
            self.samples.append(sample)

        self.missing_files = missing
        if require_files and records and not self.samples:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                "No usable CSL-Daily samples found because required .npy files are missing. "
                f"First missing examples:\n{preview}"
            )

    def _prepare_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        vid = str(record["vid"])
        skeleton_path = _first_present(record, ("skeleton_path", "skeleton", "joints_path"))
        teacher_feature_path = _first_present(record, ("teacher_feature_path", "feature_path", "features_path"))
        teacher_logits_path = _first_present(record, ("teacher_logits_path", "logits_path"))

        return {
            "vid": vid,
            "skeleton_path": _resolve(skeleton_path) if skeleton_path else self.skeleton_dir / f"{vid}.npy",
            "teacher_feature_path": _resolve(teacher_feature_path)
            if teacher_feature_path
            else self.teacher_feature_dir / f"{vid}.npy",
            "teacher_logits_path": _resolve(teacher_logits_path)
            if teacher_logits_path
            else self.teacher_logits_dir / f"{vid}.npy",
            "gloss": _first_present(record, ("gloss", "glosses", "gloss_sequence"), ""),
            "text": _first_present(record, ("text", "translation", "chinese", "sentence"), ""),
            "gloss_ids": _ids_from_value(_first_present(record, ("gloss_ids", "gloss_id")), self.gloss_vocab),
            "text_ids": _ids_from_value(_first_present(record, ("text_ids", "text_id", "translation_ids")), self.text_vocab),
            "raw": dict(record),
        }

    @staticmethod
    def _required_paths(sample: Mapping[str, Any]) -> List[Path]:
        return [
            Path(sample["skeleton_path"]),
            Path(sample["teacher_feature_path"]),
            Path(sample["teacher_logits_path"]),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        skeleton = _load_npy_float32(Path(sample["skeleton_path"]))
        if skeleton.ndim != 3:
            raise ValueError(f"{sample['vid']} skeleton must be [T,V,C], got {skeleton.shape}")

        if skeleton.shape[-1] >= 9:
            joints = skeleton[..., 0:3]
            bones = skeleton[..., 3:6]
            motion = skeleton[..., 6:9]
        elif skeleton.shape[-1] >= 3:
            joints = skeleton[..., 0:3]
            bones = np.zeros_like(joints)
            motion = np.zeros_like(joints)
        else:
            raise ValueError(f"{sample['vid']} skeleton channel dim must be >=3, got {skeleton.shape}")

        teacher_features = _load_npy_float32(Path(sample["teacher_feature_path"]))
        teacher_logits = _load_npy_float32(Path(sample["teacher_logits_path"]))
        if teacher_features.ndim == 1:
            teacher_features = teacher_features[None, :]
        if teacher_logits.ndim == 1:
            teacher_logits = teacher_logits[None, :]

        gloss_ids = torch.tensor(sample["gloss_ids"], dtype=torch.long)
        text_ids = torch.tensor(sample["text_ids"], dtype=torch.long)

        return {
            "vid": sample["vid"],
            "joints": _to_stgcn_tensor(joints),
            "bones": _to_stgcn_tensor(bones),
            "motion": _to_stgcn_tensor(motion),
            "input_length": torch.tensor(joints.shape[0], dtype=torch.long),
            "teacher_features": torch.from_numpy(np.array(teacher_features, dtype=np.float32, copy=True)),
            "teacher_feature_length": torch.tensor(teacher_features.shape[0], dtype=torch.long),
            "teacher_logits": torch.from_numpy(np.array(teacher_logits, dtype=np.float32, copy=True)),
            "teacher_logits_length": torch.tensor(teacher_logits.shape[0], dtype=torch.long),
            "gloss_ids": gloss_ids,
            "gloss_length": torch.tensor(gloss_ids.numel(), dtype=torch.long),
            "text_ids": text_ids,
            "text_length": torch.tensor(text_ids.numel(), dtype=torch.long),
            "gloss": sample["gloss"],
            "text": sample["text"],
            "paths": {
                "skeleton": str(sample["skeleton_path"]),
                "teacher_features": str(sample["teacher_feature_path"]),
                "teacher_logits": str(sample["teacher_logits_path"]),
            },
        }


def _pad_time_stgcn(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pad [C,T,V,M] tensors along T and stack to [B,C,T,V,M]."""
    max_t = max(int(t.shape[1]) for t in tensors)
    padded: List[torch.Tensor] = []
    for tensor in tensors:
        out = tensor.new_zeros((tensor.shape[0], max_t, tensor.shape[2], tensor.shape[3]))
        out[:, : tensor.shape[1]] = tensor
        padded.append(out)
    return torch.stack(padded, dim=0)


def _pad_time_2d(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pad [T,D] tensors to [B,T,D]."""
    max_t = max(int(t.shape[0]) for t in tensors)
    max_d = max(int(t.shape[1]) for t in tensors)
    padded: List[torch.Tensor] = []
    for tensor in tensors:
        out = tensor.new_zeros((max_t, max_d))
        out[: tensor.shape[0], : tensor.shape[1]] = tensor
        padded.append(out)
    return torch.stack(padded, dim=0)


def cslt_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    gloss_ids = [item["gloss_ids"] for item in batch]
    text_ids = [item["text_ids"] for item in batch]

    return {
        "vid": [item["vid"] for item in batch],
        "joints": _pad_time_stgcn([item["joints"] for item in batch]),
        "bones": _pad_time_stgcn([item["bones"] for item in batch]),
        "motion": _pad_time_stgcn([item["motion"] for item in batch]),
        "input_lengths": torch.stack([item["input_length"] for item in batch]),
        "teacher_features": _pad_time_2d([item["teacher_features"] for item in batch]),
        "teacher_feature_lengths": torch.stack([item["teacher_feature_length"] for item in batch]),
        "teacher_logits": _pad_time_2d([item["teacher_logits"] for item in batch]),
        "teacher_logits_lengths": torch.stack([item["teacher_logits_length"] for item in batch]),
        "gloss_ids": pad_sequence(gloss_ids, batch_first=True, padding_value=0)
        if any(x.numel() for x in gloss_ids)
        else torch.zeros((len(batch), 0), dtype=torch.long),
        "gloss_lengths": torch.stack([item["gloss_length"] for item in batch]),
        "text_ids": pad_sequence(text_ids, batch_first=True, padding_value=0)
        if any(x.numel() for x in text_ids)
        else torch.zeros((len(batch), 0), dtype=torch.long),
        "text_lengths": torch.stack([item["text_length"] for item in batch]),
        "gloss": [item["gloss"] for item in batch],
        "text": [item["text"] for item in batch],
        "paths": [item["paths"] for item in batch],
    }


__all__ = ["CSLTDataset", "cslt_collate_fn"]
