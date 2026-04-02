#!/usr/bin/env python
import argparse
import json
import os
import pickle

import numpy as np
from numpy.lib.format import open_memmap


"""
Convert custom WLASL keypoints to ST-GCN format.

Input annotation json format (list):
[
  {
    "sample_name": "video_0001",
    "subset": "train",            # train or val
    "label": 12,                    # int label id (0-based) or class name string
    "keypoint_path": "kp/video_0001.npy"
  }
]

Supported keypoint tensor shapes per sample:
- (T, V, C)
- (C, T, V)
- (C, T, V, M)
- (T, V, C, M)

Output:
- <out_folder>/train_data.npy, <out_folder>/train_label.pkl
- <out_folder>/val_data.npy, <out_folder>/val_label.pkl
"""


def _load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        if "samples" in obj and isinstance(obj["samples"], list):
            samples = obj["samples"]
        else:
            raise ValueError("Annotation json dict must contain a 'samples' list.")
    elif isinstance(obj, list):
        samples = obj
    else:
        raise ValueError("Annotation json must be a list or a dict with key 'samples'.")

    required = {"sample_name", "subset", "label", "keypoint_path"}
    for i, s in enumerate(samples):
        miss = required - set(s.keys())
        if miss:
            raise ValueError(f"Sample index {i} missing keys: {sorted(miss)}")

    return samples


def _to_ctvm(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (T,V,C)
        if arr.shape[-1] in (2, 3):
            arr = arr.transpose(2, 0, 1)  # -> (C,T,V)
        # already (C,T,V)
        elif arr.shape[0] in (2, 3):
            pass
        else:
            raise ValueError(f"Ambiguous 3D keypoint shape: {arr.shape}")
        arr = arr[..., np.newaxis]  # -> (C,T,V,1)

    elif arr.ndim == 4:
        # (T,V,C,M)
        if arr.shape[2] in (2, 3):
            arr = arr.transpose(2, 0, 1, 3)
        # already (C,T,V,M)
        elif arr.shape[0] in (2, 3):
            pass
        else:
            raise ValueError(f"Ambiguous 4D keypoint shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported keypoint ndim={arr.ndim}, shape={arr.shape}")

    # force C=3
    c, t, v, m = arr.shape
    if c == 2:
        score = np.ones((1, t, v, m), dtype=arr.dtype)
        arr = np.concatenate([arr, score], axis=0)
    elif c != 3:
        raise ValueError(f"Expected C in {{2,3}}, got C={c}")

    return arr.astype("float32", copy=False)


def _resample_or_pad(data_ctvm, max_frame):
    c, t, v, m = data_ctvm.shape
    if t == max_frame:
        return data_ctvm

    out = np.zeros((c, max_frame, v, m), dtype=np.float32)

    if t == 0:
        return out

    if t > max_frame:
        idx = np.linspace(0, t - 1, num=max_frame, dtype=np.int64)
        out = data_ctvm[:, idx, :, :]
    else:
        out[:, :t, :, :] = data_ctvm

    return out


def _normalize_label(samples):
    # Keep integer labels unchanged; map string labels to contiguous ids.
    str_labels = [s["label"] for s in samples if isinstance(s["label"], str)]
    label_map = {}
    if str_labels:
        classes = sorted(set(str_labels))
        label_map = {name: i for i, name in enumerate(classes)}

    for s in samples:
        if isinstance(s["label"], str):
            s["label"] = label_map[s["label"]]
        else:
            s["label"] = int(s["label"])

    return label_map


def _write_split(samples, keypoint_root, out_folder, split, max_frame, num_person_out):
    split_samples = [s for s in samples if s["subset"] == split]
    if not split_samples:
        raise ValueError(f"No samples found for subset='{split}'")

    first_path = os.path.join(keypoint_root, split_samples[0]["keypoint_path"])
    first_arr = _to_ctvm(np.load(first_path))
    _, _, v, _ = first_arr.shape

    data_path = os.path.join(out_folder, f"{split}_data.npy")
    label_path = os.path.join(out_folder, f"{split}_label.pkl")

    fp = open_memmap(
        data_path,
        dtype="float32",
        mode="w+",
        shape=(len(split_samples), 3, max_frame, v, num_person_out),
    )

    names = []
    labels = []

    for i, s in enumerate(split_samples):
        kp_path = os.path.join(keypoint_root, s["keypoint_path"])
        data = _to_ctvm(np.load(kp_path))
        data = _resample_or_pad(data, max_frame)
        data = data[:, :, :, :num_person_out]

        # If sample has fewer persons than requested, keep zero padding.
        c, t, v_cur, m_cur = data.shape
        if v_cur != v:
            raise ValueError(
                f"Joint number mismatch at {kp_path}: expected V={v}, got V={v_cur}"
            )

        fp[i, :, :, :, :m_cur] = data
        names.append(s["sample_name"])
        labels.append(int(s["label"]))

    with open(label_path, "wb") as f:
        pickle.dump((names, labels), f)

    print(f"[{split}] samples={len(split_samples)}, V={v}, output={data_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ST-GCN input from WLASL keypoints")
    parser.add_argument("--annotation", required=True, help="Path to annotation json")
    parser.add_argument("--keypoint_root", default=".", help="Root directory for keypoint_path")
    parser.add_argument("--out_folder", required=True, help="Output folder for npy/pkl")
    parser.add_argument("--max_frame", type=int, default=300, help="Temporal length T")
    parser.add_argument("--num_person_out", type=int, default=1, help="Output person number M")
    parser.add_argument("--expected_num_class", type=int, default=2000, help="Sanity check class count")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    samples = _load_annotations(args.annotation)
    label_map = _normalize_label(samples)

    labels = [int(s["label"]) for s in samples]
    if min(labels) < 0:
        raise ValueError("Label id must be >= 0")

    class_count = len(set(labels))
    print(f"Detected class count: {class_count}")
    if args.expected_num_class > 0 and class_count != args.expected_num_class:
        print(
            "WARNING: detected class count != expected_num_class "
            f"({class_count} vs {args.expected_num_class})"
        )

    _write_split(samples, args.keypoint_root, args.out_folder, "train", args.max_frame, args.num_person_out)
    _write_split(samples, args.keypoint_root, args.out_folder, "val", args.max_frame, args.num_person_out)

    if label_map:
        map_path = os.path.join(args.out_folder, "label_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"Saved string-label map to: {map_path}")


if __name__ == "__main__":
    main()
