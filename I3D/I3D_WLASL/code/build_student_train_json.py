import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np


def load_split_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("split json must be dict: video_id -> metadata")
    return data


def build_skeleton_index(skeleton_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(skeleton_dir):
        for f in files:
            if not f.endswith(".npy"):
                continue
            stem = os.path.splitext(f)[0]
            index[stem] = os.path.join(root, f)
    return index


def has_logits(logits_dir: str, vid: str) -> bool:
    return os.path.exists(os.path.join(logits_dir, f"{vid}.npy"))


def calc_missing_ratio(skeleton_path: str) -> Tuple[float, int]:
    arr = np.load(skeleton_path)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected skeleton shape {arr.shape}, expected (T,21,3): {skeleton_path}")
    if arr.shape[0] == 0:
        return 1.0, 0
    missing_mask = np.all(arr == 0.0, axis=(1, 2))
    missing_count = int(missing_mask.sum())
    missing_ratio = float(missing_count) / float(arr.shape[0])
    return missing_ratio, int(arr.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter bad skeleton samples and rebuild train json.")
    parser.add_argument("--input_json", default="preprocess/nslt_2000.json")
    parser.add_argument("--output_json", default="preprocess/nslt_2000_train_filtered.json")
    parser.add_argument("--bad_ids_json", default="preprocess/bad_samples.json")
    parser.add_argument("--subset", default="train", choices=["train", "val", "test"])
    parser.add_argument("--skeleton_dir", default="output_skeletons")
    parser.add_argument("--logits_dir", default="teacher_logits_2000")
    parser.add_argument(
        "--bad_missing_threshold",
        type=float,
        default=0.5,
        help="Mark as bad when missing_ratio > threshold.",
    )
    args = parser.parse_args()

    data = load_split_json(args.input_json)
    skeleton_index = build_skeleton_index(args.skeleton_dir)

    filtered: Dict[str, dict] = {}
    bad_info: Dict[str, dict] = {}

    total_target = 0
    kept = 0
    removed_missing_skeleton = 0
    removed_missing_logits = 0
    removed_bad_quality = 0
    removed_invalid_skeleton = 0

    for vid, meta in data.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("subset") != args.subset:
            continue
        total_target += 1

        skeleton_path = skeleton_index.get(str(vid))
        if skeleton_path is None:
            removed_missing_skeleton += 1
            bad_info[str(vid)] = {"reason": "missing_skeleton"}
            continue

        if not has_logits(args.logits_dir, str(vid)):
            removed_missing_logits += 1
            bad_info[str(vid)] = {"reason": "missing_logits"}
            continue

        try:
            missing_ratio, num_frames = calc_missing_ratio(skeleton_path)
        except Exception as e:
            removed_invalid_skeleton += 1
            bad_info[str(vid)] = {"reason": "invalid_skeleton", "error": str(e)}
            continue

        if missing_ratio > args.bad_missing_threshold:
            removed_bad_quality += 1
            bad_info[str(vid)] = {
                "reason": "bad_quality",
                "missing_ratio": missing_ratio,
                "num_frames": num_frames,
            }
            continue

        filtered[str(vid)] = meta
        kept += 1

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    with open(args.bad_ids_json, "w", encoding="utf-8") as f:
        json.dump(bad_info, f, ensure_ascii=False, indent=2)

    print(f"[INFO] subset={args.subset}")
    print(f"[INFO] total target ids: {total_target}")
    print(f"[INFO] kept ids: {kept}")
    print(f"[INFO] removed missing skeleton: {removed_missing_skeleton}")
    print(f"[INFO] removed missing logits: {removed_missing_logits}")
    print(f"[INFO] removed bad quality (> {args.bad_missing_threshold:.2f}): {removed_bad_quality}")
    print(f"[INFO] removed invalid skeleton: {removed_invalid_skeleton}")
    print(f"[INFO] output train json: {args.output_json}")
    print(f"[INFO] bad sample report: {args.bad_ids_json}")


if __name__ == "__main__":
    main()

