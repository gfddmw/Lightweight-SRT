import argparse
import json
import os
from typing import Dict, List, Set


def load_split_ids(split_json_path: str, subset: str = "train") -> Set[str]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("split json must be dict: video_id -> metadata")

    ids: Set[str] = set()
    for vid, meta in data.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("subset") == subset:
            ids.add(str(vid))
    return ids


def collect_npy_stems(root_dir: str, recursive: bool = True) -> Set[str]:
    stems: Set[str] = set()
    if recursive:
        for current_root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".npy"):
                    stems.add(os.path.splitext(f)[0])
    else:
        for f in os.listdir(root_dir):
            if f.endswith(".npy"):
                stems.add(os.path.splitext(f)[0])
    return stems


def summarize_alignment(
    train_ids: Set[str], skeleton_ids: Set[str], logits_ids: Set[str]
) -> Dict[str, List[str]]:
    inter_all = train_ids & skeleton_ids & logits_ids
    missing_skeleton = sorted(train_ids - skeleton_ids)
    missing_logits = sorted(train_ids - logits_ids)
    skeleton_without_train = sorted(skeleton_ids - train_ids)
    logits_without_train = sorted(logits_ids - train_ids)
    skeleton_without_logits = sorted(skeleton_ids - logits_ids)
    logits_without_skeleton = sorted(logits_ids - skeleton_ids)

    return {
        "aligned_all_ids": sorted(inter_all),
        "missing_skeleton_from_train": missing_skeleton,
        "missing_logits_from_train": missing_logits,
        "skeleton_without_train": skeleton_without_train,
        "logits_without_train": logits_without_train,
        "skeleton_without_logits": skeleton_without_logits,
        "logits_without_skeleton": logits_without_skeleton,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ID alignment among train json, skeletons and teacher logits.")
    parser.add_argument("--split_json", default="../../data/WLASL2000/preprocess/nslt_2000.json")
    parser.add_argument("--skeleton_dir", default="../../processed/skeletons")
    parser.add_argument("--logits_dir", default="../../processed/logits")
    parser.add_argument("--subset", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--report_json",
        default="../../data/WLASL2000/preprocess/id_alignment_report.json",
        help="Where to save the alignment report.",
    )
    args = parser.parse_args()

    train_ids = load_split_ids(args.split_json, subset=args.subset)
    skeleton_ids = collect_npy_stems(args.skeleton_dir, recursive=True)
    logits_ids = collect_npy_stems(args.logits_dir, recursive=False)

    report = summarize_alignment(train_ids, skeleton_ids, logits_ids)

    print(f"[INFO] subset={args.subset}")
    print(f"[INFO] train ids: {len(train_ids)}")
    print(f"[INFO] skeleton ids: {len(skeleton_ids)}")
    print(f"[INFO] logits ids: {len(logits_ids)}")
    print(f"[INFO] aligned ids (train ∩ skeleton ∩ logits): {len(report['aligned_all_ids'])}")
    print(f"[WARN] missing skeleton from train: {len(report['missing_skeleton_from_train'])}")
    print(f"[WARN] missing logits from train: {len(report['missing_logits_from_train'])}")

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] alignment report saved: {args.report_json}")


if __name__ == "__main__":
    main()

