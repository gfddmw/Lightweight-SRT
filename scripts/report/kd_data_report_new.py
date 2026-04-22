import os
import sys
import json
import argparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)


def new_collect_npy_stems(root_dir):
    stems = set()
    if not os.path.exists(root_dir):
        return stems

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".npy"):
                stems.add(os.path.splitext(filename)[0])
    return stems


def new_load_split_records(split_json):
    with open(split_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("split json must be dict: video_id -> metadata")
    return data


def new_summarize_subset(records, subset_name, skeleton_ids, logits_ids, teacher_feature_ids):
    subset_ids = []
    for vid, meta in records.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("subset") != subset_name:
            continue
        if "action" not in meta or len(meta["action"]) == 0:
            continue
        subset_ids.append(str(vid))

    subset_id_set = set(subset_ids)
    has_skeleton = subset_id_set & skeleton_ids
    has_logits = subset_id_set & logits_ids
    has_teacher_features = subset_id_set & teacher_feature_ids
    final_usable = subset_id_set & skeleton_ids & logits_ids & teacher_feature_ids

    return {
        "subset": subset_name,
        "raw_count": len(subset_id_set),
        "has_skeleton": len(has_skeleton),
        "has_logits": len(has_logits),
        "has_teacher_features": len(has_teacher_features),
        "final_usable_count": len(final_usable),
        "usable_ratio": 0.0 if len(subset_id_set) == 0 else len(final_usable) / len(subset_id_set),
        "missing_skeleton": len(subset_id_set - skeleton_ids),
        "missing_logits": len(subset_id_set - logits_ids),
        "missing_teacher_features": len(subset_id_set - teacher_feature_ids),
    }


def new_print_summary(summary):
    print(f"Subset: {summary['subset']}")
    print(f"  raw_count: {summary['raw_count']}")
    print(f"  has_skeleton: {summary['has_skeleton']}")
    print(f"  has_logits: {summary['has_logits']}")
    print(f"  has_teacher_features: {summary['has_teacher_features']}")
    print(f"  final_usable_count: {summary['final_usable_count']}")
    print(f"  usable_ratio: {summary['usable_ratio']:.4f}")
    print(f"  missing_skeleton: {summary['missing_skeleton']}")
    print(f"  missing_logits: {summary['missing_logits']}")
    print(f"  missing_teacher_features: {summary['missing_teacher_features']}")


def new_run_report(split_json, skeleton_dir, logits_dir, teacher_feature_dir):
    records = new_load_split_records(split_json)
    skeleton_ids = new_collect_npy_stems(skeleton_dir)
    logits_ids = new_collect_npy_stems(logits_dir)
    teacher_feature_ids = new_collect_npy_stems(teacher_feature_dir)

    for subset_name in ["train", "test", "val"]:
        summary = new_summarize_subset(
            records=records,
            subset_name=subset_name,
            skeleton_ids=skeleton_ids,
            logits_ids=logits_ids,
            teacher_feature_ids=teacher_feature_ids,
        )
        new_print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KD data coverage summary for reports")
    parser.add_argument("--split_json", default="./data/WLASL2000/preprocess/nslt_2000.json")
    parser.add_argument("--skeleton_dir", default="./processed/skeletons")
    parser.add_argument("--logits_dir", default="./processed/logits")
    parser.add_argument("--teacher_feature_dir", default="./processed/teacher_features")
    args = parser.parse_args()

    new_run_report(
        split_json=args.split_json,
        skeleton_dir=args.skeleton_dir,
        logits_dir=args.logits_dir,
        teacher_feature_dir=args.teacher_feature_dir,
    )
