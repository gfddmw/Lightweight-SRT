#!/usr/bin/env python3
"""
Verify CSL-Daily skeleton / teacher-feature / logits alignment.

This script is intentionally read-only. It checks:
  1) IDs exist in all required modalities.
  2) skeleton shape is [T, V, C].
  3) teacher features and logits are [T', D].
  4) teacher feature length == teacher logits length.
  5) downsampled skeleton length == teacher temporal length.

Example:
    python scripts/data_prep/verify_csl_daily_alignment.py \
        --csl_json data/CSL/csl-daily.json \
        --skeleton_dir processed/csl_daily/skeletons \
        --teacher_feature_dir processed/csl_daily/teacher_features \
        --teacher_logits_dir processed/csl_daily/teacher_logits \
        --downsample_rate 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SampleIssue:
    vid: str
    issue: str
    detail: str


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_ids_from_json(data: Any, subset: Optional[str]) -> List[str]:
    if isinstance(data, dict):
        ids: List[str] = []
        for key, meta in data.items():
            if subset and isinstance(meta, dict):
                item_subset = meta.get("subset", meta.get("split"))
                if item_subset != subset:
                    continue
            if isinstance(meta, dict):
                vid = meta.get("vid", meta.get("id", meta.get("name", key)))
            else:
                vid = key
            ids.append(str(vid))
        return sorted(set(ids))

    if isinstance(data, list):
        ids = []
        for idx, meta in enumerate(data):
            if subset and isinstance(meta, dict):
                item_subset = meta.get("subset", meta.get("split"))
                if item_subset != subset:
                    continue
            if isinstance(meta, dict):
                vid = meta.get("vid", meta.get("id", meta.get("name", idx)))
            else:
                vid = idx
            ids.append(str(vid))
        return sorted(set(ids))

    raise ValueError(f"Unsupported JSON root type: {type(data)!r}")


def collect_ids_from_dir(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted(p.stem for p in path.rglob("*.npy"))


def expected_teacher_len(skeleton_len: int, downsample_rate: int, rounding: str) -> int:
    if downsample_rate <= 0:
        raise ValueError("--downsample_rate must be positive")
    value = skeleton_len / downsample_rate
    if rounding == "ceil":
        return int(math.ceil(value))
    if rounding == "floor":
        return int(math.floor(value))
    if rounding == "round":
        return int(round(value))
    if rounding == "exact":
        return skeleton_len
    raise ValueError(f"Unsupported rounding rule: {rounding}")


def npy_shape(path: Path) -> Sequence[int]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return arr.shape


def verify_one(
    vid: str,
    skeleton_dir: Path,
    teacher_feature_dir: Path,
    teacher_logits_dir: Path,
    downsample_rate: int,
    rounding: str,
    expected_skeleton_v: int,
    expected_skeleton_c: int,
    expected_feature_dim: int,
    expected_logits_dim: int,
) -> List[SampleIssue]:
    issues: List[SampleIssue] = []
    skeleton_path = skeleton_dir / f"{vid}.npy"
    feature_path = teacher_feature_dir / f"{vid}.npy"
    logits_path = teacher_logits_dir / f"{vid}.npy"

    for label, path in (
        ("skeleton_missing", skeleton_path),
        ("teacher_feature_missing", feature_path),
        ("teacher_logits_missing", logits_path),
    ):
        if not path.exists():
            issues.append(SampleIssue(vid, label, str(path)))
    if issues:
        return issues

    try:
        skeleton_shape = tuple(npy_shape(skeleton_path))
    except Exception as exc:
        return [SampleIssue(vid, "skeleton_load_error", str(exc))]
    try:
        feature_shape = tuple(npy_shape(feature_path))
    except Exception as exc:
        return [SampleIssue(vid, "teacher_feature_load_error", str(exc))]
    try:
        logits_shape = tuple(npy_shape(logits_path))
    except Exception as exc:
        return [SampleIssue(vid, "teacher_logits_load_error", str(exc))]

    if len(skeleton_shape) != 3:
        issues.append(SampleIssue(vid, "skeleton_shape_error", f"{skeleton_shape}, expected [T,V,C]"))
        return issues
    if skeleton_shape[1] != expected_skeleton_v or skeleton_shape[2] != expected_skeleton_c:
        issues.append(
            SampleIssue(
                vid,
                "skeleton_shape_error",
                f"{skeleton_shape}, expected [T,{expected_skeleton_v},{expected_skeleton_c}]",
            )
        )

    if len(feature_shape) != 2 or feature_shape[1] != expected_feature_dim:
        issues.append(
            SampleIssue(vid, "teacher_feature_shape_error", f"{feature_shape}, expected [T',{expected_feature_dim}]")
        )
    if len(logits_shape) != 2 or logits_shape[1] != expected_logits_dim:
        issues.append(SampleIssue(vid, "teacher_logits_shape_error", f"{logits_shape}, expected [T',{expected_logits_dim}]"))

    if len(feature_shape) == 2 and len(logits_shape) == 2 and feature_shape[0] != logits_shape[0]:
        issues.append(
            SampleIssue(
                vid,
                "teacher_length_mismatch",
                f"feature T'={feature_shape[0]} != logits T'={logits_shape[0]}",
            )
        )

    if len(feature_shape) == 2:
        expected_len = expected_teacher_len(skeleton_shape[0], downsample_rate, rounding)
        if expected_len != feature_shape[0]:
            issues.append(
                SampleIssue(
                    vid,
                    "temporal_alignment_mismatch",
                    f"skeleton T={skeleton_shape[0]} -> expected teacher T'={expected_len} "
                    f"(rate={downsample_rate}, rule={rounding}), actual T'={feature_shape[0]}",
                )
            )

    return issues


def write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CSL-Daily ID and temporal alignment.")
    parser.add_argument("--csl_json", default="data/CSL/csl-daily.json", help="CSL-Daily annotation or split/index JSON.")
    parser.add_argument("--subset", default=None, help="Optional subset/split filter, e.g. train/dev/test.")
    parser.add_argument("--skeleton_dir", default="processed/csl_daily/skeletons")
    parser.add_argument("--teacher_feature_dir", default="processed/csl_daily/teacher_features")
    parser.add_argument("--teacher_logits_dir", default="processed/csl_daily/teacher_logits")
    parser.add_argument("--downsample_rate", type=int, default=4)
    parser.add_argument("--rounding", choices=["ceil", "floor", "round", "exact"], default="ceil")
    parser.add_argument("--expected_skeleton_v", type=int, default=42)
    parser.add_argument("--expected_skeleton_c", type=int, default=9)
    parser.add_argument("--expected_feature_dim", type=int, default=1024)
    parser.add_argument("--expected_logits_dim", type=int, default=2001)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all samples.")
    parser.add_argument("--report_json", default="processed/csl_daily/alignment_report.json")
    args = parser.parse_args()

    csl_json = resolve_path(args.csl_json)
    skeleton_dir = resolve_path(args.skeleton_dir)
    teacher_feature_dir = resolve_path(args.teacher_feature_dir)
    teacher_logits_dir = resolve_path(args.teacher_logits_dir)
    report_json = resolve_path(args.report_json)

    required_paths = [csl_json, skeleton_dir, teacher_feature_dir, teacher_logits_dir]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        print("[ERROR] Missing required CSL-Daily inputs:")
        for path in missing:
            print(f"  - {path}")
        print("Please download/extract the annotation JSON, skeletons, teacher_features and teacher_logits first.")
        return 2

    sample_ids = iter_ids_from_json(load_json(csl_json), args.subset)
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    skeleton_ids = set(collect_ids_from_dir(skeleton_dir))
    feature_ids = set(collect_ids_from_dir(teacher_feature_dir))
    logits_ids = set(collect_ids_from_dir(teacher_logits_dir))

    issues: List[SampleIssue] = []
    for idx, vid in enumerate(sample_ids, start=1):
        issues.extend(
            verify_one(
                vid=vid,
                skeleton_dir=skeleton_dir,
                teacher_feature_dir=teacher_feature_dir,
                teacher_logits_dir=teacher_logits_dir,
                downsample_rate=args.downsample_rate,
                rounding=args.rounding,
                expected_skeleton_v=args.expected_skeleton_v,
                expected_skeleton_c=args.expected_skeleton_c,
                expected_feature_dim=args.expected_feature_dim,
                expected_logits_dim=args.expected_logits_dim,
            )
        )
        if idx % 2000 == 0:
            print(f"[INFO] checked {idx}/{len(sample_ids)}")

    issue_payload = [asdict(issue) for issue in issues]
    issue_counts: Dict[str, int] = {}
    for issue in issues:
        issue_counts[issue.issue] = issue_counts.get(issue.issue, 0) + 1

    aligned_count = len(sample_ids) - len({issue.vid for issue in issues})
    report = {
        "config": vars(args),
        "paths": {
            "csl_json": str(csl_json),
            "skeleton_dir": str(skeleton_dir),
            "teacher_feature_dir": str(teacher_feature_dir),
            "teacher_logits_dir": str(teacher_logits_dir),
        },
        "counts": {
            "json_samples": len(sample_ids),
            "skeleton_files": len(skeleton_ids),
            "teacher_feature_files": len(feature_ids),
            "teacher_logits_files": len(logits_ids),
            "aligned_samples": aligned_count,
            "samples_with_issues": len({issue.vid for issue in issues}),
        },
        "issue_counts": issue_counts,
        "issues": issue_payload,
    }
    write_report(report_json, report)

    print("=" * 72)
    print("CSL-Daily alignment verification")
    print("=" * 72)
    print(f"JSON samples:          {len(sample_ids)}")
    print(f"Skeleton files:        {len(skeleton_ids)}")
    print(f"Teacher feature files: {len(feature_ids)}")
    print(f"Teacher logits files:  {len(logits_ids)}")
    print(f"Aligned samples:       {aligned_count}")
    print(f"Samples with issues:   {len({issue.vid for issue in issues})}")
    if issue_counts:
        print("\nIssue counts:")
        for key, value in sorted(issue_counts.items()):
            print(f"  - {key}: {value}")
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue.vid}: {issue.issue} | {issue.detail}")
    else:
        print("\nAll checked samples are aligned.")
    print(f"\nReport saved to: {report_json}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
