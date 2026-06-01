#!/usr/bin/env python3
"""
Render a random CSL-Daily sample as skeleton frames plus gloss/text annotation.

The tool reads real files only. If CSL-Daily data is missing, it reports the
missing path and exits instead of creating placeholder data.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.datasets.cslt_dataset import CSLTDataset


HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
DUAL_HAND_EDGES = HAND_EDGES + [(a + 21, b + 21) for a, b in HAND_EDGES]


def choose_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def draw_skeleton(ax, points: np.ndarray, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")

    xy = points[:, :2]
    valid = np.any(points[:, :3] != 0, axis=1)
    for a, b in DUAL_HAND_EDGES:
        if valid[a] and valid[b]:
            color = "#2a9d8f" if a < 21 else "#e76f51"
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=color, linewidth=1.6)
    ax.scatter(xy[valid, 0], xy[valid, 1], s=10, c="#263238")


def render_contact_sheet(sample, output_path: Path, num_frames: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it in this environment, "
            "for example: pip install matplotlib"
        ) from exc

    skeleton = np.load(sample["paths"]["skeleton"], mmap_mode="r", allow_pickle=False)
    joints = np.asarray(skeleton[..., :3], dtype=np.float32)
    frame_ids = choose_frame_indices(joints.shape[0], num_frames)
    if not frame_ids:
        raise ValueError(f"{sample['vid']} has no skeleton frames")

    fig = plt.figure(figsize=(3.2 * len(frame_ids), 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, len(frame_ids), height_ratios=[4, 1])
    for col, frame_idx in enumerate(frame_ids):
        ax = fig.add_subplot(grid[0, col])
        draw_skeleton(ax, joints[frame_idx], f"frame {frame_idx}/{joints.shape[0] - 1}")

    text_ax = fig.add_subplot(grid[1, :])
    text_ax.axis("off")
    text_ax.text(
        0.01,
        0.68,
        f"vid: {sample['vid']}",
        fontsize=10,
        ha="left",
        va="center",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.01,
        0.38,
        f"Gloss: {sample['gloss'] or '[missing gloss annotation]'}",
        fontsize=11,
        ha="left",
        va="center",
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.01,
        0.08,
        f"中文: {sample['text'] or '[missing Chinese translation]'}",
        fontsize=11,
        ha="left",
        va="center",
        transform=text_ax.transAxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize one CSL-Daily multimodal sample.")
    parser.add_argument("--index_json", default="scripts/data_prep/generate_csl_daily_splits.json")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--skeleton_dir", default="processed/csl_daily/skeletons")
    parser.add_argument("--teacher_feature_dir", default="processed/csl_daily/teacher_features")
    parser.add_argument("--teacher_logits_dir", default="processed/csl_daily/teacher_logits")
    parser.add_argument("--sample_id", default=None, help="Specific vid. If omitted, sample randomly.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--output", default="processed/csl_daily/preview/csl_daily_sample_preview.png")
    args = parser.parse_args()

    try:
        dataset = CSLTDataset(
            index_json=args.index_json,
            subset=args.subset,
            skeleton_dir=args.skeleton_dir,
            teacher_feature_dir=args.teacher_feature_dir,
            teacher_logits_dir=args.teacher_logits_dir,
            require_files=True,
        )
    except Exception as exc:
        print(f"[ERROR] Cannot load CSL-Daily dataset: {exc}")
        return 2
    if len(dataset) == 0:
        print("[ERROR] No usable samples found.")
        return 2

    if args.sample_id:
        positions = [idx for idx, item in enumerate(dataset.samples) if item["vid"] == args.sample_id]
        if not positions:
            print(f"[ERROR] sample_id not found or missing required files: {args.sample_id}")
            return 2
        index = positions[0]
    else:
        rng = random.Random(args.seed)
        index = rng.randrange(len(dataset))

    sample = dataset[index]
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    try:
        render_contact_sheet(sample, output, max(1, args.num_frames))
    except Exception as exc:
        print(f"[ERROR] Cannot render preview: {exc}")
        return 2

    print(f"Rendered sample preview: {output}")
    print(f"vid: {sample['vid']}")
    print(f"Gloss: {sample['gloss']}")
    print(f"中文: {sample['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
