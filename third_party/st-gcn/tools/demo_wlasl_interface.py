#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F



def build_fake_keypoints(root: Path, train_num: int, val_num: int, t: int, v: int, m: int):
    root.mkdir(parents=True, exist_ok=True)
    ann = []

    total = train_num + val_num
    for i in range(total):
        name = f"sample_{i:05d}"
        subset = "train" if i < train_num else "val"
        label = i % 2000

        # (T, V, C, M), C=3 -> x,y,score
        arr = np.random.rand(t, v, 3, m).astype("float32")
        arr[..., 2, :] = np.clip(arr[..., 2, :], 0.1, 1.0)

        rel_path = f"keypoints/{name}.npy"
        out_path = root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)

        ann.append(
            {
                "sample_name": name,
                "subset": subset,
                "label": int(label),
                "keypoint_path": rel_path,
            }
        )

    ann_path = root / "annotation.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)

    return ann_path



def run_gendata(repo_root: Path, ann_path: Path, data_out: Path, max_frame: int, num_person_out: int):
    data_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "wlasl_gendata.py"),
        "--annotation",
        str(ann_path),
        "--keypoint_root",
        str(ann_path.parent),
        "--out_folder",
        str(data_out),
        "--max_frame",
        str(max_frame),
        "--num_person_out",
        str(num_person_out),
        "--expected_num_class",
        "0",
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))



def smoke_forward(repo_root: Path, data_out: Path):
    sys.path.insert(0, str(repo_root))

    from feeder.feeder import Feeder
    from net.st_gcn import Model

    train_data = data_out / "train_data.npy"
    train_label = data_out / "train_label.pkl"

    ds = Feeder(data_path=str(train_data), label_path=str(train_label), mmap=False)
    if len(ds) < 2:
        raise RuntimeError("Need at least 2 train samples for smoke test.")

    x0, y0 = ds[0]
    x1, y1 = ds[1]
    x = torch.tensor(np.stack([x0, x1]), dtype=torch.float32)
    y = torch.tensor([y0, y1], dtype=torch.long)

    model = Model(
        in_channels=3,
        num_class=2000,
        edge_importance_weighting=True,
        graph_args={"layout": "openpose", "strategy": "spatial"},
    )
    model.eval()

    with torch.no_grad():
        out = model(x)
        loss = F.cross_entropy(out, y)

    if out.shape != (2, 2000):
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")

    print("Interface smoke test passed.")
    print(f"Input batch shape: {tuple(x.shape)}")
    print(f"Output logits shape: {tuple(out.shape)}")
    print(f"CrossEntropy loss: {loss.item():.6f}")



def main():
    parser = argparse.ArgumentParser(description="Quick smoke test for WLASL->ST-GCN interface")
    parser.add_argument("--repo_root", default=".", help="Path to st-gcn repo root")
    parser.add_argument("--work_dir", default="./tmp/wlasl_demo", help="Temporary demo workspace")
    parser.add_argument("--train_num", type=int, default=8)
    parser.add_argument("--val_num", type=int, default=2)
    parser.add_argument("--t", type=int, default=48, help="Temporal length of fake keypoints")
    parser.add_argument("--v", type=int, default=18, help="Number of joints, must match graph layout")
    parser.add_argument("--m", type=int, default=1, help="Number of persons")
    parser.add_argument("--max_frame", type=int, default=60)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    raw_root = work_dir / "raw"
    data_out = work_dir / "stgcn_data"

    print(f"Repo root: {repo_root}")
    print(f"Demo work dir: {work_dir}")

    ann_path = build_fake_keypoints(
        root=raw_root,
        train_num=args.train_num,
        val_num=args.val_num,
        t=args.t,
        v=args.v,
        m=args.m,
    )

    run_gendata(
        repo_root=repo_root,
        ann_path=ann_path,
        data_out=data_out,
        max_frame=args.max_frame,
        num_person_out=args.m,
    )

    smoke_forward(repo_root=repo_root, data_out=data_out)


if __name__ == "__main__":
    main()
