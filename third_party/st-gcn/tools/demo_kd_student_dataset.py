#!/usr/bin/env python
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feeder.student_dataset import StudentDataset
from net.st_gcn import Model


def prepare_fake_data(root: Path):
    split_path = root / 'split.json'
    skeleton_dir = root / 'skeletons'
    logits_dir = root / 'teacher_logits'

    skeleton_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)

    split = {}
    for i in range(12):
        vid = f'vid_{i:05d}'
        subset = 'train' if i < 10 else 'val'
        label = i % 2000

        # (T, V, C) with V=18 so it can be reshaped back to ST-GCN input.
        np.save(skeleton_dir / f'{vid}.npy', np.random.randn(64, 18, 3).astype(np.float32))
        np.save(logits_dir / f'{vid}.npy', np.random.randn(2000).astype(np.float32))

        split[vid] = {'subset': subset, 'action': [label]}

    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    return split_path, skeleton_dir, logits_dir


def run_demo(root: Path):
    split_path, skeleton_dir, logits_dir = prepare_fake_data(root)

    ds = StudentDataset(
        split_json=str(split_path),
        skeleton_dir=str(skeleton_dir),
        logits_dir=str(logits_dir),
        subset='train',
        seq_len=64,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    skeleton, teacher_logits, label, vid = next(iter(loader))

    # Convert (N, T, F) -> (N, C, T, V, M)
    n, t, f = skeleton.shape
    assert f % 3 == 0
    v = f // 3
    x = skeleton.view(n, t, v, 3).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

    model = Model(
        in_channels=3,
        num_class=2000,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={'layout': 'openpose', 'strategy': 'spatial'},
    )
    model.eval()

    with torch.no_grad():
        student_logits = model(x.float())

    temperature = 4.0
    alpha = 0.5
    ce = torch.nn.functional.cross_entropy(student_logits, label.long())
    kd = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=1),
        torch.nn.functional.softmax(teacher_logits.float() / temperature, dim=1),
        reduction='batchmean',
    ) * (temperature ** 2)
    total = (1.0 - alpha) * ce + alpha * kd

    print('KD demo passed')
    print('batch vids:', list(vid))
    print('skeleton (N,T,F):', tuple(skeleton.shape))
    print('stgcn input (N,C,T,V,M):', tuple(x.shape))
    print('teacher logits:', tuple(teacher_logits.shape))
    print('student logits:', tuple(student_logits.shape))
    print('ce={:.6f}, kd={:.6f}, total={:.6f}'.format(ce.item(), kd.item(), total.item()))


if __name__ == '__main__':
    run_demo(Path('tmp/kd_demo'))
