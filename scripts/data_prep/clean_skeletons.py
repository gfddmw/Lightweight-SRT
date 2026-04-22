# 用于过滤抖动严重、有效帧过少或置信度低的样本，生成干净的索引文件供 Dataset 加载。
import numpy as np
import json
from pathlib import Path
import argparse

def clean_skeletons_mediapipe(npy_dir, output_path, min_frames=10):
    npy_path = Path(npy_dir).resolve()
    if not npy_path.exists():
        print(f"❌ 目录不存在: {npy_path}")
        return

    npy_files = list(npy_path.glob("*.npy"))
    print(f"🔍 找到 {len(npy_files)} 个 .npy 文件")
    if len(npy_files) == 0:
        print("⚠️ 未找到文件！请检查路径。")
        return

    clean_indices = []
    stats = {"short_frames": 0, "empty_data": 0, "ok": 0}

    for fp in npy_files:
        try:
            data = np.load(fp, allow_pickle=True)
            
            # 验证形状是否符合 MediaPipe 单手格式 (T, 21, 3)
            if data.ndim != 3 or data.shape[1] != 21 or data.shape[2] != 3:
                stats["empty_data"] += 1
                continue

            # ✅ 核心修复：MediaPipe 第3列是深度Z，不是置信度！
            # 有效性判断：只要 x,y 坐标不全为 0，即视为有效检测帧
            valid_frame_mask = np.any(data[:, :, :2] != 0, axis=1)
            valid_frames_count = np.sum(valid_frame_mask)
            
            if valid_frames_count < min_frames:
                stats["short_frames"] += 1
                continue

            clean_indices.append(fp.stem)
            stats["ok"] += 1

        except Exception as e:
            print(f"⚠️ 跳过 {fp.name}: {e}")
            stats["empty_data"] += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(clean_indices, f)
        
    print(f"✅ 清洗完成 | 保留: {stats['ok']} | 丢弃(帧不足): {stats['short_frames']} | 异常/空: {stats['empty_data']}")


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent 
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", type=str, default=PROJECT_ROOT / "processed" / "skeletons", help="Directory containing .npy skeleton files")
    parser.add_argument("--output", type=str, default=PROJECT_ROOT / "processed" / "clean_indices.json")
    parser.add_argument("--min_frames", type=int, default=10)
    
    args = parser.parse_args()
    clean_skeletons_mediapipe(args.npy_dir, args.output, args.min_frames)