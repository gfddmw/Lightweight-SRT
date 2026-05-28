"""
prepare_csl_daily_videos.py
=============================
Task A · 任务点一：路径映射与视频完整性核对

目录结构:
    data/CSL/
    ├── csl-daily.json          ← 标注文件
    ├── csl-daily_list.json     ← 视频列表文件
    ├── data_load.py
    └── video/                  ← 视频平铺，无子目录
        ├── S000006_P0000_T00.mp4
        ├── S000002_P0000_T00.mp4
        └── ...

Usage:
python scripts/data_prep/prepare_csl_daily_videos.py --csl_root data/CSL --output configs/csl_daily/video_mapping.json --workers 8
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_v'] = '0'

import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv")


# ══════════════════════════════════════════
#  1. 解析 JSON 标注文件
# ══════════════════════════════════════════

def parse_csl_daily_json(json_path: str) -> Dict[str, Dict]:
    """
    解析 csl-daily.json，返回 {vid: {gloss, signer, ...}}。

    自适应两种常见格式:
      字典格式: {"0": {"gloss": "...", ...}, "1": {...}, ...}
      列表格式: [{"gloss": "...", ...}, ...]

    vid 生成优先级: 标注中的 id/vid/name 字段 > 字典 key > 列表索引
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[str, Dict] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            vid = (
                str(value.get("id", ""))
                or str(value.get("vid", ""))
                or str(value.get("name", ""))
                or str(key)
            )
            mapping[vid] = {
                "vid": vid,
                "gloss": value.get("gloss", value.get("text", "")),
                "signer": value.get("signer", value.get("speaker", "")),
                "_raw": value,
            }
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            vid = (
                str(item.get("id", ""))
                or str(item.get("vid", ""))
                or str(item.get("name", ""))
                or str(idx)
            )
            mapping[vid] = {
                "vid": vid,
                "gloss": item.get("gloss", item.get("text", "")),
                "signer": item.get("signer", item.get("speaker", "")),
                "_raw": item,
            }
    else:
        raise ValueError(f"不支持的 JSON 格式: {type(raw)}")

    logger.info(f"从 {json_path} 解析到 {len(mapping)} 条标注")
    return mapping


def parse_csl_daily_list_json(json_path: str) -> Dict[str, Dict]:
    """解析 csl-daily_list.json，补充路径线索。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[str, Dict] = {}
    items = raw.values() if isinstance(raw, dict) else raw

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        vid = (
            str(item.get("id", ""))
            or str(item.get("vid", ""))
            or str(item.get("name", ""))
            or str(idx)
        )
        mapping[vid] = item

    logger.info(f"从 {json_path} 解析到 {len(mapping)} 条记录")
    return mapping


# ══════════════════════════════════════════
#  2. 扫描 video/ 目录（平铺结构）
# ══════════════════════════════════════════

def scan_video_dir(video_dir: Path) -> Dict[str, Path]:
    """
    扫描 video/ 目录下的视频文件（无子目录，直接平铺）。
    返回 {stem: absolute_path}，stem 即为 vid。
    """
    video_dir = video_dir.resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    index: Dict[str, Path] = {}
    for ext in VIDEO_EXTENSIONS:
        for f in video_dir.glob(f"*{ext}"):
            stem = f.stem
            if stem in index:
                logger.warning(f"重名文件，后者覆盖: {stem}")
            index[stem] = f

    logger.info(f"扫描 video/: 找到 {len(index)} 个视频文件")
    return index


# ══════════════════════════════════════════
#  3. 匹配：JSON vid ↔ 视频文件
# ══════════════════════════════════════════

def match_videos(
    annotations: Dict[str, Dict],
    video_index: Dict[str, Path],
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    将标注中的 vid 与视频文件 stem 进行匹配。

    匹配策略（按优先级）:
      1) vid 精确匹配 stem
      2) vid 补零匹配 stem（如 "1" 匹配 "000001"）
      3) 标注中 name 字段匹配
      4) stem 包含 vid 或 vid 包含 stem

    Returns:
        matched:   {vid: {path, gloss, signer}}
        unmatched: [vid, ...]
    """
    matched: Dict[str, Dict] = {}
    unmatched: List[str] = []

    # 构建 stem 小写索引
    stem_lower = {k.lower(): k for k in video_index}

    for vid, meta in annotations.items():
        path_found = None

        # ── 策略1: 精确匹配 ──
        if vid in video_index:
            path_found = video_index[vid]
        elif vid.lower() in stem_lower:
            path_found = video_index[stem_lower[vid.lower()]]

        # ── 策略2: 补零匹配 ──
        #       标注里可能是 "1" 但文件名是 "000001"
        if path_found is None:
            for pad_len in (6, 5, 4, 3, 2):
                padded = vid.zfill(pad_len)
                if padded in video_index:
                    path_found = video_index[padded]
                    break
                if padded.lower() in stem_lower:
                    path_found = video_index[stem_lower[padded.lower()]]
                    break

        # ── 策略3: 标注中 name 字段 ──
        if path_found is None:
            for field in ("name", "url", "file", "filename", "video_path"):
                raw_val = meta.get("_raw", {}).get(field, "")
                if raw_val:
                    name = str(raw_val).rsplit(".", 1)[0]
                    if name in video_index:
                        path_found = video_index[name]
                        break
                    if name.lower() in stem_lower:
                        path_found = video_index[stem_lower[name.lower()]]
                        break

        # ── 策略4: 子串包含 ──
        if path_found is None:
            for stem in video_index:
                if vid in stem or stem in vid:
                    path_found = video_index[stem]
                    break

        if path_found is not None:
            matched[vid] = {
                "vid": vid,
                "path": str(path_found),
                "gloss": meta.get("gloss", ""),
                "signer": meta.get("signer", ""),
            }
        else:
            unmatched.append(vid)

    return matched, unmatched


# ══════════════════════════════════════════
#  4. 视频完整性校验
# ══════════════════════════════════════════

def check_video_integrity(video_path: str) -> Optional[Dict]:
    """校验单个视频文件完整性。返回元信息字典或 None。"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
            return None

        # 读取首帧验证解码器
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        return {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
        }
    except Exception:
        return None


def _worker_check(args: Tuple[str, str]) -> Tuple[str, Optional[Dict]]:
    vid, path = args
    return vid, check_video_integrity(path)


# ══════════════════════════════════════════
#  5. 主流程
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSL-Daily: 路径映射与视频完整性核对"
    )
    parser.add_argument("--csl_root", type=str, default="data/CSL",
                        help="CSL-Daily 数据根目录")
    parser.add_argument("--output", type=str,
                        default="configs/csl_daily/video_mapping.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    csl_root   = Path(args.csl_root)
    video_dir  = csl_root / "video"
    json_main  = csl_root / "csl-daily.json"
    json_list  = csl_root / "csl-daily_list.json"

    # ── Step 1: 解析 JSON ──
    logger.info("=" * 60)
    logger.info("Step 1: 解析 JSON 标注文件")

    annotations: Dict[str, Dict] = {}
    if json_main.exists():
        annotations = parse_csl_daily_json(str(json_main))
    else:
        logger.warning(f"未找到: {json_main}")

    if json_list.exists():
        list_data = parse_csl_daily_list_json(str(json_list))
        for vid, meta in list_data.items():
            if vid not in annotations:
                annotations[vid] = {"vid": vid, "gloss": "", "signer": "", "_raw": meta}

    if not annotations:
        logger.error("未解析到任何标注，退出"); sys.exit(1)
    logger.info(f"标注总数: {len(annotations)}")

    # ── Step 2: 扫描 video/ ──
    logger.info("=" * 60)
    logger.info("Step 2: 扫描 video/ 目录（平铺结构）")
    video_index = scan_video_dir(video_dir)

    # 打印前 20 个 stem 帮助确认格式
    logger.info("── 视频 stem 示例（前20）──")
    for stem in sorted(video_index.keys())[:20]:
        logger.info(f"  {stem}")

    # ── Step 3: 匹配 ──
    logger.info("=" * 60)
    logger.info("Step 3: 匹配标注 ↔ 视频文件")

    matched, unmatched = match_videos(annotations, video_index)
    logger.info(f"匹配成功: {len(matched)} / {len(annotations)}")
    logger.info(f"未匹配: {len(unmatched)}")

    if unmatched:
        unmatch_path = Path(args.output).with_name("unmatched_videos.txt")
        with open(unmatch_path, "w", encoding="utf-8") as f:
            for vid in sorted(unmatched):
                meta = annotations.get(vid, {})
                f.write(f"{vid}\t{meta.get('gloss', '')}\t{meta.get('signer', '')}\n")
        logger.info(f"未匹配列表 → {unmatch_path}")

        logger.info("── 未匹配示例（前10）──")
        for vid in sorted(unmatched)[:10]:
            meta = annotations.get(vid, {})
            logger.info(f"  {vid}  gloss={meta.get('gloss', '')}  signer={meta.get('signer', '')}")

    if not matched:
        logger.error("无匹配视频，请检查 vid 格式与文件名是否对应")
        sys.exit(1)

    # ── Step 4: 完整性校验 ──
    logger.info("=" * 60)
    logger.info(f"Step 4: 完整性校验 (workers={args.workers})")

    valid_mapping: Dict[str, Dict] = {}
    invalid_list: List[str] = []

    tasks = [(vid, info["path"]) for vid, info in matched.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker_check, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            vid, result = fut.result()
            if result is not None:
                info = matched[vid]
                info.update(result)
                info["valid"] = True
                valid_mapping[vid] = info
            else:
                invalid_list.append(vid)

    logger.info(f"有效视频: {len(valid_mapping)} / {len(matched)}")
    logger.info(f"损坏视频: {len(invalid_list)}")

    if invalid_list:
        bad_path = Path(args.output).with_name("invalid_videos.txt")
        with open(bad_path, "w", encoding="utf-8") as f:
            for vid in sorted(invalid_list):
                f.write(f"{vid}\t{matched[vid]['path']}\n")
        logger.info(f"损坏列表 → {bad_path}")

    # ── Step 5: 保存 ──
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for vid in valid_mapping:
        valid_mapping[vid].pop("_raw", None)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(valid_mapping, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"映射已保存 → {out}")
    logger.info(f"统计: 总标注 {len(annotations)} | 匹配 {len(matched)} | "
                f"有效 {len(valid_mapping)} | 损坏 {len(invalid_list)} | "
                f"未匹配 {len(unmatched)}")


if __name__ == "__main__":
    main()