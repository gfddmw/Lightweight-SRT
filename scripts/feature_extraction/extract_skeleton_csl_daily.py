"""
extract_skeleton_csl_daily.py
==============================
CSL-Daily 手部骨骼点提取 (Holistic + 时序插值)
输出: [T, V, C] numpy  (T=帧数, V=42, C=9)

Usage:
    #单条测试
    python scripts/feature_extraction/extract_skeleton_csl_daily.py single --video_path data/CSL/video/S000000_P0000_T00.mp4 --output_path processed/csl_daily/skeletons/S000000_P0000_T00_test.npy --inspect
    #全局运行
    python scripts/feature_extraction/extract_skeleton_csl_daily.py multi --video_path data/CSL/video/ --output_path processed/csl_daily/skeletons/ --inspect
"""

# ══════════════════════════════════════════════════
#  第一步：在 OS 层面永久重定向 fd=2 → /dev/null
#  C++ 的 write(2, ...) 全部丢弃
#  必须在 import mediapipe 之前执行
# ══════════════════════════════════════════════════
import os as _os
import sys as _sys

# 保存原始 stderr fd 的副本 (指向控制台)
_orig_stderr_fd = _os.dup(_sys.stderr.fileno())

# fd=2 → /dev/null，C++ 警告彻底消失
_null_fd = _os.open(_os.devnull, _os.O_WRONLY)
_os.dup2(_null_fd, _sys.stderr.fileno())
_os.close(_null_fd)

# 用原始 fd 创建 Python 文件对象，供 logging 使用
_py_stderr = open(_orig_stderr_fd, 'w', encoding='utf-8', closefd=False)

# 注册异常钩子，确保 Python 报错能输出到原始 stderr，而不是被 /dev/null 吞掉
def _custom_excepthook(exctype, value, tb):
    import traceback
    traceback.print_exception(exctype, value, tb, file=_py_stderr)
_sys.excepthook = _custom_excepthook

_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os.environ['GLOG_minloglevel'] = '3'
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

# ══════ import 阶段：C++ 警告被 fd=2 重定向吞掉 ══════
import os, sys, json, argparse, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.holistic as mp_holistic
import mediapipe.python.solutions.hands as mp_hands
from tqdm import tqdm

# ══════ logging 使用原始 fd，正常输出到控制台 ══════
logging.basicConfig(stream=_py_stderr, level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# ─── 拓扑 ───
HAND_PARENT = np.array([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], dtype=np.int32)
DUAL_HAND_PARENT = np.array(
    list(HAND_PARENT) + [p+21 if p>=0 else -1 for p in HAND_PARENT], dtype=np.int32)


def _suppress_cpp_warnings():
    """在子进程中重定向 fd=2，消除 C++ 警告。"""
    import os as __os, sys as __sys
    try:
        __null = __os.open(__os.devnull, __os.O_WRONLY)
        __os.dup2(__null, __sys.stderr.fileno())
        __os.close(__null)
    except Exception:
        pass


class SkeletonExtractor:
    def __init__(self, method="holistic",
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 interpolate=True, interpolate_max_gap=999,
                 min_valid_frames_ratio=0.05,
                 center_on_wrist=False):
        self.method = method
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.interpolate = interpolate
        self.interpolate_max_gap = interpolate_max_gap
        self.min_valid_frames_ratio = min_valid_frames_ratio
        self.center_on_wrist = center_on_wrist

    def extract(self, video_path: str) -> Optional[np.ndarray]:
        if self.method == "holistic":
            joints = self._extract_holistic(video_path)
        else:
            joints = self._extract_full_frame(video_path)

        if joints is None or len(joints) == 0:
            return None

        j = joints[:, :, :3]
        any_hand = np.any(j != 0, axis=(1, 2))
        if any_hand.mean() < self.min_valid_frames_ratio:
            return None

        if self.interpolate:
            joints = self._interpolate_gaps(joints, self.interpolate_max_gap)

        if self.center_on_wrist:
            joints[:, :21, :]  -= joints[:, 0:1, :]
            joints[:, 21:42, :] -= joints[:, 21:22, :]

        bones  = self._compute_bones(joints)
        motion = self._compute_motion(joints)
        return np.concatenate([joints, bones, motion], axis=-1)

    def _extract_holistic(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        joints_list: List[np.ndarray] = []
        try:
            with mp_holistic.Holistic(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize to speed up without losing relative accuracy
                    h, w = frame.shape[:2]
                    max_dim = 480
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    r = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    left  = np.zeros((21, 3), dtype=np.float32)
                    right = np.zeros((21, 3), dtype=np.float32)
                    if r.left_hand_landmarks:
                        left = np.array([[l.x, l.y, l.z] for l in r.left_hand_landmarks.landmark], dtype=np.float32)
                    if r.right_hand_landmarks:
                        right = np.array([[l.x, l.y, l.z] for l in r.right_hand_landmarks.landmark], dtype=np.float32)
                    joints_list.append(np.concatenate([left, right], axis=0))
        finally:
            cap.release()
        if not joints_list:
            return None
        return np.stack(joints_list, axis=0).astype(np.float32)

    def _extract_full_frame(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        joints_list: List[np.ndarray] = []
        try:
            with mp_hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize to speed up without losing relative accuracy
                    h, w = frame.shape[:2]
                    max_dim = 480
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    r = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    left, right = self._parse_hands_results(r)
                    joints_list.append(np.concatenate([left, right], axis=0))
        finally:
            cap.release()
        if not joints_list:
            return None
        return np.stack(joints_list, axis=0).astype(np.float32)

    @staticmethod
    def _parse_hands_results(results):
        left, right = np.zeros((21,3),np.float32), np.zeros((21,3),np.float32)
        if not (results.multi_hand_landmarks and results.multi_handedness):
            return left, right
        landmarks_list = []
        labels = []
        for hlm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            arr = np.array([[l.x, l.y, l.z] for l in hlm.landmark], dtype=np.float32)
            label = hd.classification[0].label
            landmarks_list.append(arr)
            labels.append(label)
        
        # Conflict resolution if both hands are detected but classified with same label
        if len(landmarks_list) == 2 and labels[0] == labels[1]:
            x0 = landmarks_list[0][:, 0].mean()
            x1 = landmarks_list[1][:, 0].mean()
            # Smaller x coordinate is Right hand, larger x is Left hand (facing camera view)
            if x0 < x1:
                right, left = landmarks_list[0], landmarks_list[1]
            else:
                left, right = landmarks_list[0], landmarks_list[1]
        else:
            for arr, label in zip(landmarks_list, labels):
                if label == "Left":
                    left = arr
                else:
                    right = arr
        return left, right

    @staticmethod
    def _interpolate_gaps(joints: np.ndarray, max_gap: int = 999) -> np.ndarray:
        result = joints.copy()
        for hand_offset in [0, 21]:
            # Check if any coordinate of the hand's 21 landmarks is non-zero (more robust than checking wrist only)
            nz = np.where(np.any(result[:, hand_offset : hand_offset + 21, :] != 0, axis=(1, 2)))[0]
            if len(nz) == 0:
                continue
            first, last = nz[0], nz[-1]
            # 首尾扩展
            if first > 0:
                for j in range(21):
                    result[:first, hand_offset+j, :] = result[first, hand_offset+j, :]
            if last < result.shape[0] - 1:
                for j in range(21):
                    result[last+1:, hand_offset+j, :] = result[last, hand_offset+j, :]
            # 中间插值
            for i in range(len(nz) - 1):
                s, e = nz[i], nz[i+1]
                gap = e - s - 1
                if gap == 0 or gap > max_gap:
                    continue
                for j in range(21):
                    idx = hand_offset + j
                    for c in range(3):
                        sv, ev = result[s, idx, c], result[e, idx, c]
                        for t in range(s+1, e):
                            result[t, idx, c] = sv + (ev - sv) * (t - s) / (e - s)
        return result

    @staticmethod
    def _compute_bones(joints):
        bones = np.zeros_like(joints)
        for j in range(joints.shape[1]):
            p = DUAL_HAND_PARENT[j]
            if p >= 0:
                bones[:, j, :] = joints[:, j, :] - joints[:, p, :]
        return bones

    @staticmethod
    def _compute_motion(joints):
        motion = np.zeros_like(joints)
        if joints.shape[0] > 1:
            motion[1:] = joints[1:] - joints[:-1]
        return motion


def _proc_wrapper(args):
    return BatchSkeletonProcessor._proc(*args)


class BatchSkeletonProcessor:
    def __init__(self, video_mapping, output_dir, checkpoint_path, workers=4, **kw):
        self.video_mapping = video_mapping
        self.output_dir = Path(output_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.workers = workers
        self.ext_params = kw
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self._load()

    def _load(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                ckpt = json.load(f)
            logger.info(f"断点: 完成 {len(ckpt.get('completed',{}))} / 失败 {len(ckpt.get('failed',{}))}")
            return ckpt
        return {"completed": {}, "failed": {}}

    def _save(self):
        with open(self.checkpoint_path, "w") as f:
            json.dump(self.checkpoint, f, ensure_ascii=False, indent=2)

    def _done(self, vid):
        if vid in self.checkpoint["completed"]:
            if (self.output_dir / f"{vid}.npy").exists():
                return True
            del self.checkpoint["completed"][vid]
        return False

    @staticmethod
    def _proc(vid, path, odir, params):
        _suppress_cpp_warnings()
        try:
            ext = SkeletonExtractor(**params)
            data = ext.extract(path)
            if data is None:
                return vid, False, None, "None"
            np.save(str(Path(odir) / f"{vid}.npy"), data)
            return vid, True, data.shape[0], "ok"
        except Exception as e:
            return vid, False, None, str(e)

    def run(self):
        pending = {v: i for v, i in self.video_mapping.items() if not self._done(v)}
        total = len(self.video_mapping)
        done = total - len(pending)
        logger.info(f"总计 {total}，已完成 {done}，待处理 {len(pending)}")
        if not pending:
            return
        ok = fail = 0
        import multiprocessing
        tasks = [(v, i["path"], str(self.output_dir), self.ext_params) for v, i in pending.items()]
        with multiprocessing.Pool(processes=self.workers, maxtasksperchild=100) as pool:
            with tqdm(total=len(pending), desc="Extracting", file=_py_stderr) as pbar:
                for vid, s, n, m in pool.imap_unordered(_proc_wrapper, tasks):
                    if s:
                        self.checkpoint["completed"][vid] = {"frames": n}
                        ok += 1
                    else:
                        self.checkpoint["failed"][vid] = m
                        fail += 1
                        logger.warning(f"失败 {vid}: {m}")
                    if (ok + fail) % 50 == 0:
                        self._save()
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, fail=fail)
        self._save()
        logger.info(f"完毕: ok={ok} fail={fail} 累计={len(self.checkpoint['completed'])}/{total}")
        if self.checkpoint["failed"]:
            fl = self.checkpoint_path.with_name("extraction_failures.txt")
            with open(fl, "w") as f:
                for v, m in self.checkpoint["failed"].items():
                    f.write(f"{v}\t{m}\n")


def load_video_mapping(p):
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    logger.info(f"映射: {len(m)} 条")
    return m


def inspect_npy(p):
    d = np.load(p)
    T, V, C = d.shape
    j = d[:, :, :3]
    l = np.any(j[:, :21, :] != 0, axis=(1, 2))
    r = np.any(j[:, 21:, :] != 0, axis=(1, 2))
    _py_stderr.write(f"{p}\n")
    _py_stderr.write(f"  shape={d.shape}  dtype={d.dtype}\n")
    _py_stderr.write(f"  左手帧: {l.sum()}/{T} ({l.mean():.0%})  "
                     f"右手帧: {r.sum()}/{T} ({r.mean():.0%})  "
                     f"至少一手: {(l|r).sum()}/{T} ({(l|r).mean():.0%})\n")
    _py_stderr.flush()


def main():
    pa = argparse.ArgumentParser()
    sub = pa.add_subparsers(dest="mode")

    p1 = sub.add_parser("single")
    p1.add_argument("--video_path", required=True)
    p1.add_argument("--output_path", required=True)
    p1.add_argument("--method", default="holistic", choices=["holistic", "full_frame"])
    p1.add_argument("--no_interpolate", action="store_true")
    p1.add_argument("--inspect", action="store_true")

    p2 = sub.add_parser("batch")
    p2.add_argument("--video_mapping", required=True)
    p2.add_argument("--output_dir", default="processed/csl_daily/skeletons")
    p2.add_argument("--checkpoint_dir", default="processed/csl_daily/checkpoints")
    p2.add_argument("--workers", type=int, default=4)
    p2.add_argument("--method", default="holistic", choices=["holistic", "full_frame"])
    p2.add_argument("--no_interpolate", action="store_true")

    p3 = sub.add_parser("inspect")
    p3.add_argument("npy_path")

    args = pa.parse_args()

    if args.mode == "single":
        ext = SkeletonExtractor(method=args.method, interpolate=not args.no_interpolate)
        data = ext.extract(args.video_path)
        if data is None:
            logger.error("提取失败"); sys.exit(1)
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), data)
        logger.info(f"已保存: {out}  shape={data.shape}")
        if args.inspect:
            inspect_npy(str(out))

    elif args.mode == "batch":
        m = load_video_mapping(args.video_mapping)
        ckpt = Path(args.checkpoint_dir) / "skeleton_checkpoint.json"
        BatchSkeletonProcessor(
            m, args.output_dir, str(ckpt), workers=args.workers,
            method=args.method, interpolate=not args.no_interpolate
        ).run()

    elif args.mode == "inspect":
        inspect_npy(args.npy_path)

    else:
        pa.print_help()


if __name__ == "__main__":
    main()