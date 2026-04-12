import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import time
import sys

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_REL_PATH = "hand_landmarker.task"
VIDEO_REL_PATH = "../../data/WLASL2000"
OUTPUT_REL_PATH = "../../processed/skeletons"

MODEL_PATH = os.path.join(BASE_DIR, MODEL_REL_PATH)
VIDEO_ROOT = os.path.join(BASE_DIR, VIDEO_REL_PATH)
OUTPUT_ROOT = os.path.join(BASE_DIR, OUTPUT_REL_PATH)

# 如果你的内存非常大 (32G+)，可以尝试改为 4
NUM_WORKERS = 32
# ===========================================

def linear_interpolation(data):
    frames, points, coords = data.shape
    for p in range(points):
        for c in range(coords):
            series = data[:, p, c]
            valid_idx = np.where(series != 0)[0]
            if len(valid_idx) < 2: continue
            all_idx = np.arange(frames)
            data[:, p, c] = np.interp(all_idx, valid_idx, series[valid_idx])
    return data

def process_single_video(task):
    """
    工作函数：严格隔离环境，处理完立即释放资源
    """
    video_path, output_path, video_id = task

    # 定义局部变量，避免污染全局
    landmarker = None
    cap = None

    try:
        # 1. 初始化模型 (每个进程独立加载)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 使用 context manager 确保退出时释放模型资源
        with vision.HandLandmarker.create_from_options(options) as landmarker:

            # 2. 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return video_id, "FAIL_OPEN", 0, 0

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 25.0

            all_frames_data = []
            missing_count = 0
            total_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret: break

                total_frames += 1

                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((total_frames - 1) * (1000 / fps))
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame_data = np.zeros((21, 3), dtype=np.float32)

                if results.hand_landmarks:
                    lm_list = results.hand_landmarks[0]
                    for i, lm in enumerate(lm_list):
                        frame_data[i] = [lm.x, lm.y, lm.z]
                else:
                    missing_count += 1

                all_frames_data.append(frame_data)

                # 每处理 100 帧手动释放一次临时内存（防止长视频爆内存）
                if total_frames % 100 == 0:
                    gc.collect()

            cap.release() # 释放视频句柄
            cap = None    # 断开引用

            if total_frames == 0:
                return video_id, "EMPTY", 0, 0

            # 数据处理
            # 移除针对 bad_sample 的剔除逻辑，保留所有视频的数据
            data_array = np.array(all_frames_data, dtype=np.float32)

            # 清空原始列表释放内存
            del all_frames_data
            gc.collect()

            # 只要有数据就尝试进行插值处理
            if missing_count > 0 and missing_count < total_frames:
                data_array = linear_interpolation(data_array)

            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, data_array)

            # 状态统一标记为 OK，除非完全没有帧
            status = "OK"
            return video_id, status, total_frames, missing_count

    except Exception as e:
        # 捕获所有异常，防止子进程静默死亡
        error_msg = f"CRASH:{str(e)}"
        # 尝试清理资源
        if cap is not None: cap.release()
        return video_id, error_msg, 0, 0

    finally:
        # 无论如何，强制清理
        if cap is not None: cap.release()
        gc.collect()
        # 在某些 Windows 环境下，显式清理 OpenCV 窗口
        cv2.destroyAllWindows()

def main():
    print("="*60)
    print("🦴 MediaPipe 骨骼提取 (稳定模式 - 限制进程数)")
    print("="*60)

    print(f"📂 基准目录：{BASE_DIR}")
    print(f"🎬 视频源：{VIDEO_ROOT}")
    print(f"💾 输出地：{OUTPUT_ROOT}")
    print(f"⚙️ 并行进程数：{NUM_WORKERS} (已限制以防崩溃)")
    print("-" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 '{MODEL_REL_PATH}'")
        return

    if not os.path.exists(VIDEO_ROOT):
        print(f"❌ 错误：找不到视频目录 '{VIDEO_REL_PATH}'")
        return

    print(f"🔍 正在扫描视频...")
    tasks = []
    video_exts = ('.mp4', '.avi', '.mov')

    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.lower().endswith(video_exts):
                vid_path = os.path.join(root, f)
                rel_path = os.path.relpath(vid_path, VIDEO_ROOT)
                out_name = os.path.splitext(rel_path)[0] + '.npy'
                out_path = os.path.join(OUTPUT_ROOT, out_name)

                if os.path.exists(out_path):
                    continue

                tasks.append((vid_path, out_path, os.path.splitext(f)[0]))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("✅ 所有视频已处理完毕。")
        return

    print(f"📝 待处理：{total_tasks} 个视频")
    print("-" * 60)

    success_count = 0
    fail_count = 0
    bad_quality_count = 0

    start_time = time.time()

    # 保持默认，主要靠限制进程数和资源清理来稳定

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_video, t): t for t in tasks}

        for i, future in enumerate(as_completed(future_to_task)):
            try:
                vid_id, status, frames, missing = future.result()

                if status == "OK":
                    success_count += 1
                    symbol = "✅"
                elif status == "BAD_QUALITY":
                    bad_quality_count += 1
                    symbol = "⚠️"
                else:
                    fail_count += 1
                    symbol = "❌"
                    print(f"   [ERROR] {vid_id}: {status}") # 打印具体错误

                if i % 5 == 0 or i == total_tasks - 1:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1) if (i+1) > 0 else 0
                    eta = avg_time * (total_tasks - i - 1)
                    print(f"[{i+1}/{total_tasks}] {symbol} {vid_id} | Frames: {frames} | ETA: {eta:.0f}s")

            except BrokenProcessPool as e:
                print(f"\n❌ 致命错误：进程池崩溃！({e})")
                print("💡 建议：电脑内存可能不足，请关闭其他软件，或将 NUM_WORKERS 改为 1。")
                break
            except Exception as e:
                print(f"\n❌ 获取结果时出错：{e}")
                fail_count += 1

    end_time = time.time()
    print("-" * 60)
    print("🎉 执行结束")
    print(f"⏱️  耗时：{end_time - start_time:.2f}s")
    print(f"📊 成功：{success_count} | 低质：{bad_quality_count} | 失败：{fail_count}")

if __name__ == '__main__':
    # Windows 多进程入口保护
    from multiprocessing import freeze_support
    freeze_support()
    main()
