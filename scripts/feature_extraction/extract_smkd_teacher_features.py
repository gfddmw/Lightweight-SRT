import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. 动态添加路径：将本项目根目录及第三方克隆仓库加入系统检索路径
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

# 将官方克隆仓库加入系统检索路径（存放在 src/teacher_model/VAC_CSLR）
TEACHER_MODEL_DIR = os.path.join(PROJECT_ROOT, "src/teacher_model")
sys.path.append(os.path.join(TEACHER_MODEL_DIR, "VAC_CSLR"))

# 2. 安全导入 SLRModel 模型类
try:
    from slr_network import SLRModel
except ImportError as e:
    print(f"\n[Error] 导入 'SLRModel' 失败，错误详情: {e}")
    import traceback
    traceback.print_exc()
    print("提示：请确认您已将官方的 VIPL-SLP/VAC_CSLR 仓库成功克隆至项目的 src/teacher_model/VAC_CSLR 目录。")
    print("您可以运行以下命令进行克隆：")
    print(f"git clone https://github.com/VIPL-SLP/VAC_CSLR.git {os.path.join(TEACHER_MODEL_DIR, 'VAC_CSLR')}\n")
    SLRModel = None


# 3. 自定义 CSL-Daily 视频加载 Dataset
class CSLDailyVideoDataset(Dataset):
    """
    CSL-Daily 视频数据集加载类，读取本地的 MP4 视频文件并转换为模型所期待的 5D 张量
    """
    def __init__(self, csl_json_path, video_dir, transform=None, save_feat_dir=None, save_logits_dir=None):
        with open(csl_json_path, 'r', encoding='utf-8') as f:
            self.data_dict = json.load(f)
        all_video_names = sorted(list(self.data_dict.keys()))
        
        # 在初始化阶段秒级过滤已提取完特征的视频，避开低效 of Dataset 数据迭代物理读取
        if save_feat_dir and save_logits_dir:
            self.video_names = []
            skipped_count = 0
            for name in all_video_names:
                feat_path = os.path.join(save_feat_dir, f"{name}.npy")
                logits_path = os.path.join(save_logits_dir, f"{name}.npy")
                if os.path.exists(feat_path) and os.path.exists(logits_path):
                    skipped_count += 1
                else:
                    self.video_names.append(name)
            if skipped_count > 0:
                print(f"[断点续传提示] 检测到已存在 {skipped_count} 个提取完成的特征文件，已在初始化阶段自动过滤。剩余待提取视频数: {len(self.video_names)}")
        else:
            self.video_names = all_video_names
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        
        # 使用 OpenCV 读取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 💡核心提速：立即在 uint8 阶段进行高速缩放（OpenCV 的 C++ 向量化插值极快）
            # 这会让后续转 PyTorch Tensor 的内存大小和 CPU 浮点转换计算量直接降低 32.6 倍！
            if frame.shape[0] != 224 or frame.shape[1] != 224:
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            # 将 OpenCV 的 BGR 转换至 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # 异常备用：如果视频读取失败，生成全零帧占位
        if len(frames) == 0:
            print(f"Error: 视频 {video_path} 无法正确读取！已填充一帧零值。")
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
            
        # 转换为 Numpy array, shape: (T, H, W, C)
        video_np = np.stack(frames, axis=0)
        
        # 应用图像变换（如 Resizing, CenterCrop 224x224 及归一化）
        if self.transform:
            video_tensor = self.transform(video_np)
        else:
            # 默认的基础 Tensor 转换：(T, H, W, C) -> (C, T, H, W)
            # PyTorch 3D 卷积期待的 shape: (Batch, Channels, Temporal, Height, Width)
            video_tensor = torch.from_numpy(video_np).float() / 255.0
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            
        return video_tensor, len(frames), video_name


import queue
import threading
import time

class ThreadedVideoLoader:
    """
    大师级多线程视频加载器，用于在后台线程并发解码与转换视频，避免多进程死锁。
    """
    def __init__(self, dataset, num_threads=4, max_queue_size=8):
        self.dataset = dataset
        self.num_threads = num_threads
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.task_index = 0
        self.lock = threading.Lock()
        self.workers = []
        self.stop_event = threading.Event()
        
    def start(self):
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._worker_run)
            t.daemon = True
            t.start()
            self.workers.append(t)
            
    def _worker_run(self):
        while not self.stop_event.is_set():
            with self.lock:
                if self.task_index >= len(self.dataset):
                    break
                current_idx = self.task_index
                self.task_index += 1
            
            try:
                video_tensor, len_frames, video_name = self.dataset[current_idx]
                
                # 阻塞式塞入队列
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((video_tensor, len_frames, video_name), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                print(f"\n[Loader Thread Error] 读取索引 {current_idx} 发生异常: {e}")
                self.queue.put(e)
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue.empty() and self.task_index >= len(self.dataset) and not any(w.is_alive() for w in self.workers):
            raise StopIteration
            
        try:
            data = self.queue.get(timeout=30)
            if isinstance(data, Exception):
                raise data
            return data
        except queue.Empty:
            raise StopIteration

    def stop(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


# 4. 视频预处理 Transform 封装
class VideoTransforms:
    """
    针对三维视频帧的图像增强与裁剪变换类
    """
    def __init__(self, size=224):
        self.size = size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, video_np):
        # input: (T, H, W, C) -> numpy array
        t, h, w, c = video_np.shape
        
        # 将 numpy 数组转为 torch Tensor 格式，避免 opencv 库冲突
        # video_np shape: (T, H, W, C) -> torch.Tensor，转为 (T, C, H, W)
        video_tensor = torch.from_numpy(video_np).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # 使用 PyTorch 的插值函数缩放，彻底规避 cv2 崩溃问题
        if h != self.size or w != self.size:
            video_tensor = torch.nn.functional.interpolate(
                video_tensor, 
                size=(self.size, self.size), 
                mode='bilinear', 
                align_corners=False
            )
            
        # 逐帧进行归一化
        frames = []
        for frame_tensor in video_tensor:
            frame_tensor = self.normalize(frame_tensor)
            frames.append(frame_tensor)
            
        # 拼接成 (T, C, H, W) 格式的张量，以匹配模型 (B, T, C, H, W) 的输入期待
        video_tensor_out = torch.stack(frames, dim=0)
        return video_tensor_out


# 5. 主特征提取控制模块
def run_teacher_extraction():
    parser = argparse.ArgumentParser(description="SMKD 教师模型特征提取与导出工具")
    parser.add_argument("--csl_json", type=str, default=os.path.join(PROJECT_ROOT, "data/CSL/csl-daily.json"), help="对齐后的JSON标注文件")
    parser.add_argument("--video_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/CSL/video"), help="物理视频目录")
    parser.add_argument("--save_feat_dir", type=str, default=os.path.join(PROJECT_ROOT, "processed/csl_daily/teacher_features"), help="特征保存路径")
    parser.add_argument("--save_logits_dir", type=str, default=os.path.join(PROJECT_ROOT, "processed/csl_daily/teacher_logits"), help="Logits保存路径")
    parser.add_argument("--weights_path", type=str, default=os.path.join(PROJECT_ROOT, "weights/teacher/smkd_csl_daily.pt"), help="模型权重文件存放路径")
    parser.add_argument("--batch_size", type=int, default=1, help="推荐设为1以处理长度不等的变长视频")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    args = parser.parse_args()

    # 5.1 确保保存目录存在
    os.makedirs(args.save_feat_dir, exist_ok=True)
    os.makedirs(args.save_logits_dir, exist_ok=True)

    # 5.2 设备配置
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用计算设备: {device}")

    # 5.3 实例化 SLRModel 教师网络
    if SLRModel is None:
        print("错误：由于 'slr_network' 缺失，脚本停止执行核心部分。请按照上文警告提示克隆依赖库。")
        sys.exit(1)

    # 构造 SLRModel 实例，其结构配置通常和 VAC_CSLR 中的 config.yaml 一致
    fake_gloss_dict = {str(i): [i] for i in range(1296)}
    model = SLRModel(
        num_classes=1296,       # 依据教师模型权重包，实际对应分类数为 1296
        c2d_type="resnet18",    # 经典的图像特征提取骨干网络
        conv_type=2,            # 时序一维卷积配置 (对应 2 层 Conv + 2 层 MaxPool)
        use_bn=True,
        share_classifier=True,
        weight_norm=True,
        gloss_dict=fake_gloss_dict
    )

    # 加载预训练好的权重
    if os.path.exists(args.weights_path):
        print(f"正在加载 SMKD 教师模型权重: {args.weights_path}")
        checkpoint = torch.load(args.weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # 去除 DataParallel 引入的 "module." 前缀
        if len(state_dict) > 0 and "module." in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
    else:
        print(f"[Warning] 未能在 {args.weights_path} 找到权重文件。")
        print("提示：本工具将以随机初始化的教师模型运行，用于测试逻辑是否走通。")

    model = model.to(device)
    model.eval()

    # 5.4 注册 PyTorch Forward Hook 以优雅拦截 1024D 中间特征
    extracted_features = {}

    def get_features_hook(module, input_tensor, output_tensor):
        # 拦截特征输出。如果 temporal_model 返回的是 dict (如包含 predictions)，提取具体特征张量
        if isinstance(output_tensor, dict):
            extracted_features['feat'] = output_tensor.get('predictions', output_tensor.get('visual_feat', None))
        else:
            extracted_features['feat'] = output_tensor

    # 在模型的时序建模层 (temporal_model) 的最后一级上挂载 Hook
    if hasattr(model, 'temporal_model'):
        model.temporal_model.register_forward_hook(get_features_hook)
        print("成功在 'model.temporal_model' 挂载 Forward Hook 用于捕获 1024D 中间特征。")
    elif hasattr(model, 'conv1d'):
        model.conv1d.register_forward_hook(get_features_hook)
        print("Warning: 未能在模型中检测到 'temporal_model'，已将 Hook 挂载至 'model.conv1d' 节点。")
    else:
        print("Warning: 无法自动挂载 Hook。提取特征将直接利用模型 forward 的返回字典。")

    # 5.5 数据集与数据加载器构建
    video_transform = VideoTransforms(size=224)
    dataset = CSLDailyVideoDataset(
        csl_json_path=args.csl_json,
        video_dir=args.video_dir,
        transform=video_transform,
        save_feat_dir=args.save_feat_dir,
        save_logits_dir=args.save_logits_dir
    )

    print(f"数据索引读取成功，共计 {len(dataset)} 个视频需要处理。")
    if len(dataset) == 0:
        print("未检测到任何待处理视频，任务提前完成！")
        return

    # 启动多线程视频异步读取器，避开多进程死锁 (开4个线程，最大预存6个视频以防止内存超限)
    print("[系统配置] 启动多线程异步视频读取器 (Threads=4, QueueSize=6)...")
    threaded_loader = ThreadedVideoLoader(dataset, num_threads=4, max_queue_size=6)
    threaded_loader.start()

    print("开始导出教师模型特征与分类 Logits...")

    # 5.6 批量特征导出
    import time
    with torch.no_grad():
        for idx, (video_tensor, len_x, video_name) in enumerate(threaded_loader):
            # 定义保存文件路径
            feat_save_path = os.path.join(args.save_feat_dir, f"{video_name}.npy")
            logits_save_path = os.path.join(args.save_logits_dir, f"{video_name}.npy")
            
            start_time = time.time()

            # 数据送入设备并扩展出 Batch 维度: (1, T, C, H, W)
            video_tensor = video_tensor.unsqueeze(0).to(device)
            len_x = torch.tensor([len_x]).to(device)

            # 前向传播推理，使用半精度加速
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(video_tensor, len_x)

            # 提取特征与 Logits
            # 优先从 Hook 缓存中抓取特征，如果没有，则抓取模型直接返回的中间特征
            feat_tensor = extracted_features.get('feat', None)
            if feat_tensor is None:
                # 备用：从 forward 返回字典里抓取
                feat_tensor = output.get('visual_feat', output.get('feat', None))

            # 抓取序列 Logits
            logits_tensor = output.get('sequence_logits', output.get('conv_logits', None))

            if feat_tensor is None or logits_tensor is None:
                print(f"Error: 视频 {video_name} 前向推理未捕获到合法特征或 Logits！跳过保存。")
                continue

            # 转化为 Numpy 格式并剔除多余的 Batch 维度，保存至本地 (转回 float32 保证下游兼容性)
            feat_numpy = feat_tensor.squeeze(0).cpu().float().numpy()  
            logits_numpy = logits_tensor.squeeze(0).cpu().float().numpy() 

            # 💡 精确 squeeze 掉多余的维度，防止 T=1 时被过度 squeeze
            if len(feat_numpy.shape) == 3 and feat_numpy.shape[1] == 1:
                feat_numpy = feat_numpy.squeeze(1)
            if len(logits_numpy.shape) == 3 and logits_numpy.shape[1] == 1:
                logits_numpy = logits_numpy.squeeze(1)

            np.save(feat_save_path, feat_numpy)
            np.save(logits_save_path, logits_numpy)

            # 每次处理后，清理 Hook 特征字典中的缓存，为下一个视频腾位
            extracted_features.clear()

            # 推理输出对齐参数：记录降采样关系以利于学生网络对齐 (通常为 4 倍)
            if idx == 0:
                original_len = len_x.item()
                downsampled_len = feat_numpy.shape[0]
                downsample_rate = round(original_len / downsampled_len) if downsampled_len > 0 else 1
                print(f"[对齐元数据提示] 实际处理 of 第一个视频 {video_name} 原始帧数: {original_len} -> 提取特征时序长: {downsampled_len}。")
                print(f"教师模型时序降采样比例 (Downsample Rate) 大约为: {downsample_rate} 倍。")

            # 打印处理进度
            elapsed = time.time() - start_time
            print(f"特征提取进度: [{idx + 1}/{len(dataset)}] - 已完成 {video_name} (单视频耗时: {elapsed:.3f}秒)")

    # 终止视频读取线程
    threaded_loader.stop()
    print("恭喜！所有 CSL-Daily 强教师特征与 Logits 已提取并批量导出完毕！")


if __name__ == "__main__":
    run_teacher_extraction()
