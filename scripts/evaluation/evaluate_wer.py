import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms

# 1. 动态添加系统检索路径
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src/teacher_model/CorrNet"))

try:
    from slr_network import SLRModel
except ImportError as e:
    print(f"[Error] 导入 'SLRModel' 失败: {e}")
    sys.exit(1)

# 2. 图像归一化与预处理
class VideoTransforms:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, video_np):
        video_tensor = torch.from_numpy(video_np).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        if video_tensor.shape[2] != self.size or video_tensor.shape[3] != self.size:
            video_tensor = torch.nn.functional.interpolate(
                video_tensor, 
                size=(self.size, self.size), 
                mode='bilinear', 
                align_corners=False
            )
            
        return video_tensor / 127.5 - 1.0

# 3. 动态规划计算编辑距离 (Levenshtein Distance)
def calculate_edit_distance(ref, hyp):
    """
    计算两个词序列列表的最小编辑距离。
    ref: 参考答案 (ground truth) 词列表，例如 ['你', '好']
    hyp: 预测结果 (hypothesis) 词列表，例如 ['你', '们', '好']
    """
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref)][len(hyp)]

def main():
    parser = argparse.ArgumentParser(description="CSL-Daily 教师模型错词率 (WER) 评测工具")
    parser.add_argument("--csl_json", type=str, default=os.path.join(PROJECT_ROOT, "data/CSL/csl-daily.json"), help="标注 JSON 文件路径")
    parser.add_argument("--video_dir", type=str, default=os.path.join(PROJECT_ROOT, "data/CSL/video"), help="物理视频目录")
    parser.add_argument("--weights_path", type=str, default=os.path.join(PROJECT_ROOT, "weights/teacher/smkd_csl_daily_real.pt"), help="评估的模型权重路径")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"], help="评估的数据子集")
    parser.add_argument("--limit", type=int, default=-1, help="限制评估的样本数量，-1表示全量评估（用于快速冒烟测试）")
    parser.add_argument("--device", type=str, default="cuda", help="运行推理的设备")
    args = parser.parse_args()

    # 3.1 加载标注文件
    if not os.path.exists(args.csl_json):
        print(f"Error: 找不到标注文件 {args.csl_json}")
        sys.exit(1)
        
    with open(args.csl_json, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        
    # 筛选 split 对应子集
    eval_keys = [k for k, v in data_dict.items() if v.get("split") == args.split]
    eval_keys = sorted(eval_keys)
    
    if args.limit > 0:
        eval_keys = eval_keys[:args.limit]
        print(f"[提示] 已限制评估样本数量为前 {args.limit} 个视频。")

    total_samples = len(eval_keys)
    if total_samples == 0:
        print(f"Error: 在子集 '{args.split}' 下未筛选到任何样本，请检查 JSON 内容。")
        sys.exit(1)
        
    print(f"成功载入 CSL-Daily 标注。评估子集: {args.split} | 待评估样本总数: {total_samples}")

    # 3.2 尝试加载真实手语词汇表词典
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用评估推理设备: {device}")

    gloss_dict_path = os.path.join(PROJECT_ROOT, "src/teacher_model/CorrNet/preprocess/CSL-Daily/gloss_dict.npy")
    if os.path.exists(gloss_dict_path):
        print(f"正在载入真实 CSL-Daily 词汇表: {gloss_dict_path}")
        gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        num_classes = len(gloss_dict) + 1
    else:
        print("[Warning] 找不到真实 CSL-Daily 词表文件，将使用 fake_gloss_dict 进行评估 (输出结果为词ID)")
        gloss_dict = {str(i): [i] for i in range(2001)}
        num_classes = 2001

    model = SLRModel(
        num_classes=num_classes,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=True,
        share_classifier=True,
        weight_norm=True,
        gloss_dict=gloss_dict
    )

    if not os.path.exists(args.weights_path):
        print(f"Error: 未能在 {args.weights_path} 找到权重文件！")
        sys.exit(1)

    print(f"正在加载预训练权重: {args.weights_path}")
    checkpoint = torch.load(args.weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key_name = k.replace("module.", "")
        new_state_dict[key_name] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("模型权重全载入成功！(Strict=True)")
    model = model.to(device)
    model.eval()

    # 3.3 循环前向评估与 WER 计算
    transform = VideoTransforms(size=224)
    total_edit_distance = 0
    total_words = 0
    
    import time
    start_eval_time = time.time()
    
    print("\n开始逐视频推理并计算 WER 错词率...\n" + "="*60)

    for idx, name in enumerate(eval_keys):
        video_path = os.path.join(args.video_dir, f"{name}.mp4")
        if not os.path.exists(video_path):
            print(f"[{idx+1}/{total_samples}] Warning: 找不到视频文件 {video_path}，已跳过！")
            continue
            
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[0] != 224 or frame.shape[1] != 224:
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            print(f"[{idx+1}/{total_samples}] Warning: 视频 {name}.mp4 读取失败，已跳过！")
            continue

        video_np = np.stack(frames, axis=0)
        video_tensor = transform(video_np).unsqueeze(0).to(device)
        len_x = torch.tensor([len(frames)]).to(device)

        # 前向推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(video_tensor, len_x)

        # 获取解码后的词汇序列
        recognized_sents = output.get("recognized_sents", [])
        # 结果结构为 [[(gloss_word, idx), ...]]
        pred_words = []
        if len(recognized_sents) > 0 and len(recognized_sents[0]) > 0:
            pred_words = [item[0] for item in recognized_sents[0]]

        # 读取真实标签的 gloss 并切分为列表
        ref_gloss_str = data_dict[name].get("gloss", "").strip()
        ref_words = ref_gloss_str.split()

        # 计算该样本的编辑距离
        dist = calculate_edit_distance(ref_words, pred_words)
        total_edit_distance += dist
        total_words += len(ref_words)

        current_wer = (dist / len(ref_words)) * 100 if len(ref_words) > 0 else 0.0
        
        # 实时输出当前样本比对
        print(f"[{idx+1}/{total_samples}] 视频 ID: {name}")
        print(f"  预测: {' '.join(pred_words)}")
        print(f"  真实: {ref_gloss_str}")
        print(f"  词数: {len(ref_words)} | 编辑距离: {dist} | 当前 WER: {current_wer:.2f}%")
        print("-"*60)

    # 3.4 最终评测统计汇总
    end_eval_time = time.time()
    elapsed = end_eval_time - start_eval_time
    
    if total_words == 0:
        print("\n[Error] 未处理任何有效样本，无法计算 WER！")
        return
        
    final_wer = (total_edit_distance / total_words) * 100
    print("\n" + "="*60 + "\n[Summary] 评测汇总结果:")
    print(f"  数据集划分 (Split)  : {args.split}")
    print(f"  总评估样本数        : {total_samples}")
    print(f"  总测试手语词数 (N)  : {total_words}")
    print(f"  累计编辑距离 (S+D+I): {total_edit_distance}")
    print(f"  平均单视频推理耗时  : {elapsed / total_samples:.3f} 秒")
    print(f"  最终错词率 (WER)    : {final_wer:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
