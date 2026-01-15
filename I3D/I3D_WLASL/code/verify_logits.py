import numpy as np
import json
import os
import torch.nn.functional as F
import torch

# ================= 配置 =================
LOGITS_DIR = 'teacher_logits_2000'  # 你的 .npy 文件夹
JSON_FILE = 'preprocess/nslt_2000.json'  # 你的原始标签文件
CHECK_COUNT = 5  # 随机抽查几个


# =======================================

def verify():
    # 1. 加载标签映射表
    print(f"正在加载标签文件: {JSON_FILE} ...")
    with open(JSON_FILE, 'r') as f:
        # 兼容 List 和 Dict 格式
        data = json.load(f)

    # 把 JSON 转成 视频ID -> 真实类别(Label ID) 的字典
    id_to_label = {}
    if isinstance(data, list):  # List格式
        for item in data:
            # WLASL List格式通常只有 'video_id' 和 'gloss_id'
            # 或者 'action' (action[0] 是 label)
            vid = item.get('video_id', '')
            if 'action' in item:
                label = item['action'][0]
            elif 'gloss_id' in item:
                label = item['gloss_id']
            else:
                continue
            id_to_label[vid] = label
    else:  # Dict格式
        for vid, info in data.items():
            if 'action' in info:
                id_to_label[vid] = info['action'][0]

    # 2. 随机抽查
    npy_files = [f for f in os.listdir(LOGITS_DIR) if f.endswith('.npy')]
    if len(npy_files) == 0:
        print("❌ 错误：文件夹里没有 .npy 文件！")
        return

    print(f"找到 {len(npy_files)} 个特征文件。开始抽查...\n")

    # 随机选几个
    np.random.shuffle(npy_files)
    samples = npy_files[:CHECK_COUNT]

    correct_count = 0

    for file_name in samples:
        vid_id = file_name.split('.')[0]

        # 读取 Logits
        logits = np.load(os.path.join(LOGITS_DIR, file_name))

        # 基础检查：形状
        if logits.shape != (2000,):
            print(f"⚠️ 警告: {vid_id} 形状不对！期望 (2000,), 实际 {logits.shape}")
            continue

        # 找到模型预测的类别
        # 将 numpy 转 tensor 做 softmax (可选，直接 argmax 也可以)
        logits_t = torch.from_numpy(logits)
        probs = F.softmax(logits_t, dim=0)

        pred_label = torch.argmax(probs).item()
        confidence = probs[pred_label].item()

        # 获取真实标签
        true_label = id_to_label.get(vid_id, -1)

        # 打印对比
        status = "✅" if pred_label == true_label else "❌"
        if pred_label == true_label: correct_count += 1

        print(f"视频: {vid_id} | 真实标签: {true_label} | 预测标签: {pred_label} | 置信度: {confidence:.2f} | {status}")

    print(f"\n抽查结束。{CHECK_COUNT} 个样本中，教师模型预测对了 {correct_count} 个。")
    print("提示：如果教师模型准确率是 44%，那么 5 个里对 2-3 个是完全正常的。只要不是全错就行。")


if __name__ == '__main__':
    verify()
