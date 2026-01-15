import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pytorch_i3d import InceptionI3d
from train_i3d import videotransforms, transforms
from datasets.nslt_dataset_all import NSLT as Dataset

current_dir = os.path.dirname(os.path.realpath(__file__))

data_root_path = os.path.abspath(os.path.join(current_dir, '../../..', 'data', 'WLASL2000'))
# 查找json文件路径
json_file_path = os.path.join(current_dir, 'preprocess', 'nslt_2000.json')
# 教师模型路径
teach_weights_path = os.path.join(current_dir, 'checkpoints', 'nslt_2000_018216_0.448072.pt')

DATA_ROOT = data_root_path
# 2. 你的 JSON 路径
JSON_FILE = json_file_path
# 3. 你的最强教师模型路径
TEACHER_WEIGHTS = teach_weights_path
# 4. 结果保存位置
SAVE_DIR = 'teacher_logits_2000'


def run_extraction():
    # 1. 准备保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")

    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 3. 加载数据集
    # 注意：必须用 'train' set，因为我们要蒸馏训练集
    # 注意：shuffle=False，必须按顺序提取，方便后续通过文件名对齐
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(JSON_FILE, 'train', DATA_ROOT, 'rgb', test_transforms)

    # Batch_size 设为 1 是最稳妥的，防止视频长短不一导致 Padding 误差
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"共需提取 {len(dataset)} 个视频特征...")

    # 4. 加载教师模型
    i3d = InceptionI3d(2000, in_channels=3)
    # 处理 DataParallel 留下的 'module.' 前缀
    state_dict = torch.load(TEACHER_WEIGHTS, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    i3d.load_state_dict(state_dict)
    i3d.to(device)
    i3d.eval()  # 开启评估模式 (关闭 Dropout/BN 更新)

    # 5. 开始提取循环
    with torch.no_grad():  # 这一步很关键，不计算梯度，省显存且速度快
        for i, data in enumerate(dataloader):
            # 获取数据 (根据你的 Dataset 返回值调整)
            # 通常是 inputs, labels, video_id, ...
            inputs, labels, video_id = data

            inputs = inputs.to(device)  # inputs shape: (1, 3, T, 224, 224)

            # --- 模型推理 ---
            # I3D 输出通常是 (Batch, Classes, Time) -> (1, 2000, T)
            per_frame_logits = i3d(inputs)

            # --- 特征聚合 ---
            # 我们需要视频级的特征，所以在时间维度 T 上取平均
            # 结果变为 (1, 2000)
            video_logits = torch.mean(per_frame_logits, dim=2)

            # --- 保存 ---
            # 这里的 video_id 是个 tuple ('05723',)，取第0个元素
            vid_str = video_id[0]
            save_path = os.path.join(SAVE_DIR, f"{vid_str}.npy")

            # 转为 numpy 并保存
            # .cpu() 移回内存, .numpy() 转数组
            np.save(save_path, video_logits.cpu().numpy().flatten())

            if i % 100 == 0:
                print(f"进度: {i}/{len(dataset)} - 已保存 {vid_str}.npy")

    print("✅ 所有教师特征提取完毕！")


if __name__ == '__main__':
    run_extraction()
