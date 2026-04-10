import os
import sys
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# 将项目根目录添加到路径中，以便加载模型
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.student_model.architecture.st_gcn import Model

def export():
    # 1. 路径配置
    weights_path = os.path.join(PROJECT_ROOT, "work_dir/recognition/wlasl2000/ST_GCN_KD/epoch70_model.pt")
    save_path = os.path.join(PROJECT_ROOT, "android/app/src/main/assets/st_gcn_student.ptl")

    if not os.path.exists(weights_path):
        print(f"错误: 未找到权重文件 {weights_path}")
        return

    # 2. 初始化模型架构 (参数需与训练配置 train_kd.yaml 一致)
    print("正在加载模型架构...")
    model = Model(
        in_channels=3,
        num_class=2000,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={
            'layout': 'openpose',  # 根据你训练时的配置
            'strategy': 'spatial'
        }
    )

    # 3. 加载权重
    print(f"正在加载权重: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')

    # 如果是用 DataParallel 训练的，移除 'module.' 前缀
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    # 4. 创建示例输入 (Dummy Input)
    # ST-GCN 输入形状: (Batch, Channels, Time, Joints, Person)
    # WLASL 默认 Time=64, OpenPose 布局 Joints=18 或 根据你的提取设定
    # 注意：这里的 Joints 数量必须与训练时的骨骼提取逻辑一致
    # 假设手部关键点提取为 21 点，如果是 OpenPose 整体则是 18
    # 我们先根据模型内部 graph 获取点的数量
    num_joints = model.graph.num_node
    print(f"检测到图节点数量 (Joints): {num_joints}")

    example_input = torch.rand(1, 3, 64, num_joints, 1)

    # 5. 转换为 TorchScript (Tracing)
    print("正在进行 TorchScript Tracing...")
    try:
        traced_script_module = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"Tracing 失败: {e}")
        print("尝试使用 Scripting 模式...")
        traced_script_module = torch.jit.script(model)

    # 6. 移动端优化
    print("正在进行移动端性能优化...")
    optimized_module = optimize_for_mobile(traced_script_module)

    # 7. 保存文件
    assets_dir = os.path.dirname(save_path)
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    optimized_module._save_for_lite_interpreter(save_path)
    print(f"成功! 移动端模型已保存至: {save_path}")
    print(f"文件大小: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    export()
