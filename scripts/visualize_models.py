import torch
import os
import sys

# 将 src 加入路径以便在 scripts 目录下运行
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from student_model.architecture.st_gcn import Model as STGCN
from teacher_model.architecture.pytorch_i3d import InceptionI3d as I3D

def visualize_stgcn():
    print("--- 正在导出 ST-GCN ---")
    # 模拟参数: 该项目实际使用的是 MediaPipe 21点布局
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    # 创建模型实例
    # in_channels=3 (x, y, confidence), num_class=2000 (WLASL2000)
    model = STGCN(
        in_channels=3, 
        num_class=2000, 
        graph_args=graph_args, 
        edge_importance_weighting=True
    )
    model.eval()
    
    # 模拟输入: (N, C, T, V, M)
    # N=1, C=3, T=30 (帧), V=21 (MediaPipe 手部关节点), M=1 (人数)
    dummy_input = torch.randn(1, 3, 30, 21, 1) 
    
    output_path = "visualizations/st_gcn_architecture.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        input_names=['skeleton_input'], 
        output_names=['action_logits'],
        opset_version=12,
        verbose=False
    )
    print(f"成功：ST-GCN 已导出至 {output_path}")

def visualize_i3d():
    print("--- 正在导出 I3D ---")
    # 创建模型实例
    model = I3D(num_classes=2000, in_channels=3)
    model.eval()
    
    # 模拟输入: (N, C, T, H, W)
    # N=1, C=3, T=64 (帧), H=224, W=224
    dummy_input = torch.randn(1, 3, 64, 224, 224)
    
    output_path = "visualizations/i3d_architecture.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        input_names=['video_input'], 
        output_names=['action_logits'],
        opset_version=12,
        verbose=False
    )
    print(f"成功：I3D 已导出至 {output_path}")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("visualizations", exist_ok=True)
    
    print("开始生成模型结构文件...")
    
    try:
        visualize_stgcn()
    except Exception as e:
        print(f"ST-GCN 导出失败: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*30 + "\n")
    
    try:
        visualize_i3d()
    except Exception as e:
        print(f"I3D 导出失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n提示: 请将生成的 .onnx 文件拖入 https://netron.app/ 即可查看可视化结构图。")
