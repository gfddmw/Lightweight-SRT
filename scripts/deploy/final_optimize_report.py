import torch
import torch.nn as nn
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import sys
import os

# 确保能找到项目源代码
sys.path.append(os.path.join(os.getcwd(), 'src'))
from student_model.architecture.st_gcn import Model

def run_bench():
    # 1. 初始化模型参数 (根据 train_kd_fast.yaml)
    model_args = {
        "in_channels": 3,
        "num_class": 2000,
        "dropout": 0.5,
        "edge_importance_weighting": True,
        "graph_args": {"layout": 'openpose', "strategy": 'spatial'}
    }
    
    # 2. FP32 基线测量
    print("--- Phase 1: FP32 Baseline ---")
    model_fp32 = Model(**model_args).eval()
    dummy_input = torch.randn(1, 3, 64, 21, 1) # N, C, T, V, M
    
    # 统计参数量
    params = sum(p.numel() for p in model_fp32.parameters())
    size_fp32 = params * 4 / (1024**2) # MB
    
    # 推理时间 (CPU)
    for _ in range(10): _ = model_fp32(dummy_input) # Warmup
    start = time.time()
    for _ in range(50): _ = model_fp32(dummy_input)
    t_fp32 = (time.time() - start) / 50 * 1000 # ms
    
    print(f"FP32 Params: {params/1e6:.2f} M")
    print(f"FP32 Size: {size_fp32:.2f} MB")
    print(f"FP32 Latency: {t_fp32:.2f} ms")

    # 3. 静态量化 (针对 Android qnnpack)
    print("\n--- Phase 2: Static Quantization ---")
    model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model_fp32, inplace=True)
    
    # 校准
    print("Calibrating...")
    with torch.no_grad():
        for _ in range(20):
            model_fp32(torch.randn(1, 3, 64, 21, 1))
            
    # 转换
    print("Converting to INT8...")
    model_int8 = torch.quantization.convert(model_fp32, inplace=False)

    # 4. 移动端优化与导出
    print("\n--- Phase 3: Mobile Export ---")
    try:
        scripted_model = torch.jit.script(model_int8)
        optimized_model = optimize_for_mobile(scripted_model)
        output_path = "student_stgcn_optimized.ptl"
        optimized_model._save_for_lite_interpreter(output_path)
        
        size_int8 = os.path.getsize(output_path) / (1024**2)
        print(f"INT8 Optimized Size: {size_int8:.2f} MB")
        
        # 总结结果
        print("\n--- Final Summary ---")
        print(f"Storage Reduction: {(size_fp32 - size_int8)/size_fp32*100:.2f} %")
        print(f"Export Path: {output_path}")
        
    except Exception as e:
        print(f"Optimization Failed: {e}")

if __name__ == "__main__":
    run_bench()
