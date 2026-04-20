import torch
import time
import sys
import os

# 确保能找到项目源代码
sys.path.append(os.path.join(os.getcwd(), 'src'))
from teacher_model.architecture.pytorch_i3d import InceptionI3d

def benchmark_teacher():
    print("=== Teacher Model (I3D) Comprehensive Benchmark ===")
    
    # 1. 初始化模型
    model = InceptionI3d(num_classes=2000, in_channels=3)
    model.eval()
    
    # 典型视频输入: Batch=1, Channel=3, Time=64, H=224, W=224
    dummy_input = torch.randn(1, 3, 64, 224, 224)
    
    # 2. 参数量与计算量分析 (Params & MACs)
    print("\n[Phase 1] Analyzing Model Complexity...")
    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"Total Parameters: {params/1e6:.2f} M")
        print(f"Total MACs (FLOPs): {macs/1e9:.2f} G")
        print(f"Model Size (FP32): {params * 4 / (1024**2):.2f} MB")
    except ImportError:
        params = sum(p.numel() for p in model.parameters())
        print(f"thop not installed. Total Parameters: {params/1e6:.2f} M")

    # 3. 推理延迟测定 (Latency @ CPU)
    print("\n[Phase 2] Measuring Inference Latency (CPU)...")
    # Warmup
    print("Warming up (3 iterations)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
        
        # 实际测量
        print("Measuring (5 iterations)...")
        start = time.time()
        num_iters = 5
        for _ in range(num_iters):
            _ = model(dummy_input)
        latency = (time.time() - start) / num_iters * 1000 # ms
        
    print(f"Average Latency: {latency:.2f} ms")
    print(f"Estimated Throughput: {1000/latency:.2f} FPS")
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    benchmark_teacher()
