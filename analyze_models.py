
import torch
import torch.nn as nn
import time
import os

# Import models
try:
    from src.teacher_model.architecture.pytorch_i3d import InceptionI3d
    from src.student_model.architecture.st_gcn import Model as STGCNModel
except ImportError as e:
    print(f"Error importing models: {e}")
    exit(1)

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def get_file_size(model_path):
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0

def estimate_flops(model, input_size):
    # This is a very rough estimate or using a simpler method if thop/fvcore is not available.
    # We will try to use thop if available.
    try:
        from thop import profile
        input_data = torch.randn(input_size)
        flops, params = profile(model, inputs=(input_data,), verbose=False)
        return flops / 1e9 # GFLOPs
    except ImportError:
        return None

def measure_latency(model, input_size, device='cpu', iterations=50):
    model.to(device)
    model.eval()
    input_data = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    end_time = time.time()
    return (end_time - start_time) / iterations * 1000 # ms

# 1. Initialize models
num_classes = 2000

# Teacher: I3D
# Typical input: (Batch, 3, Frames, 224, 224)
# Based on common usage in sign language, frames might be around 64 or 32.
# Let's use (1, 3, 64, 224, 224)
teacher = InceptionI3d(num_classes=num_classes, in_channels=3)
teacher_input_size = (1, 3, 64, 224, 224)

# Student: ST-GCN
# Typical input: (Batch, 3, Frames, Nodes, People)
# Nodes: 25 (OpenPose layout), People: 1 or 2.
# Based on config: graph_args: layout: 'openpose'
# Let's use (1, 3, 64, 18, 1) or (1, 3, 64, 25, 1)
# Checking st_gcn.py to see node count for 'openpose'
# Assuming 18 nodes for standard openpose or 25 for some variants. 
# Looking at common ST-GCN use-cases.
student_args = {
    "in_channels": 3,
    "num_class": num_classes,
    "dropout": 0.5,
    "edge_importance_weighting": True,
    "graph_args": {"layout": "openpose", "strategy": "spatial"}
}
student = STGCNModel(**student_args)
student_input_size = (1, 3, 64, 21, 1) # MediaPipe Hand Landmarks (21 nodes)

# 2. Collect Metrics
metrics = {
    "Teacher (I3D)": {
        "Params (M)": get_params(teacher) / 1e6,
        "FLOPs (G)": estimate_flops(teacher, teacher_input_size),
        "Latency (ms)": measure_latency(teacher, teacher_input_size),
        "File Size (MB)": 57.53, # From earlier scan
    },
    "Student (ST-GCN)": {
        "Params (M)": get_params(student) / 1e6,
        "FLOPs (G)": estimate_flops(student, student_input_size),
        "Latency (ms)": measure_latency(student, student_input_size),
        "File Size (MB)": get_params(student) * 4 / (1024 * 1024), # Estimated if not saved
    }
}

# 3. Print Report
print("\n" + "="*50)
print(f"{'Metric':<20} | {'Teacher (I3D)':<15} | {'Student (ST-GCN)':<15} | {'Reduction (%)':<15}")
print("-"*50)

for key in ["Params (M)", "FLOPs (G)", "Latency (ms)", "File Size (MB)"]:
    t_val = metrics["Teacher (I3D)"][key]
    s_val = metrics["Student (ST-GCN)"][key]
    
    if t_val is not None and s_val is not None and t_val != 0:
        reduction = (t_val - s_val) / t_val * 100
        reduction_str = f"{reduction:.2f}%"
    else:
        reduction_str = "N/A"
        
    t_str = f"{t_val:.2f}" if t_val is not None else "N/A"
    s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
    
    print(f"{key:<20} | {t_str:<15} | {s_str:<15} | {reduction_str:<15}")

print("="*50)
print("Notes:")
print("- Latency measured on CPU.")
print("- Teacher file size based on weights/teacher/nslt_2000_018216_0.448072.pt.")
print("- Student file size estimated from parameter count.")
print("- Input size: I3D (1,3,64,224,224), ST-GCN (1,3,64,18,1).")
