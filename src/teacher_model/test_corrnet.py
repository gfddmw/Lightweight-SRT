import os
import sys
import torch

# 动态添加路径
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(CURRENT_DIR, "CorrNet"))

try:
    from slr_network import SLRModel
    print("成功从 slr_network 导入 SLRModel！")
except ImportError as e:
    print(f"导入 SLRModel 失败: {e}")
    sys.exit(1)

def test_model():
    print("正在实例化 SLRModel...")
    fake_gloss_dict = {str(i): [i] for i in range(2001)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用推理设备: {device}")

    model = SLRModel(
        num_classes=2001,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=True,
        share_classifier=True,
        weight_norm=True,
        gloss_dict=fake_gloss_dict
    ).to(device)
    model.eval()
    print("SLRModel 实例化成功！")

    # 构造 dummy 输入: shape (B, T, C, H, W) = (1, 16, 3, 224, 224)
    # 模拟 16 帧的视频
    dummy_video = torch.randn(1, 16, 3, 224, 224).to(device)
    dummy_len = torch.tensor([16]).to(device)

    print("执行 Dummy Forward 推理...")
    with torch.no_grad():
        output = model(dummy_video, dummy_len)

    print("\n--- 推理输出详情 ---")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"键: {k:<25} | 形状: {list(v.shape)}")
        else:
            print(f"键: {k:<25} | 值: {v}")

if __name__ == "__main__":
    test_model()
