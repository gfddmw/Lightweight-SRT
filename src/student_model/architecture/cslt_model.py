import torch
import torch.nn as nn
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中，以便直接运行测试
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TemporalAdapter(nn.Module):
    """
    TemporalAdapter 类 (nn.Module)：
    - 包含一维卷积，将 input_channels (默认 768) 转换为 output_channels (默认 1024)，并进行 4 倍时序下采样。
    - 使用 nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=4, padding=2)，配合 nn.BatchNorm1d 和 nn.ReLU。
    - 输入 x 的维度为 [B, T, in_channels]。前向中进行 x.permute(0, 2, 1) 转为 [B, in_channels, T]，
      然后送入卷积层做降采样和投影，最后再 permute 换回来，返回 [B, T_out, out_channels]。
    """
    def __init__(self, input_channels=768, output_channels=1024):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, T, in_channels]
        x = x.permute(0, 2, 1)  # [B, in_channels, T]
        x = self.conv(x)        # [B, out_channels, T_out]
        x = self.bn(x)          # [B, out_channels, T_out]
        x = self.relu(x)        # [B, out_channels, T_out]
        x = x.permute(0, 2, 1)  # [B, T_out, out_channels]
        return x


class CTCHead(nn.Module):
    """
    CTCHead 类 (nn.Module)：
    - 输入维度 in_channels (默认 1024)，输出维度 num_classes (默认 7388，即词表大小 7387 + 1 维 CTC blank)。
    - 内部仅含 nn.Linear(in_channels, num_classes) 全连接层。
    - 输入 x 维度为 [B, T_out, in_channels]，输出 [B, T_out, num_classes]。
    """
    def __init__(self, in_channels=1024, num_classes=7388):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, T_out, in_channels]
        return self.linear(x)


class CSLTModel(nn.Module):
    """
    CSLTModel 类 (nn.Module)：
    - 在 __init__ 中接收已经实例化的 encoder (即 MultiStreamSTGCN 模型)、in_channels (默认 768)、out_channels (默认 1024) 和 num_classes (默认 7388)。
    - 实例化 TemporalAdapter 和 CTCHead。
    - forward(self, joints, bones, motion) 方法：
      1. 调用 self.encoder(joints=joints, bones=bones, motion=motion, return_sequence=True) 获得拼接时序特征 [B, T, 768]。
      2. 送入 TemporalAdapter，获得 [B, T_out, 1024] 特征，我们记录为 adapted_feat。
      3. 将 adapted_feat 送入 CTCHead，获得 [B, T_out, num_classes] 的 Logits，记录为 ctc_logits。
      4. 返回一个字典：{"adapted_feat": adapted_feat, "ctc_logits": ctc_logits}。
    """
    def __init__(self, encoder, in_channels=768, out_channels=1024, num_classes=7388):
        super().__init__()
        self.encoder = encoder
        self.temporal_adapter = TemporalAdapter(input_channels=in_channels, output_channels=out_channels)
        self.ctc_head = CTCHead(in_channels=out_channels, num_classes=num_classes)

    def forward(self, joints, bones, motion):
        # 获取拼接时序特征 [B, T, in_channels] (默认 in_channels = 768)
        feat = self.encoder(joints=joints, bones=bones, motion=motion, return_sequence=True)
        # 时序下采样和特征映射 [B, T_out, out_channels] (默认 out_channels = 1024)
        adapted_feat = self.temporal_adapter(feat)
        # 获取 CTC 预测 logits [B, T_out, num_classes] (默认 num_classes = 7388)
        ctc_logits = self.ctc_head(adapted_feat)
        
        return {
            "adapted_feat": adapted_feat,
            "ctc_logits": ctc_logits
        }


if __name__ == '__main__':
    from src.student_model.architecture.multi_stream_stgcn import MultiStreamSTGCN
    
    print("Testing CSLTModel initialization and forward pass...")
    
    # 1. 实例化 MultiStreamSTGCN 编码器
    # 输入通道为 3 (x, y, z)，时空图布局为 openpose (默认 21 个节点)
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    encoder = MultiStreamSTGCN(
        num_class=10, 
        in_channels=3, 
        graph_args=graph_args,
        edge_importance_weighting=True
    )
    
    # 2. 实例化 CSLT 连续时序联合模型
    # encoder 拼接出的特征维度是 3 * 256 = 768
    model = CSLTModel(
        encoder=encoder,
        in_channels=768,
        out_channels=1024,
        num_classes=7388
    )
    
    # 3. 构造虚拟的多流输入数据
    # 输入形状为: [B, C, T, V, M]
    # B (Batch size) = 2
    # C (Channels) = 3 (x, y, z)
    # T (Temporal frames) = 100
    # V (Vertices/Nodes) = 21
    # M (Person/Instances) = 1
    B, C, T, V, M = 2, 3, 100, 21, 1
    
    joints = torch.randn(B, C, T, V, M)
    bones = torch.randn(B, C, T, V, M)
    motion = torch.randn(B, C, T, V, M)
    
    print(f"Input shapes: joints/bones/motion = {joints.shape}")
    
    # 4. 进行前向传播
    outputs = model(joints=joints, bones=bones, motion=motion)
    
    # 5. 打印输出形状，并验证
    adapted_feat = outputs["adapted_feat"]
    ctc_logits = outputs["ctc_logits"]
    
    print(f"adapted_feat shape: {adapted_feat.shape} (Expected: [2, 25, 1024])")
    print(f"ctc_logits shape: {ctc_logits.shape} (Expected: [2, 25, 7388])")
    
    # 自检 assertion
    assert adapted_feat.shape == (B, 25, 1024), f"adapted_feat shape mismatch: {adapted_feat.shape}"
    assert ctc_logits.shape == (B, 25, 7388), f"ctc_logits shape mismatch: {ctc_logits.shape}"
    
    print("CSLTModel unit test passed successfully!")
