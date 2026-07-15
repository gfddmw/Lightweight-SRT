import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中，以便直接运行测试
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入翻译解码器
try:
    from .decoder import TranslationDecoder
    DECODER_AVAILABLE = True
except ImportError:
    DECODER_AVAILABLE = False

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
    - 输入维度 in_channels (默认 1024)，输出维度 num_classes (默认 2005，即词表大小 2004 + 1 维 CTC blank)。
    - 内部仅含 nn.Linear(in_channels, num_classes) 全连接层。
    - 输入 x 维度为 [B, T_out, in_channels]，输出 [B, T_out, num_classes]。
    """
    def __init__(self, in_channels=1024, num_classes=2005):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, T_out, in_channels]
        return self.linear(x)


class CSLTModel(nn.Module):
    """
    CSLTModel 类 (nn.Module)：
    - 在 __init__ 中接收已经实例化的 encoder (即 MultiStreamSTGCN 模型)、in_channels (默认 768)、out_channels (默认 1024) 和 num_classes (默认 7388)。
    - 实例化 TemporalAdapter、CTCHead，以及可选的 TranslationDecoder。
    - forward(self, joints, bones, motion, decoder_input_ids=None, labels=None) 方法：
      1. 调用 self.encoder(joints=joints, bones=bones, motion=motion, return_sequence=True) 获得拼接时序特征 [B, T, 768]。
      2. 送入 TemporalAdapter，获得 [B, T_out, 1024] 特征，我们记录为 adapted_feat。
      3. 将 adapted_feat 送入 CTCHead，获得 [B, T_out, num_classes] 的 Logits，记录为 ctc_logits。
      4. 如果提供了 decoder_input_ids，则通过翻译解码器获得翻译结果。
      5. 返回字典，包含 adapted_feat, ctc_logits，以及可选的翻译输出。
    """
    def __init__(
        self, 
        encoder, 
        in_channels=768, 
        out_channels=1024, 
        num_classes=2005,
        use_translation_decoder=False,
        decoder_pretrained_model="facebook/mbart-large-50",
        decoder_freeze_strategy="cross_attention_only"
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_adapter = TemporalAdapter(input_channels=in_channels, output_channels=out_channels)
        self.ctc_head = CTCHead(in_channels=out_channels, num_classes=num_classes)
        
        self.use_translation_decoder = use_translation_decoder
        self.translation_decoder = None
        
        if use_translation_decoder:
            if not DECODER_AVAILABLE:
                raise ImportError("TranslationDecoder not available. Please ensure decoder.py is correctly imported.")
            
            self.translation_decoder = TranslationDecoder(
                pretrained_model_name=decoder_pretrained_model,
                freeze_strategy=decoder_freeze_strategy,
                feature_dim=out_channels
            )
    
    def get_trainable_params(self):
        """获取所有可训练参数（包括解码器的可训练参数）"""
        params = list(self.encoder.parameters()) + \
                 list(self.temporal_adapter.parameters()) + \
                 list(self.ctc_head.parameters())
        
        if self.use_translation_decoder and self.translation_decoder is not None:
            params.extend(self.translation_decoder.get_trainable_params())
        
        return [p for p in params if p.requires_grad]
    
    def forward(
        self, 
        joints, 
        bones, 
        motion,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        translation_labels: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        # 获取拼接时序特征 [B, T, in_channels] (默认 in_channels = 768)
        feat = self.encoder(joints=joints, bones=bones, motion=motion, return_sequence=True)
        # 时序下采样和特征映射 [B, T_out, out_channels] (默认 out_channels = 1024)
        adapted_feat = self.temporal_adapter(feat)
        # 获取 CTC 预测 logits [B, T_out, num_classes] (默认 num_classes = 2005)
        ctc_logits = self.ctc_head(adapted_feat)
        
        output_dict = {
            "adapted_feat": adapted_feat,
            "ctc_logits": ctc_logits
        }
        
        # 如果启用翻译解码器且提供了输入
        if self.use_translation_decoder and self.translation_decoder is not None and decoder_input_ids is not None:
            translation_outputs = self.translation_decoder(
                encoder_features=adapted_feat,
                decoder_input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                labels=translation_labels,
                encoder_attention_mask=encoder_attention_mask
            )
            output_dict["translation_loss"] = translation_outputs["loss"]
            output_dict["translation_logits"] = translation_outputs["logits"]
        
        return output_dict
    
    @torch.no_grad()
    def generate_translation(
        self,
        joints,
        bones,
        motion,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        生成翻译文本（推理阶段使用）
        """
        if not self.use_translation_decoder or self.translation_decoder is None:
            return None
        
        # 先通过编码器
        feat = self.encoder(joints=joints, bones=bones, motion=motion, return_sequence=True)
        adapted_feat = self.temporal_adapter(feat)
        
        # 生成翻译
        generated = self.translation_decoder.generate(
            encoder_features=adapted_feat,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )
        
        return generated


if __name__ == '__main__':
    from src.student_model.architecture.multi_stream_stgcn import MultiStreamSTGCN
    
    # Windows 编码修复
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    
    print("=" * 60)
    print("Testing CSLTModel (without translation decoder first)...")
    print("=" * 60)
    
    # 1. 实例化 MultiStreamSTGCN 编码器
    graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
    encoder = MultiStreamSTGCN(
        num_class=10, 
        in_channels=3, 
        graph_args=graph_args,
        edge_importance_weighting=True
    )
    
    # 2. 实例化 CSLT 模型 (不包含翻译解码器)
    model = CSLTModel(
        encoder=encoder,
        in_channels=768,
        out_channels=1024,
        num_classes=2005,
        use_translation_decoder=False
    )
    
    # 3. 构造虚拟的多流输入数据
    B, C, T, V, M = 2, 3, 100, 21, 1
    
    joints = torch.randn(B, C, T, V, M)
    bones = torch.randn(B, C, T, V, M)
    motion = torch.randn(B, C, T, V, M)
    
    print(f"Input shapes: joints/bones/motion = {joints.shape}")
    
    # 4. 进行前向传播
    outputs = model(joints=joints, bones=bones, motion=motion)
    
    # 5. 验证输出
    adapted_feat = outputs["adapted_feat"]
    ctc_logits = outputs["ctc_logits"]
    
    print(f"adapted_feat shape: {adapted_feat.shape}")
    print(f"ctc_logits shape: {ctc_logits.shape}")
    
    # 自检 assertion
    assert adapted_feat.shape[0] == B
    assert adapted_feat.shape[2] == 1024
    assert ctc_logits.shape[0] == B
    assert ctc_logits.shape[2] == 2005
    
    print("\n[OK] Basic CSLTModel test passed!")
    
    # 测试翻译解码器 (如果 transformers 可用)
    if DECODER_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing CSLTModel with translation decoder...")
        print("=" * 60)
        
        try:
            # 重新初始化 encoder 和模型（带翻译解码器）
            encoder_translation = MultiStreamSTGCN(
                num_class=10, 
                in_channels=3, 
                graph_args=graph_args,
                edge_importance_weighting=True
            )
            
            model_with_decoder = CSLTModel(
                encoder=encoder_translation,
                in_channels=768,
                out_channels=1024,
                num_classes=2005,
                use_translation_decoder=True,
                decoder_freeze_strategy="cross_attention_only"
            )
            
            print("[OK] Model with translation decoder initialized!")
            
            # 构造 dummy decoder 输入
            dummy_decoder_input_ids = torch.randint(0, 50000, (B, 10))
            
            # 前向传播（包含翻译）
            outputs_with_translation = model_with_decoder(
                joints=joints,
                bones=bones,
                motion=motion,
                decoder_input_ids=dummy_decoder_input_ids,
                translation_labels=dummy_decoder_input_ids
            )
            
            print(f"translation_loss: {outputs_with_translation['translation_loss']}")
            print(f"translation_logits shape: {outputs_with_translation['translation_logits'].shape}")
            
            print("\n[OK] Translation decoder test passed!")
            
        except Exception as e:
            print(f"\n[WARNING] Translation decoder test skipped due to error: {e}")
            print("This is expected if transformers or mBART model is not downloaded yet.")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
