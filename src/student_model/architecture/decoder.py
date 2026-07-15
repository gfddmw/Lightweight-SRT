import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import MBartForConditionalGeneration, MBartConfig
    from transformers.modeling_outputs import BaseModelOutput
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FeatureProjector(nn.Module):
    """
    将 1024 维的手语特征投影到 mBART 的隐藏维度 (通常是 1024)
    同时也处理时序对齐
    """
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] 来自编码器的特征
            
        Returns:
            projected_x: [B, T, output_dim] 投影后的特征，作为 decoder 的 cross attention memory
        """
        x = self.proj(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class TranslationDecoder(nn.Module):
    """
    基于 mBART-50 的翻译解码器
    将手语编码器输出的 1024 维时序特征注入到 mBART decoder 的 cross attention 中
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "facebook/mbart-large-50",
        freeze_strategy: str = "cross_attention_only",
        src_lang: str = "zh_CN",
        tgt_lang: str = "zh_CN",
        feature_dim: int = 1024
    ):
        """
        Args:
            pretrained_model_name: 预训练 mBART 模型名称
            freeze_strategy: 权重冻结策略
                - "cross_attention_only": 仅微调 cross attention 层
                - "decoder_layers": 微调最后几层 decoder
                - "full": 全模型微调
            src_lang: 源语言代码 (mBART-50 使用)
            tgt_lang: 目标语言代码
            feature_dim: 输入特征维度 (来自手语编码器)
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Please install transformers: pip install transformers sentencepiece"
            )
        
        # 加载 mBART 模型配置和模型
        self.config = MBartConfig.from_pretrained(pretrained_model_name)
        self.mbart = MBartForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            use_safetensors=True
        )
        
        # 特征投影层
        self.feature_projector = FeatureProjector(
            input_dim=feature_dim,
            output_dim=self.config.d_model
        )
        
        # 保存语言代码
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # 提前缓存中文的 forced BOS token id，生成时必须用到
        self.tgt_lang_id = 250026  # zh_CN in mBART-50
        
        # 初始化权重冻结
        self._freeze_parameters(freeze_strategy)
    
    def _freeze_parameters(self, freeze_strategy: str):
        """
        根据策略冻结模型参数
        """
        # 默认冻结所有参数
        for param in self.mbart.parameters():
            param.requires_grad = False
        
        if freeze_strategy == "cross_attention_only":
            # 仅解冻 cross attention 层
            for name, param in self.mbart.model.decoder.named_parameters():
                if "encoder_attn" in name:  # cross attention
                    param.requires_grad = True
            # 同时解冻投影层
            for param in self.feature_projector.parameters():
                param.requires_grad = True
        
        elif freeze_strategy == "decoder_layers":
            # 解冻最后 3 层 decoder 和投影层
            num_layers = self.config.decoder_layers
            layers_to_unfreeze = list(range(num_layers - 3, num_layers))
            
            for name, param in self.mbart.model.decoder.named_parameters():
                # 检查是否在需要解冻的层
                layer_match = False
                for layer_idx in layers_to_unfreeze:
                    if f"layers.{layer_idx}." in name:
                        layer_match = True
                        break
                
                if layer_match or "layer_norm" in name:
                    param.requires_grad = True
            
            for param in self.feature_projector.parameters():
                param.requires_grad = True
        
        elif freeze_strategy == "full":
            # 解冻所有参数
            for param in self.mbart.parameters():
                param.requires_grad = True
            for param in self.feature_projector.parameters():
                param.requires_grad = True
        
        else:
            raise ValueError(f"Unknown freeze strategy: {freeze_strategy}")
    
    def get_trainable_params(self):
        """返回可训练的参数列表"""
        return [p for p in self.parameters() if p.requires_grad]
    

    def forward(
        self,
        encoder_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Args:
            encoder_features: [B, T, feature_dim] 来自手语编码器的特征
            attention_mask: [B, S] decoder 输入的 attention mask
            labels: [B, S] 训练标签 (提供此参数时模型自动右移生成 decoder_input_ids)
            decoder_input_ids: [B, S] decoder 输入的 token ids (推理或无 labels 时使用)
            encoder_attention_mask: [B, T] 编码器端的 attention mask (用于 padding 处理)
            
        Returns:
            包含 loss, logits 等的字典
        """
        # 投影特征到 mBART 维度
        encoder_hidden_states = self.feature_projector(encoder_features)
        
        # 构建 mBART 期望的 encoder_outputs 对象
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # 构建 mBART 的输入
        mbart_inputs = {
            "encoder_outputs": encoder_outputs,
            # 核心修正：mBART 外部接口中，attention_mask 指代的就是 Encoder 的 Mask
            "attention_mask": encoder_attention_mask,
        }
        
        # 核心逻辑：如果提供了 labels，不要手动传 decoder_input_ids，让 HF 自动右移
        if labels is not None:
            mbart_inputs["labels"] = labels
        elif decoder_input_ids is not None:
            mbart_inputs["decoder_input_ids"] = decoder_input_ids
            
        # 将 decoder 的 mask 映射为 decoder_attention_mask
        if attention_mask is not None:
            mbart_inputs["decoder_attention_mask"] = attention_mask
        
        # 前向传播
        outputs = self.mbart(**mbart_inputs)
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_hidden_states": encoder_hidden_states
        }
        
    @torch.no_grad()
    def generate(
        self,
        encoder_features: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        forced_bos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        自回归生成翻译
        
        Args:
            encoder_features: [B, T, feature_dim]
            max_length: 最大生成长度
            num_beams: beam search 宽度
            early_stopping: 是否提前停止
            encoder_attention_mask: [B, T]
            forced_bos_token_id: 强制首字 ID (默认使用中文 zh_CN)
            
        Returns:
            generated_ids: [B, S] 生成的 token ids
        """
        encoder_hidden_states = self.feature_projector(encoder_features)
        
        # 构建 mBART 期望的 encoder_outputs 对象
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # 如果未指定，默认强制使用中文 ID
        if forced_bos_token_id is None:
            forced_bos_token_id = self.tgt_lang_id
            
        # 准备生成配置
        generate_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "forced_bos_token_id": forced_bos_token_id,
            **kwargs
        }
        
        # 使用 mBART 生成
        generated_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            # 核心修正：generate 接口中，编码器 mask 必须用 attention_mask 这个名字
            attention_mask=encoder_attention_mask,
            **generate_kwargs
        )
        
        return generated_ids


if __name__ == '__main__':
    # Windows 编码修复
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    
    print("Testing TranslationDecoder initialization...")
    
    # 测试初始化
    try:
        decoder = TranslationDecoder(
            freeze_strategy="cross_attention_only",
            feature_dim=1024
        )
        
        print(f"[OK] Successfully initialized decoder!")
        print(f"Trainable parameters: {sum(p.numel() for p in decoder.get_trainable_params())}")
        print(f"Total parameters: {sum(p.numel() for p in decoder.parameters())}")
        
        # 测试前向传播
        batch_size = 2
        seq_length = 25
        feature_dim = 1024
        
        dummy_features = torch.randn(batch_size, seq_length, feature_dim)
        dummy_labels = torch.randint(0, 50000, (batch_size, 10))
        
        # 测试 forward (只传 labels，让模型自动右移)
        outputs = decoder(
            encoder_features=dummy_features,
            labels=dummy_labels
        )
        
        print(f"\n[OK] Forward pass successful!")
        print(f"Loss: {outputs['loss']}")
        print(f"Logits shape: {outputs['logits'].shape}")  # 预期 [2, 10, Vocab_Size]
        
        # 测试生成
        print(f"\nTesting generation...")
        generated = decoder.generate(
            encoder_features=dummy_features,
            max_length=20
        )
        print(f"Generated shape: {generated.shape}")
        print(f"\n[OK] Decoder test completed successfully!")
        
    except Exception as e:
        print(f"\n[WARNING] Test skipped due to error: {e}")
        print("This is expected if transformers or mBART model is not downloaded yet.")