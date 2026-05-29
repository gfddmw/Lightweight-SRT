# 连续手语翻译 (CSLT) 后续开发计划

## 1. 目标定位

后续开发目标是从当前的 **WLASL2000 词级识别系统**，扩展到 **连续手语视频到自然语言句子** 的翻译系统。

当前项目已经完成的多流 ST-GCN 蒸馏模型不直接作为最终翻译模型使用，而是作为 **骨骼时序视觉编码器初始化**：

```text
现有模型：
WLASL 单词骨骼序列 -> MultiStream ST-GCN -> FC(2000 类)

后续模型：
连续手语骨骼序列 -> MultiStream ST-GCN Encoder -> 时序特征
                                      -> CTC Head -> gloss 序列
                                      -> Decoder -> 中文句子
```

本阶段不再优先追求 WLASL Top-1，而是建立连续手语翻译链路，核心指标改为：

- CSLR: `WER`
- SLT: `BLEU-1/2/3/4`, `ROUGE-L`
- 工程指标: 模型大小、端侧延迟、内存占用

---

## 2. 数据集选择

### 2.1 主数据集：CSL-Daily

后续主线建议使用 **CSL-Daily**，原因如下：

- 面向中国手语，输出为中文句子，符合项目后续应用方向；
- 提供连续手语视频、gloss 序列和中文文本三元标注；
- gloss 规模约 2000，和当前 WLASL2000 的词表规模有一定可比性；
- 适合先做 `video -> gloss`，再做 `gloss/video -> text`。

### 2.2 参考数据集：PHOENIX14T

PHOENIX14T 可作为国际 benchmark 参考，但不作为主开发数据集：

- 优点：连续手语翻译领域经典数据集，论文对比多；
- 缺点：德国手语和德语文本，领域集中在天气预报，和中文应用目标不一致。

### 2.3 不建议继续用 WLASL 做连续翻译

WLASL 是孤立词识别数据集，缺少连续句子视频和自然语言翻译标注，因此只能用于：

- 词级识别预训练；
- 骨骼编码器初始化；
- 轻量化部署实验。

---

## 3. 技术路线

### V0：CSL-Daily 数据管线

目标：先把连续手语数据稳定转成可训练样本。

输入：

```text
CSL-Daily video / gloss / text
```

输出：

```text
joints / bones / motion
gloss_ids / gloss_lengths
text_ids / text_lengths
input_lengths
```

关键任务：

- 下载并整理 CSL-Daily 官方划分；
- 提取手部骨骼点，优先复用当前 `joints / bones / motion` 多流格式；
- 构建 gloss vocabulary；
- 构建中文 tokenizer，包含 `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`；
- 实现变长序列 padding、mask 和长度统计；
- 输出样本可视化脚本，检查骨骼点、gloss 和文本是否对齐。

建议交付：

```text
src/common/datasets/cslt_dataset.py
scripts/data_prep/prepare_csl_daily.py
configs/cslt/csl_daily.yaml
```

### V1：连续手语识别 CSLR

目标：先跑通 `video -> gloss`，不要一开始直接端到端翻译。

模型结构：

```text
MultiStream ST-GCN Encoder -> Temporal Adapter -> CTC Head
```

训练目标：

```text
CTC Loss
```

评价指标：

```text
WER
```

迁移策略：

- 加载当前 WLASL 蒸馏模型权重；
- 丢弃 `FC / classifier / fcn` 分类头；
- 保留多流 ST-GCN backbone；
- 新增 `Temporal Adapter` 和 `CTC Head`；
- 第一阶段冻结 encoder，只训练新 head；
- 第二阶段解冻后几层，低学习率微调；
- 第三阶段全模型训练。

注意：当前最佳 WLASL 多流模型来自标准 ST-GCN 三流结构，而不是 Shift-GCN + BottleNeck。因此 V1 优先复用 **当前最佳标准 MultiStream ST-GCN encoder**。Shift-GCN 和 BottleNeck 作为后续轻量化对照实验，不作为第一版依赖。

### V2：连续手语翻译 SLT

目标：在 V1 稳定后，再加入中文句子生成能力。

推荐先做两阶段：

```text
video -> gloss -> Chinese text
```

随后再尝试联合训练：

```text
video features -> CTC gloss branch
               -> Transformer Decoder -> Chinese text
```

联合损失：

```text
Loss = CTC_Loss + lambda * Translation_Loss
```

初始建议：

```text
lambda = 0.5 或 1.0
```

翻译模块可选：

- 轻量 Transformer Decoder；
- GRU Decoder + Attention；
- gloss-to-text Transformer 作为独立 baseline。

评价指标：

- `BLEU-1/2/3/4`
- `ROUGE-L`
- 翻译样例可读性分析

### V3：端侧部署评估

目标：确认连续翻译模型是否还能保持轻量化优势。

关键任务：

- 导出 TorchScript / PyTorch Lite；
- 评估 Android 端延迟和内存；
- 对 encoder 做 INT8 量化；
- 对 decoder 做尺寸控制；
- 比较 `hand-only` 与 `hands + pose + face` 两种输入方案。

---

## 4. 模型接口定义

### 4.1 Dataset 输出

建议每个 batch 输出：

```python
{
    "features": Tensor[B, C, T, V, M],
    "input_lengths": Tensor[B],
    "gloss_ids": Tensor[B, G],
    "gloss_lengths": Tensor[B],
    "text_ids": Tensor[B, L],
    "text_lengths": Tensor[B],
}
```

如果使用多流输入，也可以输出：

```python
{
    "joints": Tensor[B, C, T, V, M],
    "bones": Tensor[B, C, T, V, M],
    "motion": Tensor[B, C, T, V, M],
    ...
}
```

### 4.2 Encoder 输出

连续任务不能过早做全局池化。Encoder 应支持返回时序特征：

```python
features = encoder(x, return_sequence=True)
# features: Tensor[B, T', D]
```

### 4.3 CTC Head

```python
logits = ctc_head(features)
# logits: Tensor[B, T', gloss_vocab_size + 1]
```

其中 `+1` 是 CTC blank 类。

### 4.4 Translation Decoder

```python
text_logits = decoder(
    memory=features,
    tgt_tokens=text_ids[:, :-1],
    src_key_padding_mask=src_mask,
    tgt_key_padding_mask=tgt_mask,
)
```

---

## 5. 四人分工方案

### A 同学：CSL-Daily 数据工程

职责：

- 整理 CSL-Daily 官方划分；
- 实现骨骼点提取与多流特征生成；
- 构建 gloss/text vocabulary；
- 实现 `CSLTDataset` 和 `collate_fn`；
- 提供样本可视化和数据统计报告。

交付物：

```text
src/common/datasets/cslt_dataset.py
scripts/data_prep/prepare_csl_daily.py
documents/CSL_Daily_Data_Report.md
```

### B 同学：连续模型结构

职责：

- 将 MultiStream ST-GCN 改造成可返回时序特征的 encoder；
- 实现 `Temporal Adapter`；
- 实现 `CTC Head`；
- 实现轻量 Transformer/GRU Decoder；
- 提供模型前向 shape 测试。

交付物：

```text
src/student_model/architecture/cslt_model.py
src/student_model/architecture/translation_decoder.py
```

### C 同学：训练与迁移

职责：

- 实现 WLASL 蒸馏权重到 CSL-Daily encoder 的迁移加载；
- 实现 CTC 训练；
- 实现 gloss-to-text 和联合训练；
- 维护训练配置和实验日志；
- 输出 WER / BLEU / ROUGE 对比。

交付物：

```text
src/student_model/train_cslt.py
configs/cslt/csl_daily_ctc.yaml
configs/cslt/csl_daily_slt.yaml
documents/CSLT_Training_Report.md
```

### D 同学：推理、评估与部署

职责：

- 实现 CTC greedy / beam search 解码；
- 实现文本 beam search；
- 集成 BLEU、ROUGE-L、WER；
- 编写端到端 demo；
- 做 TorchScript Lite 和 Android 端可行性评估。

交付物：

```text
src/student_model/inference/cslt_translator.py
src/student_model/evaluation/cslt_metrics.py
documents/CSLT_Deployment_Report.md
```

---

## 6. 里程碑计划

### W1：数据与词表

- 完成 CSL-Daily 数据目录整理；
- 完成 gloss vocabulary；
- 完成中文 text tokenizer；
- 跑通单样本骨骼点提取；
- 输出 20 个样本的可视化检查结果。

验收标准：

```text
Dataset 能返回 features / gloss_ids / text_ids / lengths
随机样本可视化与标注一致
```

### W2：CSLR baseline

- 实现 `CSLTDataset` 和 `collate_fn`；
- 实现 MultiStream ST-GCN `return_sequence=True`；
- 实现 CTC Head；
- 在小批量数据上跑通前向、反向和 CTC Loss。

验收标准：

```text
训练脚本能在 debug subset 上 loss 下降
输出 WER 计算结果
```

### W3-W4：迁移训练

- 加载 WLASL 蒸馏 encoder 权重；
- 冻结 encoder 训练 CTC Head；
- 解冻后几层低学习率微调；
- 输出第一版 CSLR WER。

验收标准：

```text
迁移模型优于随机初始化 baseline
训练日志完整可复现
```

### W5-W6：翻译模型

- 实现 gloss-to-text baseline；
- 接入 Transformer/GRU Decoder；
- 训练 `video -> text` 或 `features -> text`；
- 输出 BLEU / ROUGE-L。

验收标准：

```text
能生成中文句子
BLEU/ROUGE 指标可复现
提供若干预测样例
```

### W7：部署评估

- 导出 encoder 或完整模型；
- 尝试 INT8 量化；
- 评估 Android 端输入缓存、推理延迟、内存占用；
- 给出是否适合实时端侧部署的结论。

---

## 7. 风险与注意事项

1. **跨语言迁移收益有限**  
   WLASL 是 ASL，CSL-Daily 是中国手语。现有蒸馏模型能迁移骨骼运动建模能力，但不能迁移具体词义和标签语义。

2. **hand-only 输入可能不足**  
   连续翻译比词级识别更依赖上半身姿态、面部表情和语境。第一版可复用 hand-only 管线，第二版应评估 `hands + pose + face`。

3. **不要一开始做纯端到端翻译**  
   直接 `video -> text` 难以定位错误。应先做 `video -> gloss`，用 WER 确认视觉对齐能力。

4. **Shift-GCN / BottleNeck 暂不作为第一版依赖**  
   当前最佳 WLASL 实验显示标准三流 ST-GCN 更稳定。轻量化结构应在 CSL-Daily baseline 稳定后再加入。

5. **端侧部署需要重新评估**  
   连续翻译模型含 decoder，延迟和内存会高于词级分类模型。不能直接沿用 WLASL 识别阶段的端侧性能结论。

---

## 8. 当前优先级

最高优先级不是直接实现完整翻译，而是：

```text
CSL-Daily 数据管线 -> 预训练 MultiStream ST-GCN Encoder -> CTC gloss baseline
```

只有当 gloss 识别稳定后，再进入中文句子翻译和端侧部署优化。
