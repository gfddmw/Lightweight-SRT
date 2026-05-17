# CSL-Daily (SMKD 教师方案) 前期准备设计文档

## 1. 目标概述
本方案旨在为 CSL-Daily 数据集建立基于 SMKD (Selective Multi-Teacher Knowledge Distillation) 的蒸馏前期准备流程。教师模型将提供高质量的视觉语义特征和时序 Logits，引导轻量级 ST-GCN 学生模型学习连续手语翻译能力。

## 2. 准备阶段详解

### A. 数据集标准化 (Data Engineering)
- **原始视频**: 整理 CSL-Daily 视频库，确保路径与官方 JSON 标注映射一致。
- **标注处理**: 
    - 解析 `gloss` 序列，建立 `gloss_to_id` 词典。
    - 处理中文句子，使用 `jieba` 进行预分词，建立 `word_to_id` 词典（包含 `<PAD>, <BOS>, <EOS>, <UNK>`）。
- **划分验证**: 提取官方 `train/dev/test` 列表，生成项目专用的 `dataset_splits.json`。

### B. 教师特征提取 (SMKD Teacher Features)
- **模型获取**: 部署预训练的 SMKD 教师模型（通常包含视觉编码器和时序增强模块）。
- **特征提取脚本**: 编写 `scripts/feature_extraction/extract_smkd_teacher.py`。
- **输出格式**:
    - **Visual Features**: `[T_teacher, 1024]` (或对应维度)，保存为 `.npy`。
    - **Temporal Logits**: `[T_teacher, gloss_vocab_size]`，用于 CTC 蒸馏。
    - **Selection Masks (可选)**: 如果使用 SMKD 的选择性权重，也需一并保存。

### C. 学生骨骼点提取 (Skeleton Stream)
- **多流生成**:
    - `Joint`: MediaPipe 提取的 21 个手部关键点。
    - `Bone`: 关键点之间的向量。
    - `Motion`: 相邻帧之间的位移。
- **降采样与对齐**: 骨骼点帧率需与视频帧率及教师特征的时序步长对齐。

### D. 对齐验证 (Consistency Check)
- **时序对齐**: 验证 `T_skeleton` 与 `T_teacher_feature` 的对应关系。
- **ID 完整性**: 确保每个 `video_id` 都拥有对应的视频、骨骼点、教师特征和文本标注。

## 3. 存储结构建议
```text
processed/csl_daily/
├── skeletons/          # [B, T, V, C]
├── teacher_features/   # [B, T, D]
├── teacher_logits/     # [B, T, Vocab]
└── annotations/
    ├── gloss_vocab.json
    ├── text_vocab.json
    └── train_info.json
```

## 4. 关键接口契约 (Interface Contract)
准备工作完成后，DataLoader 必须能够返回：
- `joints/bones/motion`: 学生输入。
- `teacher_features`: 用于 Hint Loss。
- `teacher_logits`: 用于 CTC Distillation Loss。
- `gloss_ids / text_ids`: 用于监督学习。
