# CSL-Daily (CorrNet 方案) 前期准备实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 CSL-Daily 数据集基于 CorrNet 强教师方案的前期数据准备、特征提取与词表构建，为后续 CSLT 训练打下基础。

**Architecture:** 数据解耦并行流水线 (A-骨骼, B-教师特征, C-词表索引, D-整合验证)。

**Tech Stack:** MediaPipe, CorrNet (Teacher), Jieba/SentencePiece, PyTorch.

---

## Task A: 视频预处理与骨架工程 (A同学)

**Files:**
- Create: `scripts/data_prep/prepare_csl_daily_videos.py`
- Create: `scripts/feature_extraction/extract_csl_daily_skeletons.py`

- [ ] **Step 1: 整理视频路径与 ID 映射**
编写脚本扫描 `data/CSL-Daily/videos/`，生成 `vid -> video_path` 的初步映射，并验证视频文件完整性。

- [ ] **Step 2: 编写多流骨架提取脚本**
使用 MediaPipe Hands 提取 21 个手部关键点。实现 `Joint` (原始坐标), `Bone` (骨骼向量), `Motion` (相邻帧位移) 的计算逻辑。
```python
# 示例：Joint to Bone
bone = joint[:, :, 1:, :] - joint[:, :, neighbor_link, :]
```

- [ ] **Step 3: 批量提取并存储为 .npy**
将特征保存至 `processed/csl_daily/skeletons/{vid}.npy`，格式建议为 `[T, V, C]`。

---

## Task B: CorrNet 教师模型特征导出 (B同学)

**Files:**
- Clone: `src/teacher_model/CorrNet` (克隆自 https://github.com/hulianyuyy/CorrNet)

- [ ] **Step 1: CorrNet 教师模型环境部署与权重加载**
克隆并部署 CorrNet 官方仓库环境，确保能加载在 CSL-Daily 上训练好的 SOTA 权重。

- [ ] **Step 2: 使用官方内置工具导出特征与 Logits**
运行 CorrNet 官方的 `main.py --phase features` 前向传播获取中间层特征 (1024D) 和最终分类 Logits (2001D)。
```bash
python main.py --load-weights ../../weights/teacher/smkd_csl_daily_real.pt --phase features --config configs/csl_daily.yaml --device 0
```

- [ ] **Step 3: 时序对齐处理**
若教师模型进行了时序降采样，需在文件名或元数据中记录降采样率，以便学生模型对齐。

---

## Task C: 语言系统与数据索引 (C同学)

**Files:**
- Create: `scripts/data_prep/build_csl_daily_vocab.py`
- Create: `scripts/data_prep/generate_csl_daily_splits.json`

- [ ] **Step 1: 构建 Gloss 与 中文词表**
解析 CSL-Daily 标注文件，统计所有 Gloss 出现频率并构建词表。使用 `jieba` 对中文句子进行分词，构建包含特殊 Token (`<PAD>`, `<BOS>`, `<EOS>`) 的文本词表。

- [ ] **Step 2: 建立全局数据索引**
生成一个包含所有模态路径的 JSON 文件：
```json
{
  "vid123": {
    "skeleton": "processed/csl_daily/skeletons/vid123.npy",
    "teacher_feat": "processed/csl_daily/teacher_features/vid123.npy",
    "gloss": "手语 谢谢 你",
    "sentence": "谢谢你",
    "subset": "train"
  }
}
```

---

## Task D: 对齐验证与接口整合 (D同学)

**Files:**
- Create: `scripts/data_prep/verify_csl_daily_alignment.py`
- Create: `src/common/datasets/cslt_dataset.py`

- [ ] **Step 1: 编写时序与 ID 对齐校验脚本**
检查每个 ID 的骨骼点帧数与教师特征步长是否匹配。
```python
# 校验逻辑
assert T_skeleton == T_teacher * downsample_rate
```

- [ ] **Step 2: 实现 CSLTDataset 与 collate_fn**
整合 A、B、C 同学的产出，实现支持多流骨架、教师特征、教师 Logits 及变长 Padding 的 DataLoader。

- [ ] **Step 3: 开发可视化预览工具**
编写一个简单脚本，随机抽样一个样本，同时显示视频帧、骨架连线图、对应的 Gloss 和翻译结果，确保逻辑正确。

---
