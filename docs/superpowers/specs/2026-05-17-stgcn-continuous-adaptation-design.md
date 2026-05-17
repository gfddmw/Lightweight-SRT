# ST-GCN 连续化改造设计文档 (草案)

## 1. 目标
将现有的基于孤立词分类的 ST-GCN 模型重构为支持连续手语序列建模的 **CSLT Encoder**。通过保留时序维度、维度对齐和预测头更换，使其能够利用 CSL-Daily 数据集和 SMKD 教师特征进行蒸馏训练。

## 2. 核心架构变更

### A. 骨干网络 (Backbone) 改造
- **文件**: `src/student_model/architecture/st_gcn.py`
- **变更**:
    - 在 `Model.forward()` 中增加 `return_sequence=True` 标志。
    - 当标志为 `True` 时，跳过 `F.avg_pool2d(x, x.size()[2:])`。
    - 移除或绕过原本用于分类的 `self.fcn`。
- **输出**: 从 `[B, C, T, V, M]` 转换为 `[B, D, T]` 的特征图（其中 D 是最后一层卷积的通道数，T 是时序步长）。

### B. 时序适配器 (Temporal Adapter) 引入
- **目的**: 
    1. **维度对齐**: 将 ST-GCN 的通道数（如 256）映射到 SMKD 教师特征的维度（如 1024）。
    2. **时序压缩**: 若手语视频帧数过多，通过 1D 卷积（Stride > 1）进行时序降采样，提高计算效率。
- **结构**:
    - `nn.Conv1d` (kernel=3, stride=2, padding=1)
    - `nn.BatchNorm1d`
    - `nn.ReLU`
- **位置**: 紧跟在 ST-GCN Backbone 输出之后。

### C. 预测头 (CTC Head) 实现
- **目的**: 实现从时序特征到 Gloss 序列的概率映射。
- **结构**: 简单的线性层 `nn.Linear(Adapter_Dim, Gloss_Vocab_Size + 1)`。
- **损失函数**: 使用 `nn.CTCLoss`。

## 3. 多流融合策略 (Multi-Stream Integration)
- **方案**: **Feature-level Fusion**。
- **流程**:
    1. 分别通过三个 ST-GCN 分支提取 Joint、Bone、Motion 的特征图。
    2. 在通道维度（Channel Dim）进行拼接（Concat），形成 `[B, D*3, T]`。
    3. 送入统一的 `Temporal Adapter` 降维并对齐到教师空间。

## 4. 迁移学习与初始化 (Pre-training & Initialization)
- **权重复用**: 
    - 所有的 `st_gcn_networks`（Backbone 部分）加载 WLASL2000 蒸馏出的最佳权重。
    - 新增的 `Temporal Adapter` 和 `CTC Head` 使用随机初始化。
- **冻结策略**: 训练初期建议冻结 Backbone，仅训练新添加的 Head 和 Adapter，待 Loss 稳定后再全模型微调（Fine-tune）。

## 5. 接口契约更新
重构后的 `CSLT_STGCN` 应支持以下调用方式：
```python
# 训练阶段：输出特征用于蒸馏，输出 Logits 用于 CTC
logits, features = model(joints, bones, motion, return_features=True)

# 推理阶段：仅输出 Logits
logits = model(joints, bones, motion)
```

## 6. 成功标准 (Success Criteria)
1.  **Shape 验证**: 模型输出的 T 轴长度应与教师特征 T 轴长度一致。
2.  **收敛验证**: 在 CSL-Daily 子集上，CTC Loss 能够正常下降。
3.  **对齐验证**: 提取的学生特征经过 Adapter 后，与教师特征的 MSE 距离应逐渐减小。
