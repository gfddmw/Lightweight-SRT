# Lightweight-SRT 蒸馏训练与调度实验报告

## 1. 训练目标与配置策略
本阶段聚焦“轻量化学生模型在 WLASL2000 上的知识蒸馏训练”，核心策略是将传统 Logits KD 升级为“Logits + 多层特征 Hint”的联合蒸馏。实验主配置基于 `configs/student/st_gcn/wlasl2000/train_kd_hint_new.yaml` 与 `configs/student/st_gcn/wlasl2000/train_kd_hint_optuna_new.yaml`：
- **KD Alpha 初值**: 0.5
- **KD Temperature**: 4.0（Optuna 搜索范围 2~10）
- **Hint Weight**: 0.1
- **Hint Layer Weights**: [1.0, 0.5]
- **Weight Decay**: 1e-4（Optuna 搜索范围 1e-5~1e-3）
- **快速验证策略**: WLASL100 子集 + 10 Epoch

## 2. 蒸馏算法实现与核心技术手段

### 2.1 多层特征蒸馏（Hint Loss）
为解决教师（RGB-I3D）与学生（Skeleton-STGCN）特征空间不一致问题，已在 `src/student_model/distillation/recognition_kd_hint_new.py` 中实现如下机制：
- **1x1 适配器映射**：`NewFeatureAdapter` 使用 `nn.Conv1d(..., kernel_size=1)`，将学生中间层特征映射到教师维度空间。
- **多层 Hint 对齐**：学生侧抽取两级中间特征（全局池化分支 + 时序池化分支），并通过 `hint_layer_weights` 进行加权融合。
- **监督损失**：适配后学生特征与教师特征采用 `MSELoss` 计算 Hint Loss。

### 2.2 动态蒸馏权重调度
已实现按训练进度衰减的蒸馏权重机制：

`alpha = alpha_initial * (1 - epoch / total_epochs)`

对应总损失为：

`loss = (1 - alpha) * CE + alpha * KD + hint_weight * Hint`

该策略保证训练前期更依赖教师软标签，后期逐步回归硬标签以细化最终判别边界。

### 2.3 自动超参搜索（Optuna）
已新增 `scripts/train/optuna_search_kd_hint_new.py`，并完成以下实现：
- **搜索参数**：`kd_temperature` 与 `weight_decay`；
- **目标函数**：调用 KD-Hint 训练脚本并提取 Top1 作为 trial 评分；
- **快速筛选机制**：默认 WLASL100 + 10 Epoch；
- **Windows 兼容**：加入 OpenMP 冲突规避环境变量，避免 `libiomp5md.dll` 重复加载。

## 3. 运行验证与当前结果

### 3.1 环境状态
- **GPU 可见性**：`nvidia-smi` 可识别 RTX 4060（硬件/驱动正常）。
- **PyTorch 后端**：当前环境为 `torch 2.11.0+cpu`，`torch.cuda.is_available() = False`。

结论：硬件具备 CUDA 条件，但当前 Python 环境仍是 CPU 版 PyTorch。

### 3.2 数据覆盖检查结果
执行：

`python scripts/report/kd_data_report_new.py --split_json ./data/nslt_2000.json --skeleton_dir ./processed/skeletons --logits_dir ./processed/logits --teacher_feature_dir ./processed/teacher_features`

结果：
- **train**: final usable = 0 / 14296
- **test**: final usable = 0 / 2879
- **val**: final usable = 0 / 3920

说明：`processed/skeletons`、`processed/logits`、`processed/teacher_features` 当前均为空，训练样本为 0。

### 3.3 Optuna 试运行结果
执行：

`python scripts/train/optuna_search_kd_hint_new.py --trials 1 --fast_epochs 1`

当前报错：

`ValueError: num_samples should be a positive integer value, but got num_samples=0`

结论：失败根因是数据未就绪，不是蒸馏算法实现错误。

## 4. 已完成修正项与后续执行清单

### 4.1 已完成修正
- 已修正配置中的 `split_json` 路径为 `./data/nslt_2000.json`：
  - `configs/student/st_gcn/wlasl2000/train_kd_hint_new.yaml`
  - `configs/student/st_gcn/wlasl2000/train_kd_hint_optuna_new.yaml`
- 已在 Optuna 训练子进程中加入 OpenMP 兼容设置，避免 Windows 下重复加载冲突。

### 4.2 后续可直接执行流程
补齐以下数据后即可启动完整训练：
- `processed/skeletons/*.npy`
- `processed/logits/*.npy`
- `processed/teacher_features/*.npy`

推荐顺序：
1. 生成骨架样本（学生输入）。
2. 提取教师 logits 与中间特征。
3. 运行 Optuna 快速筛选超参。
4. 使用最优超参执行完整 WLASL2000 训练。

## 5. 复现命令

### 5.1 快速搜索（WLASL100，10 Epoch）
`python scripts/train/optuna_search_kd_hint_new.py --trials 20 --fast_epochs 10`

## 6. 多流蒸馏性能专题调研与突破 (2026-05-01)

### 6.1 问题现象
在 WLASL2000 全量训练中，多流模型（Joint+Bone+Motion）初始性能极差，在 50-80 Epoch 期间的 Top1 准确率仅为 **19%**，远低于单流 Fast-KD 模型（**28.24%**）。

### 6.2 根因分析
1.  **标签对齐失效 (致命)**：多流训练曾错误使用基于 Gloss 字母排序的 `label_map.json`，而教师模型与单流模型使用的是基于 ID 的 `nslt_2000.json`。这导致学生模型接收到了错误的蒸馏信号。
2.  **空间信息丢失**：原配置开启了 `normalize_wrist: true`，丢失了手部在空间中的全局位置。而教师模型（I3D）是基于原始视频训练的，保留位置信息对对齐教师逻辑至关重要。
3.  **架构与策略失配**：过早引入 `Shift-GCN` 和 `Bottleneck` 等复杂结构，在纯 Logits 蒸馏模式下反而增加了收敛难度。

### 6.3 优化措施 (aligned_v2 方案)
1.  **统一标签体系**：重构 `MultiStreamSkeletonDataset` 逻辑，强制使用 `nslt_2000.json` 索引，实现学生与教师的“零误差对齐”。
2.  **保留全局特征**：将 `normalize_wrist` 设为 `false`，让学生模型看到与教师模型一致的坐标分布。
3.  **回归标准架构**：暂时禁用 `Shift-GCN` 与 `Bottleneck`，改用标准 `ST-GCN` 进行三流融合，降低训练噪声。
4.  **超参精细对齐**：采用 `batch_size: 64` 与 `base_lr: 0.1`（严格遵守线性缩放原则）。

### 6.4 最终实验结果 (Epoch 70)
经过上述优化，多流模型展现了极强的性能爆发力：
- **最好 Top1 准确率**：**37.13%** (对比单流 28.24%，净提升 **+8.89%**)
- **最好 Top5 准确率**：**69.26%**
- **收敛状态**：在第 70 Epoch 降学习率至 0.001 后，模型进入极稳健的精修期，Top1 稳定在 37% 以上。

### 6.5 结论
通过“标签强制对齐”和“保留全局空间信息”，多流架构（Joint+Bone+Motion）在不使用特征蒸馏（Hint Loss）的情况下，仅靠 Logits 蒸馏就已大幅超越单流模型。这证明了多流特征在手语识别中具有极强的信息互补性，是实现高精度轻量化模型的关键路径。

