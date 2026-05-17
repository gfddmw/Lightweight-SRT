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

### 5.2 正式训练（WLASL2000）
`python src/student_model/distillation/recognition_kd_hint_new.py --config configs/student/st_gcn/wlasl2000/train_kd_hint_new.yaml`

