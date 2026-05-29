# Lightweight-SRT 多流蒸馏训练实验报告 (WLASL2000)

## 1. 实验结果综述
经过针对性优化的多流模型在 **WLASL2000** 数据集上取得了显著的性能突破，全面超越了单流 Fast-KD 基准。

| 模型版本 | Top1 准确率   | Top5 准确率 | 状态 | 备注 |
| :--- |:-----------| :--- | :--- | :--- |
| **多流优化版 (aligned_v2)** | **37.13%** | **69.26%** | **当前最优 (SOTA)** | 标签对齐 + 关闭归一化 |
| 单流 Fast-KD 基准 | 23.33%     | 53.11% | 已超越 | 基于 nslt_2000 索引 |
| 多流初始版本 | ~19.49%    | ~44.91% | 性能不达标 | 标签错位 + 开启归一化 |

---

## 2. 核心优化办法 (Key Methodology)

为了解决多流模型前期表现不如单流的问题，本实验实施了以下核心改进措施：

### 2.1 标签索引强制对齐 (Label Alignment)
*   **实施方法**：重构 `MultiStreamSkeletonDataset` 类，使其直接读取 `data/nslt_2000.json` 中的 `action` 索引，彻底弃用之前基于字母排序生成的 `label_map.json`。
*   **核心逻辑**：确保学生模型（ST-GCN）的分类目标与教师模型（I3D）输出的 Logits 索引实现 **100% 同步**。
*   **效果**：消除了由于“标签错位”导致的冲突梯度信号，让知识蒸馏过程变得真正有效。

### 2.2 全局位置特征保留 (Spatial Preservation)
*   **实施方法**：在配置文件中设置 `normalize_wrist: false`，停止将骨架坐标移动到以手腕为原点的局部空间。
*   **核心逻辑**：由于教师模型是通过视频进行全局感知的，手部在空间中的绝对位置（如：在头顶、胸前或腰部）包含关键语义。
*   **效果**：降低了跨模态对齐的难度，使学生模型能够捕捉到教师关注的全局空间特征。

### 2.3 架构简化与超参对齐 (Stabilization)
*   **架构调整**：暂时禁用 `Shift-GCN` 与 `Bottleneck` 优化项，回归最稳健的标准 `ST-GCN` 堆叠架构。
*   **参数配置**：
    *   `batch_size: 64`
    *   `base_lr: 0.1` (严格遵守线性缩放原则)
    *   `use_aux_loss: true` (权重 0.2)，确保 Joint、Bone、Motion 各流独立收敛。
*   **效果**：极大降低了模型在 2000 类识别任务中的训练不稳定性，Top1 曲线更加平滑。

---

## 3. 训练启动指南 (Training Instructions)

为了复现上述 **37.13%** 的 SOTA 结果，请遵循以下指令进行训练：

### 3.1 核心配置确认
在启动前，请确保 `configs/student/multi_stream.yaml` 中的关键参数如下：
- `split_json: data/nslt_2000.json` (对齐标签)
- `normalize_wrist: false` (保留全局空间信息)
- `batch_size: 64`
- `base_lr: 0.1`

### 3.2 启动命令
在项目根目录下执行以下单行指令：

```powershell
python src/student_model/distillation/recognition_kd.py --config configs/student/multi_stream.yaml --work_dir work_dir/recognition/wlasl2000/multistream_aligned_v2
```

### 3.3 训练过程监控
- **首个评估点 (Epoch 4)**：Top1 预计在 5% 左右，这是正常起步。
- **关键跳变点 (Epoch 40)**：学习率降至 0.01，Top1 预计会冲刺至 30% 以上。
- **精修稳定点 (Epoch 70)**：学习率降至 0.001，Top1 预计达到 37%+ 的最终峰值。

### 3.4 纯多流分类对比训练 (对齐蒸馏标签)
若需对比蒸馏效果，建议使用同一套标签体系（即 `nslt_2000.json`）：
```powershell
python src/student_model/distillation/recognition_kd.py --config configs/student/multi_stream.yaml --kd_alpha 0 --work_dir work_dir/recognition/wlasl2000/multistream_kd_baseline
```

### 3.5 标准分类训练流 (独立 Baseline)
若需按照传统流程，使用字母排序标签进行独立训练，请执行以下完整步骤：

1. **准备索引与标签**：
   ```powershell
   python scripts/data_prep/split.py
   python scripts/data_prep/generate_label_map.py
   ```
2. **启动训练**：
   ```powershell
   python src/student_model/train_student.py --config configs/student/multi_stream.yaml --work_dir work_dir/recognition/wlasl2000/multistream_standard_baseline
   ```
   *注意：此流程会使用 `processed/label_map.json`，标签索引与 `nslt_2000.json` 不一致，仅用于非蒸馏场景。*

---

## 4. 实验结论
通过本次实验证明，**多流特征（关节、骨骼、运动）在手语识别中具有极强的信息互补性**。

实验关键发现：
1.  **标签一致性是蒸馏的前提**：跨模态蒸馏对标签顺序极其敏感，任何微小的索引错位都会导致模型无法收敛。
2.  **多流潜力巨大**：在对齐数据分布后，多流模型仅凭 Logits 蒸馏即可在 WLASL2000 上实现 **+13.8%** 的绝对精度提升。
3.  **收敛周期长**：多流模型由于参数量大，需要跑满至少 70-120 Epoch 才能在第二次学习率下调后展现出其真正的上限性能。

**当前状态**：该模型已在第 70 Epoch 成功收敛并稳定，准确率达到 37.13%，是目前工程化部署的最佳备选方案。
