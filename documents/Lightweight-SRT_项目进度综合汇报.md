# Lightweight-SRT 项目进度综合汇报 (Nature Polished)

## 1. 词级识别阶段 (WLASL2000) 核心突破与 SOTA 成果

在项目的第一阶段，我们聚焦于 **WLASL2000** 大规模孤立词数据集上的轻量化手语识别。针对中期遇到的性能瓶颈，项目组通过算法重构与训练策略调整，成功实现了精度的重大跨越。

### 1.1 弱教师瓶颈与多流对齐优化 (`aligned_v2`)
早期蒸馏实验（Hint Loss + Logits KD）在 **33% - 34%** 的 Top1 精度处遭遇瓶颈。经深度诊断，发现教师模型（I3D）在该任务下的 Top1 仅为 33.0%，产生了明显的“弱教师锚定效应”；此外，由于误用字母排序的 `label_map.json`，导致学生模型与教师模型的 Logits 索引产生偏差，梯度信号互相冲突。

为此，我们实施了 `aligned_v2` 优化方案：
- **标签索引强制对齐**：重构 Dataset 逻辑，强制学生模型读取 `nslt_2000.json` 索引，确保蒸馏信号的 100% 对齐。
- **全局位置特征保留**：关闭手腕归一化（`normalize_wrist: false`），保留手部在空间中的绝对绝对坐标，从而对齐教师模型的全局视觉感知。
- **回归标准三流架构**：暂时禁用 Shift-GCN 与 Bottleneck 结构，采用标准三流（Joint + Bone + Motion）ST-GCN。
- **最终成效**：仅通过 Logits 蒸馏，多流基准模型的 Top1 准确率即大幅跃升至 **37.13%**（Top5 准确率达 **69.26%**）。

### 1.2 冠军精炼策略与顶峰性能 (`v4-Refine`)
当学生模型性能反超教师模型后，项目组迅速进入“去教师化”阶段，提出“精英权重二次提纯”方案：
- **弱约束微调**：加载前期最优的 37.13% 多流预训练权重，将蒸馏干预权重下调至极微弱水平（`kd_alpha = 0.05`，`hint_weight = 0.02`），使蒸馏转为特征正则项，规避教师模型的性能限制。
- **超参精细调节**：使用 0.001 极低学习率（LR）和 5.0 高温平滑（T），在 Batch Size=64 下进行精修微调。
- **最终成效**：Top1 准确率冲刺至 **38.28%**（Top5 达到 **71.17%**），创下当前项目 SOTA（最高水平）记录，大幅超越了 33.0% 的教师模型，达到了业界同类轻量化骨骼点架构的顶尖水平。

---

## 2. 连续手语翻译阶段 (CSLT) 路线规划与进展

随着词级识别任务完成收敛，项目目前正在全面向连续手语视频到自然语言句子的翻译系统（Continuous Sign Language Translation）演进。

### 2.1 主数据集定位：CSL-Daily
我们已选定 **CSL-Daily（中国手语连续翻译数据集）** 作为后续主攻数据集。该数据集提供连续手语视频、gloss 序列和中文文本的三元标注，非常契合我们输出自然语言句子的实用化方向。原 WLASL2000 模型将不再作为最终分类器，而是作为骨骼时序视觉编码器（Encoder）的预训练基底。

### 2.2 四阶段技术开发管线 (V0 - V3)
- **V0 阶段（数据管线）**：提取手部骨骼点多流特征，构建变长序列的 Padding/Mask 处理，生成词表与中文 Tokenizer。
- **V1 阶段（连续手语识别 CSLR）**：搭建 `ST-GCN Encoder -> Temporal Adapter -> CTC Head` 结构，加载预训练多流权重，利用 CTC Loss 微调以降低 WER（词错误率）。
- **V2 阶段（连续手语翻译 SLT）**：引入轻量级 Transformer Decoder，实现基于视频特征/时序特征向中文自然语言句子的翻译，优化 BLEU-1/2/3/4 和 ROUGE-L 指标。
- **V3 阶段（端侧部署）**：导出 TorchScript Lite，对 Encoder 进行 INT8 量化，评估 Android 端的推理延迟与内存占用。

### 2.3 当前里程碑状态
目前项目正处于 **W1（数据与词表准备）** 向 **W2（CSLR 快速基准验证）** 迈进的关键阶段。数据清理与词表建立已基本就绪，正着手调整 `return_sequence=True` 的 Encoder 结构。

---

## 3. 模型架构与自蒸馏机制的可视化表示

为确保学术严谨性，项目组使用 Matplotlib 自动绘图脚本绘制了教师模型和学生模型的学术级图表表示，已导出至 `visualizations/` 目录中：

1. **CorrNet 教师模型 (Panel a, b, c)**:
   - *Panel a* 展示了从 3D 原始帧输入到 3D ResNet18（含 CorrBlock 残差融合）、1D 时序卷积，以及由 2-Layer BiLSTM 引导的自监督相互蒸馏（SMKD）全景数据流。
   - *Panel b* 详解了 Get_Correlation 模块中 Affinity（时序相似度）与 Multi-scale Spatial Aggregation（空间多尺度聚合）的调制机制。
   - *Panel c* 表征了自蒸馏中 $\mathcal{L}_{SeqCTC}$、$\mathcal{L}_{ConvCTC}$ 与 $\mathcal{L}_{SeqKD}$ 共同构成的损失约束。
2. **ST-GCN 学生模型 (Panel a, b, c)**:
   - *Panel a* 绘制了 10 层 st_gcn 级联骨干网络（时序降采样、通道 64-128-256）及 FCN 分类输出。
   - *Panel b* 梳理了 st_gcn 单元内部 GCN（邻接矩阵重要度乘积）、TCN（时序卷积）及残差路径的级联激活关系。
   - *Panel c* 展示了时空图（Spatial-Temporal Graph）中节点（Joints）、帧内空间骨骼（Bones）与跨帧时序边（Temporal Edges）的拓扑表示。
3. **I3D 视频教师模型 (Panel a, b, c)**:
   - *Panel a* 刻画了 3D 卷积/最大池化 Stem 与堆叠 3D Inception 模块（Mixed 3b-5c，通道至 1024D）的串联结构。
   - *Panel b* 拆解了 Inception-3D 模块的 4 个并行分支（$1\times1\times1$、分支 1 和 2 的串联 $3\times3\times3$、分支 3 的池化合并）并拼接通道的逻辑。
   - *Panel c* 表征了多感受野时空特征的拼接与维度关系（$C_{out} = \Sigma d_{branch}$）。

---

## 4. 团队职责分工安排

- **A 同学 (数据工程)**：负责 CSL-Daily 数据集清理、多流特征提取、词表构建及数据管线可视化。
- **B 同学 (模型结构)**：负责时序 Encoder 升级、Temporal Adapter、CTC 分类头以及 Transformer Decoder 翻译模块搭建。
- **C 同学 (训练与迁移)**：负责 WLASL2000 模型权重向连续翻译网络的迁移微调、联合训练优化及训练日志维护。
- **D 同学 (推理与部署)**：负责实现 Greedy / Beam Search 文本解码器、翻译指标评测（BLEU/ROUGE）以及端侧 TorchScript 量化导出与延迟评估。
