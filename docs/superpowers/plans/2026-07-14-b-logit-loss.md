# B 角色：序列概率蒸馏模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现时序 Sequence KD 损失模块（KL 散度 + 温度缩放），并整合到联合训练脚本中，为后续翻译 CE Loss 预留接口。

**Architecture:** 新建 `logit_loss.py` 文件，包含 `SequenceKDLoss` 类（处理教师-学生 logits 维度对齐、时序长度对齐、padding mask），以及 `MultiTaskLoss` 组合器（CTC + KD + 未来 CE Loss 的统一权重调度）。修改 `train_cslt.py` 集成新损失模块，添加命令行参数控制。

**Tech Stack:** PyTorch, torch.nn.functional (KL div, log_softmax, softmax, CTCLoss)

## 全局约束

- 教师 logits 维度: `[B, T_t, 2001]`（CorrNet 输出，2000 gloss + 1 blank）
- 学生 CTC logits 维度: `[B, T_out, 7388]`（当前代码，7387 + 1 blank）
- Gloss 词表实际大小: 2004（`gloss_vocab.json`）
- 温度参数默认 T=5.0，KD 权重默认 λ=0.3
- 所有损失模块必须支持 fp16 混合精度下的数值稳定
- 必须处理变长序列的 padding mask

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/student_model/architecture/logit_loss.py` | **新建** | SequenceKDLoss + MultiTaskLoss 组合器 |
| `src/student_model/train_cslt.py` | **修改** | 集成新损失模块，添加 CLI 参数，修复 num_classes |
| `src/student_model/architecture/cslt_model.py` | **修改** | 修复 CTCHead 的 num_classes 从 7388 → 2005 |

---

### Task 1: 修复 CTCHead 的 num_classes 词表维度错误

**Files:**
- Modify: `src/student_model/architecture/cslt_model.py:48`

**Interfaces:**
- Consumes: 无（独立修复）
- Produces: `CTCHead(num_classes=2005)` — 正确匹配 gloss_vocab 大小 2004 + 1 blank

**说明:** 当前 CTCHead 的 `num_classes=7388` 是 bug。`dataset_summary.json` 中 `total_glosses: 7387` 是"所有样本的 gloss token 总数"，不是词表大小。实际 gloss_vocab 只有 2004 个唯一 token。正确值应为 2004 + 1 (blank) = 2005。教师 logits 是 2001 维（CorrNet 训练时用了 2000 个 gloss），与 2005 差 4 个 token，后续 KD 损失会处理这个映射。

- [ ] **Step 1: 修改 CTCHead 默认参数**

将 `cslt_model.py:48` 的 `num_classes` 默认值从 `7388` 改为 `2005`：

```python
# 修改前 (line 48)
def __init__(self, in_channels=1024, num_classes=7388):

# 修改后
def __init__(self, in_channels=1024, num_classes=2005):  # 2004 gloss + 1 blank
```

- [ ] **Step 2: 同步修改 CSLTModel 的默认参数**

将 `cslt_model.py:68` 的 `num_classes` 默认值从 `7388` 改为 `2005`：

```python
# 修改前 (line 68)
def __init__(self, encoder, in_channels=768, out_channels=1024, num_classes=7388):

# 修改后
def __init__(self, encoder, in_channels=768, out_channels=1024, num_classes=2005):
```

- [ ] **Step 3: 同步修改 `__main__` 测试代码中的断言**

将 `cslt_model.py:109, 138-139` 的测试断言从 `7388` 改为 `2005`：

```python
# 修改前 (line 109)
num_classes=7388

# 修改后
num_classes=2005

# 修改前 (lines 138-139)
assert adapted_feat.shape == (B, 25, 1024), ...
assert ctc_logits.shape == (B, 25, 7388), ...

# 修改后
assert adapted_feat.shape == (B, 25, 1024), ...
assert ctc_logits.shape == (B, 25, 2005), ...
```

- [ ] **Step 4: 运行单元测试验证**

```bash
cd D:\nju\2\SLT\code\Lightweight-SRT && python src/student_model/architecture/cslt_model.py
```

Expected: `CSLTModel unit test passed successfully!`

- [ ] **Step 5: 同步修改 train_cslt.py 和 evaluate_cslt.py 中的 num_classes**

在 `train_cslt.py:207` 和 `evaluate_cslt.py:142`，将 `num_classes=7388` 改为 `num_classes=2005`：

```python
# train_cslt.py line 207
model = CSLTModel(
    encoder=encoder,
    in_channels=768,
    out_channels=1024,
    num_classes=2005  # 修改
)

# evaluate_cslt.py line 142
model = CSLTModel(
    encoder=encoder,
    in_channels=768,
    out_channels=1024,
    num_classes=2005  # 修改
)
```

同步修改 `evaluate_cslt.py:63, 180` 的 blank_id 从 `7387` 改为 `2004`：

```python
# evaluate_cslt.py line 63
def ctc_greedy_decode(ctc_logits, input_lengths, blank_id=2004):  # 修改

# evaluate_cslt.py line 180
decoded_preds = ctc_greedy_decode(ctc_logits, input_lengths, blank_id=2004)  # 修改
```

同步修改 `train_cslt.py:223` 的 CTC blank 参数：

```python
# train_cslt.py line 223
ctc_loss = nn.CTCLoss(blank=2004, zero_infinity=True)  # 修改
```

- [ ] **Step 6: Commit**

```bash
git add src/student_model/architecture/cslt_model.py src/student_model/train_cslt.py src/student_model/evaluate_cslt.py
git commit -m "fix: correct CTCHead num_classes from 7388 to 2005 (gloss vocab size + blank)"
```

---

### Task 2: 创建 SequenceKDLoss 模块

**Files:**
- Create: `src/student_model/architecture/logit_loss.py`

**Interfaces:**
- Consumes: 教师 logits `[B, T_t, 2001]`、学生 logits `[B, T_out, 2005]`、teacher_lengths `[B]`、student_lengths `[B]`
- Produces: `SequenceKDLoss.forward()` → scalar loss (float), `MultiTaskLoss.forward()` → dict of losses + total scalar

**说明:** 教师 logits 为 2001 维（CorrNet 的 2000 gloss + 1 blank），学生 logits 修复后为 2005 维（2004 gloss + 1 blank）。KD 损失需要先将教师 logits 映射到学生的 gloss 空间。映射策略：教师 blank 在 index 2000，学生 blank 在 index 2004，gloss 0-1999 对应 student gloss 0-1999 的前 2000 个，student 多出的 gloss 2000-2003 在 KD 中 mask 掉。

- [ ] **Step 1: 创建文件骨架和 docstring**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时序序列知识蒸馏损失模块。

包含:
- SequenceKDLoss: KL 散度 + 温度缩放的时序概率分布对齐
- MultiTaskLoss: CTC + KD + CE 多任务损失组合器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceKDLoss(nn.Module):
    """时序序列知识蒸馏损失。
    
    通过 KL 散度对齐教师与学生的时序预测概率分布。
    处理维度不匹配（教师 2001 vs 学生 2005）和时序长度不匹配。
    
    Args:
        temperature: 蒸馏温度，默认 5.0
        teacher_blank: 教师 blank index，默认 2000
        student_blank: 学生 blank index，默认 2004
        teacher_vocab_size: 教师词表大小（不含 blank），默认 2000
        student_vocab_size: 学生词表大小（不含 blank），默认 2004
    """
    
    def __init__(self, temperature=5.0, teacher_blank=2000, student_blank=2004,
                 teacher_vocab_size=2000, student_vocab_size=2004):
        super().__init__()
        self.temperature = temperature
        self.teacher_blank = teacher_blank
        self.student_blank = student_blank
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
    
    def _align_vocabulary(self, teacher_logits):
        """将教师 logits 从 [..., 2001] 映射到 [..., 2005]。
        
        教师: [gloss_0, ..., gloss_1999, blank_2000]  (2001 dim)
        学生: [gloss_0, ..., gloss_1999, gloss_2000, ..., gloss_2003, blank_2004]  (2005 dim)
        
        映射策略:
        - 教师 gloss 0-1999 → 学生 gloss 0-1999
        - 教师 blank 2000 → 学生 blank 2004
        - 学生 gloss 2000-2003 (教师没有的) 填充极小值 -inf
        """
        B, T_t, _ = teacher_logits.shape
        aligned = torch.full((B, T_t, self.student_vocab_size + 1),
                             float('-inf'),
                             device=teacher_logits.device,
                             dtype=teacher_logits.dtype)
        # 复制 gloss 0-1999
        aligned[:, :, :self.teacher_vocab_size] = teacher_logits[:, :, :self.teacher_vocab_size]
        # 复制 blank
        aligned[:, :, self.student_blank] = teacher_logits[:, :, self.teacher_blank]
        return aligned
    
    def _align_temporal(self, student_logits, teacher_logits, student_lengths, teacher_lengths):
        """处理时序长度不匹配。
        
        学生输出的时间步 T_out 与教师的时间步 T_t 可能不同。
        策略：将教师 logits 在时间维度上线性插值到学生的时间步数。
        如果 T_t == T_out，直接返回。
        """
        T_out = student_logits.size(1)
        T_t = teacher_logits.size(1)
        
        if T_t == T_out:
            return teacher_logits
        
        # 线性插值: [B, T_t, D] → [B, T_out, D]
        teacher_logits = teacher_logits.permute(0, 2, 1)  # [B, D, T_t]
        teacher_logits = F.interpolate(teacher_logits, size=T_out, mode='linear', align_corners=False)
        teacher_logits = teacher_logits.permute(0, 2, 1)  # [B, T_out, D]
        return teacher_logits
    
    def _build_mask(self, student_lengths, student_logits, teacher_lengths):
        """构建时序 padding mask。
        
        同时考虑学生和教师的实际长度，取两者较小值作为有效区域。
        返回 [B, T_out] 的 bool mask，True 表示有效。
        """
        B, T_out, _ = student_logits.shape
        device = student_logits.device
        
        # 学生有效长度 (下采样后)
        student_mask = torch.arange(T_out, device=device).unsqueeze(0) < student_lengths.unsqueeze(1)
        
        # 教师有效长度 (插值后与学生 T_out 对齐)
        teacher_mask = torch.arange(T_out, device=device).unsqueeze(0) < teacher_lengths.unsqueeze(1)
        
        # 取交集
        mask = student_mask & teacher_mask
        return mask  # [B, T_out]
    
    def forward(self, student_logits, teacher_logits, student_lengths, teacher_lengths):
        """计算 Sequence KD Loss。
        
        Args:
            student_logits: [B, T_out, 2005] 学生 CTC head 输出
            teacher_logits: [B, T_t, 2001] 教师 logits
            student_lengths: [B] 学生下采样后的有效长度
            teacher_lengths: [B] 教师原始有效长度
        
        Returns:
            scalar loss (float tensor)
        """
        # 1. 词汇对齐: 教师 2001 → 2005
        teacher_aligned = self._align_vocabulary(teacher_logits)
        
        # 2. 时序对齐: 教师 T_t → 学生 T_out
        teacher_aligned = self._align_temporal(
            student_logits, teacher_aligned, student_lengths, teacher_lengths
        )
        
        # 3. 构建 mask
        mask = self._build_mask(student_lengths, student_logits, teacher_lengths)  # [B, T_out]
        
        # 4. 温度缩放 + KL 散度
        # 学生: log-softmax
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        # 教师: softmax (detach 防止梯度回传)
        teacher_probs = F.softmax(teacher_aligned / self.temperature, dim=-1).detach()
        
        # KL 散度: sum(p_teacher * (log(p_teacher) - log(p_student)))
        # 等价于对每个位置计算 KL，然后 mask 求和
        kl_per_token = F.kl_div(
            student_log_probs, teacher_probs,
            reduction='none', log_target=False
        ).sum(dim=-1)  # [B, T_out]
        
        # 5. Mask 应用 + 归一化
        kl_per_token = kl_per_token * mask  # padding 位置置零
        loss = kl_per_token.sum() / mask.sum().clamp(min=1)
        
        # 温度缩放回正常尺度: loss * T^2
        loss = loss * (self.temperature ** 2)
        
        return loss
```

- [ ] **Step 2: 创建 MultiTaskLoss 组合器**

在同一个文件中追加：

```python
class MultiTaskLoss(nn.Module):
    """多任务损失组合器：CTC + KD + CE (翻译)。
    
    Args:
        ctc_weight: CTC 损失权重，默认 1.0
        kd_weight: KD 蒸馏损失权重，默认 0.3
        ce_weight: 翻译 CE 损失权重，默认 0.0（翻译模块未接入时为 0）
        kd_temperature: KD 温度，默认 5.0
        ctc_blank: CTC blank index，默认 2004
    """
    
    def __init__(self, ctc_weight=1.0, kd_weight=0.3, ce_weight=0.0,
                 kd_temperature=5.0, ctc_blank=2004):
        super().__init__()
        self.ctc_weight = ctc_weight
        self.kd_weight = kd_weight
        self.ce_weight = ce_weight
        
        self.ctc_loss = nn.CTCLoss(blank=ctc_blank, zero_infinity=True)
        self.kd_loss = SequenceKDLoss(temperature=kd_temperature)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # PAD=0
    
    def forward(self, student_logits, teacher_logits, gloss_ids,
                student_lengths, teacher_lengths, gloss_lengths,
                translation_logits=None, text_ids=None):
        """计算组合多任务损失。
        
        Args:
            student_logits: [B, T_out, 2005] 学生 CTC logits
            teacher_logits: [B, T_t, 2001] 教师 logits
            gloss_ids: [B, G] gloss 标注
            student_lengths: [B] 学生输出有效长度
            teacher_lengths: [B] 教师有效长度
            gloss_lengths: [B] gloss 序列长度
            translation_logits: [B, L, text_vocab_size] 翻译输出 (可选, 为 C 预留)
            text_ids: [B, L] 文本标注 (可选, 为 C 预留)
        
        Returns:
            dict: {
                'total': total_loss,
                'ctc': ctc_loss_value,
                'kd': kd_loss_value,
                'ce': ce_loss_value (if applicable)
            }
        """
        losses = {}
        
        # CTC Loss
        log_probs = F.log_softmax(student_logits, dim=-1).transpose(0, 1)  # [T_out, B, 2005]
        ctc_val = self.ctc_loss(log_probs, gloss_ids, student_lengths, gloss_lengths)
        losses['ctc'] = ctc_val
        total = self.ctc_weight * ctc_val
        
        # KD Loss
        if self.kd_weight > 0:
            kd_val = self.kd_loss(student_logits, teacher_logits, student_lengths, teacher_lengths)
            losses['kd'] = kd_val
            total = total + self.kd_weight * kd_val
        
        # CE Loss (翻译, 预留)
        if self.ce_weight > 0 and translation_logits is not None and text_ids is not None:
            B, L, V = translation_logits.shape
            ce_val = self.ce_loss(
                translation_logits.reshape(B * L, V),
                text_ids.reshape(B * L)
            )
            losses['ce'] = ce_val
            total = total + self.ce_weight * ce_val
        
        losses['total'] = total
        return losses
```

- [ ] **Step 3: 运行单元测试验证 SequenceKDLoss**

```bash
cd D:\nju\2\SLT\code\Lightweight-SRT && python -c "
import torch
import sys
sys.path.insert(0, '.')
from src.student_model.architecture.logit_loss import SequenceKDLoss, MultiTaskLoss

# 测试 SequenceKDLoss
kd = SequenceKDLoss(temperature=5.0)
student_logits = torch.randn(2, 25, 2005)
teacher_logits = torch.randn(2, 30, 2001)
student_lengths = torch.tensor([20, 25])
teacher_lengths = torch.tensor([24, 30])

loss = kd(student_logits, teacher_logits, student_lengths, teacher_lengths)
print(f'KD Loss: {loss.item():.6f}')
assert not torch.isnan(loss), 'Loss should not be NaN'
assert loss.item() > 0, 'Loss should be positive'
print('SequenceKDLoss test passed!')

# 测试 MultiTaskLoss
mtl = MultiTaskLoss(ctc_weight=1.0, kd_weight=0.3)
gloss_ids = torch.randint(1, 2004, (2, 10))
gloss_lengths = torch.tensor([8, 10])
result = mtl(student_logits, teacher_logits, gloss_ids,
             student_lengths, teacher_lengths, gloss_lengths)
print(f'Total: {result[\"total\"].item():.6f}, CTC: {result[\"ctc\"].item():.6f}, KD: {result[\"kd\"].item():.6f}')
assert 'total' in result and 'ctc' in result and 'kd' in result
print('MultiTaskLoss test passed!')
"
```

Expected: 所有 assert 通过，无 NaN。

- [ ] **Step 4: Commit**

```bash
git add src/student_model/architecture/logit_loss.py
git commit -m "feat: add SequenceKDLoss and MultiTaskLoss for temporal knowledge distillation"
```

---

### Task 3: 集成新损失模块到 train_cslt.py

**Files:**
- Modify: `src/student_model/train_cslt.py`

**Interfaces:**
- Consumes: `SequenceKDLoss`, `MultiTaskLoss` from `logit_loss.py`; 现有 `CSLTDataset`, `CSLTModel`, `prepare_input`
- Produces: 联合损失训练流程（CTC + KD），支持 CLI 控制权重

**说明:** 将现有的单独 CTC Loss 训练替换为 `MultiTaskLoss` 组合器。添加命令行参数控制 KD 权重和温度。保留 debug_overfit 模式兼容性。修正 `prepare_input` 不再返回不需要的 `input_lengths` 原始值，改为返回下采样后的 student_lengths。

- [ ] **Step 1: 添加 import**

在 `train_cslt.py` 顶部现有 import 之后追加：

```python
from src.student_model.architecture.logit_loss import MultiTaskLoss, SequenceKDLoss
```

- [ ] **Step 2: 修改 `prepare_input` 返回 teacher 数据**

修改 `train_cslt.py:106-124` 的 `prepare_input` 函数，使其也返回 teacher 相关信息：

```python
def prepare_input(batch, device):
    """处理输入数据，拆分左右手，并返回所有训练所需张量。"""
    joints = batch["joints"]
    bones = batch["bones"]
    motion = batch["motion"]
    
    joints_split = torch.cat([joints[..., :21, :], joints[..., 21:, :]], dim=-1).to(device)
    bones_split = torch.cat([bones[..., :21, :], bones[..., 21:, :]], dim=-1).to(device)
    motion_split = torch.cat([motion[..., :21, :], motion[..., 21:, :]], dim=-1).to(device)
    
    input_lengths = batch["input_lengths"].to(device)
    gloss_ids = batch["gloss_ids"].to(device)
    gloss_lengths = batch["gloss_lengths"].to(device)
    teacher_logits = batch["teacher_logits"].to(device)
    teacher_lengths = batch["teacher_lengths"].to(device)
    
    return (joints_split, bones_split, motion_split,
            input_lengths, gloss_ids, gloss_lengths,
            teacher_logits, teacher_lengths)
```

- [ ] **Step 3: 添加 CLI 参数**

在 `train_cslt.py` 的 `argparse` 部分（约 line 148 之后）追加：

```python
    # 蒸馏相关参数
    parser.add_argument('--kd_weight', type=float, default=0.3,
                        help='KD loss weight (default: 0.3, set 0 to disable)')
    parser.add_argument('--kd_temperature', type=float, default=5.0,
                        help='KD temperature for softening (default: 5.0)')
    parser.add_argument('--ce_weight', type=float, default=0.0,
                        help='Translation CE loss weight (default: 0.0, reserved for W4)')
```

- [ ] **Step 4: 替换损失函数和优化逻辑**

将 `train_cslt.py:222-224` 的 CTC Loss 单独创建替换为 MultiTaskLoss：

```python
    # 6. 损失函数与优化器
    # 使用 MultiTaskLoss 组合 CTC + KD + CE
    multi_task_loss = MultiTaskLoss(
        ctc_weight=1.0,
        kd_weight=args.kd_weight,
        ce_weight=args.ce_weight,
        kd_temperature=args.kd_temperature,
        ctc_blank=2004
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
```

- [ ] **Step 5: 修改 debug_overfit 训练循环**

将 `train_cslt.py:232-269` 的 debug_overfit 循环替换为使用 MultiTaskLoss：

```python
    if args.debug_overfit:
        print("开始单批次过拟合训练循环 (50 Epochs)...")
        debug_batch = next(iter(dataloader))
        
        joints, bones, motion, input_lengths, gloss_ids, gloss_lengths, teacher_logits, teacher_lengths = \
            prepare_input(debug_batch, device)
        
        # 预计算学生输出有效长度
        student_lengths_debug = (input_lengths - 1) // 4 + 1
        
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(joints=joints, bones=bones, motion=motion)
                ctc_logits = outputs["ctc_logits"]
            
            # 转为 FP32 计算损失
            ctc_logits_fp32 = ctc_logits.float()
            teacher_logits_fp32 = teacher_logits.float()
            
            with torch.cuda.amp.autocast(enabled=False):
                loss_dict = multi_task_loss(
                    student_logits=ctc_logits_fp32,
                    teacher_logits=teacher_logits_fp32,
                    gloss_ids=gloss_ids,
                    student_lengths=student_lengths_debug,
                    teacher_lengths=teacher_lengths,
                    gloss_lengths=gloss_lengths,
                )
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ctc_v = loss_dict['ctc'].item()
            kd_v = loss_dict.get('kd', torch.tensor(0.0)).item()
            print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] - CTC: {ctc_v:.6f} | KD: {kd_v:.6f} | Total: {loss.item():.6f}")
            
        print("Debug 过拟合训练完成！")
        overfit_path = work_dir / "overfit_model.pt"
        torch.save(model.state_dict(), overfit_path)
        print(f"已保存过拟合权重至: {overfit_path}")
```

- [ ] **Step 6: 修改正式训练循环**

将 `train_cslt.py:271-322` 的正式训练循环替换为使用 MultiTaskLoss：

```python
    else:
        print(f"开始正式训练循环 ({args.epochs} Epochs)...")
        for epoch in range(args.epochs):
            epoch_losses = []
            epoch_ctc_losses = []
            epoch_kd_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                joints, bones, motion, input_lengths, gloss_ids, gloss_lengths, teacher_logits, teacher_lengths = \
                    prepare_input(batch, device)
                
                student_lengths = (input_lengths - 1) // 4 + 1
                
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    outputs = model(joints=joints, bones=bones, motion=motion)
                    ctc_logits = outputs["ctc_logits"]
                
                ctc_logits_fp32 = ctc_logits.float()
                teacher_logits_fp32 = teacher_logits.float()
                
                with torch.cuda.amp.autocast(enabled=False):
                    loss_dict = multi_task_loss(
                        student_logits=ctc_logits_fp32,
                        teacher_logits=teacher_logits_fp32,
                        gloss_ids=gloss_ids,
                        student_lengths=student_lengths,
                        teacher_lengths=teacher_lengths,
                        gloss_lengths=gloss_lengths,
                    )
                    loss = loss_dict['total']
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(loss.item())
                epoch_ctc_losses.append(loss_dict['ctc'].item())
                kd_val = loss_dict.get('kd', torch.tensor(0.0)).item()
                epoch_kd_losses.append(kd_val)
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
                          f"Batch [{batch_idx+1:03d}/{len(dataloader):03d}] "
                          f"- CTC: {loss_dict['ctc'].item():.6f} "
                          f"| KD: {kd_val:.6f} "
                          f"| Total: {loss.item():.6f}")
            
            mean_loss = np.mean(epoch_losses)
            mean_ctc = np.mean(epoch_ctc_losses)
            mean_kd = np.mean(epoch_kd_losses)
            print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] Finished | "
                  f"Avg CTC: {mean_ctc:.6f} | Avg KD: {mean_kd:.6f} | Avg Total: {mean_loss:.6f}")
            
            # 保存 checkpoint
            checkpoint_path = work_dir / f"epoch_{epoch+1}_loss_{mean_loss:.4f}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                'ctc_loss': mean_ctc,
                'kd_loss': mean_kd,
            }, checkpoint_path)
            print(f"已保存 Checkpoint 至: {checkpoint_path}")
            
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_path = work_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                print(f"更新最佳模型 (Best Total Loss: {best_loss:.6f}) 并保存至: {best_path}")
```

- [ ] **Step 7: 验证语法正确性**

```bash
cd D:\nju\2\SLT\code\Lightweight-SRT && python -c "import py_compile; py_compile.compile('src/student_model/train_cslt.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 8: 验证 import 链路**

```bash
cd D:\nju\2\SLT\code\Lightweight-SRT && python -c "
import sys; sys.path.insert(0, '.')
from src.student_model.architecture.logit_loss import SequenceKDLoss, MultiTaskLoss
print('All imports successful')
"
```

- [ ] **Step 9: Commit**

```bash
git add src/student_model/train_cslt.py
git commit -m "feat: integrate MultiTaskLoss (CTC+KD) into train_cslt.py with CLI controls"
```

---

### Task 4: 同步更新 evaluate_cslt.py 的 blank_id

**Files:**
- Modify: `src/student_model/evaluate_cslt.py`

**Interfaces:**
- Consumes: 修复后的 `CSLTModel`（num_classes=2005）
- Produces: 正确使用 blank_id=2004 的 WER 评估

**说明:** 评估脚本中的 blank_id 需要与 CTCHead 的 num_classes 一致。已在 Task 1 Step 5 中同步修改了 `evaluate_cslt.py`，此 Task 做最终验证。

- [ ] **Step 1: 确认 evaluate_cslt.py 中所有 blank_id 和 num_classes 引用**

在 `evaluate_cslt.py` 中搜索确认：
- Line 63: `blank_id=2004`
- Line 142: `num_classes=2005`
- Line 180: `blank_id=2004`

- [ ] **Step 2: 运行语法检查**

```bash
cd D:\nju\2\SLT\code\Lightweight-SRT && python -c "import py_compile; py_compile.compile('src/student_model/evaluate_cslt.py', doraise=True); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/student_model/evaluate_cslt.py
git commit -m "fix: update evaluate_cslt.py blank_id to 2004 for corrected vocab size"
```

---

## 自审清单

1. **Spec 覆盖**: 
   - B 角色职责1 "编写时序 Sequence KD 损失模块" → Task 2 (SequenceKDLoss + MultiTaskLoss)
   - B 角色职责2 "联合微调时 CE Loss 整合 + 多任务权重调优" → Task 2 (MultiTaskLoss 预留 ce_weight) + Task 3 (CLI 参数 --kd_weight, --ce_weight)
   - 核心产出 `logit_loss.py` → Task 2
   - 核心产出 `train_cslt.py` → Task 3

2. **Placeholder 扫描**: 无 "TBD"、"TODO"、"implement later" 等占位符。所有代码均为完整实现。

3. **类型一致性**: 
   - `SequenceKDLoss.forward()` 签名在 Task 2 定义，在 Task 3 的 `MultiTaskLoss.forward()` 中调用，签名一致
   - `MultiTaskLoss.forward()` 返回 `dict` 格式在 Task 2 定义，在 Task 3 中解包 `loss_dict['total']`, `loss_dict['ctc']`, `loss_dict['kd']`，一致
   - `prepare_input` 新返回值在 Task 3 Step 2 定义，在 Step 5/6 中解包，一致

4. **维度对齐验证**:
   - 教师 logits: `[B, T_t, 2001]` → `_align_vocabulary` → `[B, T_t, 2005]` → `_align_temporal` → `[B, T_out, 2005]`
   - 学生 logits: `[B, T_out, 2005]`（修复后）
   - 词汇映射: 教师 gloss 0-1999 → 学生 gloss 0-1999, 教师 blank 2000 → 学生 blank 2004, 学生 gloss 2000-2003 mask 为 -inf
   - 修复后 `num_classes=2005` 与 `blank_id=2004` 一致