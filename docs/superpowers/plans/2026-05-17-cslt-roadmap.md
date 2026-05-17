# CSLT Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the current WLASL2000 isolated sign language system into a Continuous Sign Language Translation (CSLT) system on CSL-Daily with loosely coupled parallel workflows.

**Architecture:** MultiStream ST-GCN Encoder -> Temporal Adapter -> CTC Head -> Translation Decoder.

**Tech Stack:** PyTorch, MediaPipe, Jieba/SentencePiece (Tokenizer), TorchMetrics.

---

## 核心解耦策略 (Decoupling Strategy)

为了实现4人无阻塞并行开发，各阶段必须依赖**接口契约 (Interface Contracts)**而非具体实现。

**数据契约 (Data Contract):**
```python
# A同学输出的数据字典标准格式
batch = {
    "joints": Tensor[B, C, T, V, M],     # 输入特征
    "input_lengths": Tensor[B],          # 变长序列实际长度
    "gloss_ids": Tensor[B, G],           # Gloss标签
    "gloss_lengths": Tensor[B],          # Gloss实际长度
    "text_ids": Tensor[B, L],            # 翻译文本标签
    "text_lengths": Tensor[B],           # 文本实际长度
}
```

**模型契约 (Model Contract):**
```python
# B同学输出的模型组件接口
features = encoder(joints)               # features: [B, T', D]
logits = ctc_head(features)              # logits: [B, T', num_glosses]
text_logits = decoder(features, texts)   # text_logits: [B, L, num_words]
```

---

## Phase 1: CSLR Baseline (W1 - W2)

**目标**: 跑通 `Video -> Gloss` 的连续手语识别 (CTC训练)。

### Task 1.A: Data Pipeline (A同学)
**Files:**
- Create: `src/common/datasets/cslt_dataset.py`
- Create: `tests/common/datasets/test_cslt_dataset.py`

- [ ] **Step 1: Write the failing test for Dataset Contract**
```python
# tests/common/datasets/test_cslt_dataset.py
import torch
from src.common.datasets.cslt_dataset import CSLTDataset, cslt_collate_fn

def test_cslt_dataset_mock():
    # Mock data source
    dataset = CSLTDataset(mock=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=cslt_collate_fn)
    batch = next(iter(loader))
    assert "joints" in batch and batch["joints"].dim() == 5
    assert "gloss_ids" in batch and batch["gloss_ids"].dim() == 2
```
- [ ] **Step 2: Implement Mock Dataset & Collate Function**
```python
# src/common/datasets/cslt_dataset.py
import torch
from torch.utils.data import Dataset

class CSLTDataset(Dataset):
    def __init__(self, mock=False):
        self.mock = mock
    def __len__(self): return 10
    def __getitem__(self, idx):
        return {
            "joints": torch.randn(3, 100, 21, 2),
            "input_lengths": torch.tensor(100),
            "gloss_ids": torch.randint(1, 1000, (5,)),
            "gloss_lengths": torch.tensor(5)
        }

def cslt_collate_fn(batch):
    # simple collation padding logic
    joints = torch.stack([b["joints"] for b in batch])
    gloss_ids = torch.nn.utils.rnn.pad_sequence([b["gloss_ids"] for b in batch], batch_first=True)
    return {"joints": joints, "gloss_ids": gloss_ids, 
            "input_lengths": torch.stack([b["input_lengths"] for b in batch]),
            "gloss_lengths": torch.stack([b["gloss_lengths"] for b in batch])}
```
- [ ] **Step 3: Process Real CSL-Daily Data** (后续独立逐步替换Mock逻辑，不阻塞C同学)。

### Task 1.B: Encoder & CTC Head (B同学)
**Files:**
- Modify: `src/student_model/architecture/multi_stream_stgcn.py:30-40` (Add return_sequence)
- Create: `src/student_model/architecture/cslt_model.py`
- Create: `tests/student_model/test_cslt_model.py`

- [ ] **Step 1: Write shape test for Encoder and CTC Head**
```python
# tests/student_model/test_cslt_model.py
import torch
from src.student_model.architecture.cslt_model import TemporalAdapter, CTCHead

def test_ctc_head_shapes():
    features = torch.randn(2, 50, 256) # B, T', D
    adapter = TemporalAdapter(256, 512)
    head = CTCHead(512, 1000)
    out = head(adapter(features))
    assert out.shape == (2, 50, 1000)
```
- [ ] **Step 2: Implement TemporalAdapter & CTCHead**
```python
# src/student_model/architecture/cslt_model.py
import torch.nn as nn

class TemporalAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
    def forward(self, x): # x: (B, T, C)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

class CTCHead(nn.Module):
    def __init__(self, in_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, vocab_size)
    def forward(self, x):
        return self.fc(x)
```

### Task 1.C: CTC Training Loop (C同学)
**Files:**
- Create: `src/student_model/train_cslr.py`
- Create: `tests/student_model/test_cslr_train.py`

- [ ] **Step 1: Setup Train Loop with Dummy Data/Model**
(依赖A和B同学的Mock/基础实现，直接开始编写损失函数和梯度更新)。
```python
# src/student_model/train_cslr.py
import torch
import torch.nn.functional as F

def train_step(model, batch, optimizer):
    features = batch["joints"] # (B, C, T, V, M) -> 假设模型已处理
    logits = model(features) # (B, T, V)
    
    # log_probs: (T, B, C)
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    loss = F.ctc_loss(log_probs, batch["gloss_ids"], 
                      batch["input_lengths"], batch["gloss_lengths"], zero_infinity=True)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Task 1.D: Decoding & WER Metric (D同学)
**Files:**
- Create: `src/student_model/inference/ctc_decoder.py`
- Create: `src/student_model/evaluation/cslt_metrics.py`

- [ ] **Step 1: Implement Greedy Search and WER**
```python
# src/student_model/inference/ctc_decoder.py
import torch

def greedy_decode(logits, blank_id=0):
    # logits: (B, T, Vocab)
    preds = torch.argmax(logits, dim=-1)
    results = []
    for i in range(preds.size(0)):
        seq = []
        prev = blank_id
        for t in range(preds.size(1)):
            p = preds[i, t].item()
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        results.append(seq)
    return results
```
(D同学独立于模型训练，使用随机生成的 `logits` 矩阵来测试解码和计算 WER 模块)。

---

## Phase 2: SLT Translation (W3 - W4)

**目标**: 引入 Transformer Decoder，跑通 `Gloss/Features -> Text`。

### Task 2.A: Text Tokenizer (A同学)
- [ ] 编写中文 `Tokenizer`，生成 `text_ids`。将文本字典加入 `CSLTDataset`。

### Task 2.B: Translation Decoder (B同学)
- [ ] 基于 `torch.nn.TransformerDecoder` 编写从 `(B, T', D)` 到 `text_logits` 的生成模型。

### Task 2.C: Joint Training (C同学)
- [ ] 编写联合损失：`Loss = CTC_Loss + lambda * CE_Loss_Text`。
- [ ] 实现 Teacher Forcing 训练策略。

### Task 2.D: BLEU/ROUGE & Beam Search (D同学)
- [ ] 引入 `torchmetrics.text` 计算 BLEU-4 和 ROUGE-L。
- [ ] 实现文本生成的 Beam Search。

---

## Phase 3: Deployment & Optimization (W5 - W6)

### Task 3.A/B: Edge Export & Quantization (A/B同学)
- [ ] 编写 `export_to_torchscript.py`，将 Encoder+Decoder 导出为 Android 可用格式。
- [ ] 对 Encoder 应用 INT8 动态量化。

### Task 3.C/D: Android Integation & Profiling (C/D同学)
- [ ] 在 `android/app` 中集成新模型。
- [ ] 测量推理延迟 (Latency) 与内存占用 (RAM)。

---
