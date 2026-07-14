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
        aligned[:, :, :self.teacher_vocab_size] = teacher_logits[:, :, :self.teacher_vocab_size]
        aligned[:, :, self.student_blank] = teacher_logits[:, :, self.teacher_blank]
        return aligned

    def _align_temporal(self, student_logits, teacher_logits, student_lengths, teacher_lengths):
        """处理时序长度不匹配。

        学生输出的时间步 T_out 与教师的时间步 T_t 可能不同。
        策略：将教师 logits 在时间维度上线性插值到学生的时间步数。
        """
        T_out = student_logits.size(1)
        T_t = teacher_logits.size(1)

        if T_t == T_out:
            return teacher_logits

        teacher_logits = teacher_logits.permute(0, 2, 1)
        teacher_logits = F.interpolate(teacher_logits, size=T_out, mode='linear', align_corners=False)
        teacher_logits = teacher_logits.permute(0, 2, 1)
        return teacher_logits

    def _build_mask(self, student_lengths, student_logits, teacher_lengths):
        """构建时序 padding mask。

        同时考虑学生和教师的实际长度，取两者较小值作为有效区域。
        返回 [B, T_out] 的 bool mask，True 表示有效。
        """
        B, T_out, _ = student_logits.shape
        device = student_logits.device

        student_mask = torch.arange(T_out, device=device).unsqueeze(0) < student_lengths.unsqueeze(1)
        teacher_mask = torch.arange(T_out, device=device).unsqueeze(0) < teacher_lengths.unsqueeze(1)
        mask = student_mask & teacher_mask
        return mask

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
        teacher_aligned = self._align_vocabulary(teacher_logits)
        teacher_aligned = self._align_temporal(
            student_logits, teacher_aligned, student_lengths, teacher_lengths
        )

        mask = self._build_mask(student_lengths, student_logits, teacher_lengths)

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_aligned / self.temperature, dim=-1).detach()

        kl_per_token = F.kl_div(
            student_log_probs, teacher_probs,
            reduction='none', log_target=False
        ).sum(dim=-1)

        kl_per_token = kl_per_token * mask
        loss = kl_per_token.sum() / mask.sum().clamp(min=1)
        loss = loss * (self.temperature ** 2)

        return loss


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
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)

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

        log_probs = F.log_softmax(student_logits, dim=-1).transpose(0, 1)
        ctc_val = self.ctc_loss(log_probs, gloss_ids, student_lengths, gloss_lengths)
        losses['ctc'] = ctc_val
        total = self.ctc_weight * ctc_val

        if self.kd_weight > 0:
            kd_val = self.kd_loss(student_logits, teacher_logits, student_lengths, teacher_lengths)
            losses['kd'] = kd_val
            total = total + self.kd_weight * kd_val

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