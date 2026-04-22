#!/usr/bin/env python
# pylint: disable=W0201
import os
import sys
import argparse
import numpy as np

# Add project root and student model root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'student_model'))

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchlight import str2bool

from distillation.processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_KD_Processor(Processor):
    """
    Recognition processor with optional KD loss.

    Supports two batch formats:
    - (data, label)
    - (data, teacher_logits, label, video_id)

    If teacher logits are provided, total loss is:
      (1-alpha) * CE + alpha * KL(student/T, teacher/T) * T^2
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.ce_loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    # 在 REC_KD_Processor 类中
    def _parse_batch(self, batch):
        """
        支持两种格式:
        1. 单流: (data, label) 或 (data, teacher_logits, label, vid)
        2. 多流: ({"joints":..., "bones":..., "motion":...}, label) 
                或 (dict, teacher_logits, label, vid)
        """
        # 解包基础字段
        if len(batch) == 2:
            data_or_dict, label = batch
            teacher_logits = None
            vid = None
        elif len(batch) == 4:
            data_or_dict, teacher_logits, label, vid = batch
        else:
            raise ValueError(f'Unsupported batch format, got {len(batch)} items')

        # 判断是否为多流字典输入
        if isinstance(data_or_dict, dict):
            # 多流模式: 对每个流做格式转换
            streams = {}
            for k in ['joints', 'bones', 'motion']:
                if k in data_or_dict:
                    streams[k] = self._to_stgcn_input(data_or_dict[k])
            return streams, teacher_logits, label, vid
        else:
            # 单流模式: 保持原有逻辑
            data = self._to_stgcn_input(data_or_dict)
            return data, teacher_logits, label, vid

    def _to_stgcn_input(self, data):
        """
        将 (N, T, F) 或 (N, C, T, V) 转换为 ST-GCN 需要的 (N, C, T, V, M)
        M=1 (单人员), 保持向后兼容
        """
        if data.dim() == 5:
            return data  # 已经是目标格式
        
        # 处理展平的 (N, T, F) 格式
        if data.dim() == 3:
            n, t, f = data.shape
            if f % 3 != 0 and f % 2 != 0:
                # 尝试直接作为 (N, T, V*C) 处理，C=2 或 3
                raise ValueError(f'Feature dim {f} not divisible by 2 or 3')
            
            # 假设 C=2 (x,y)，则 V = f // 2
            c = 2 if f % 2 == 0 else 3
            v = f // c
            # (N, T, V, C) -> (N, C, T, V) -> (N, C, T, V, 1)
            data = data.view(n, t, v, c).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            return data
        
        # 处理已经是 (N, C, T, V) 的情况
        if data.dim() == 4:
            return data.unsqueeze(-1)
            
        raise ValueError(f'Unsupported data dim {data.dim()} for ST-GCN input')
    def _kd_loss(self, student_logits, teacher_logits, label):
        ce = self.ce_loss(student_logits, label)
        if teacher_logits is None:
            return ce, ce.detach(), torch.tensor(0.0, device=ce.device)

        t = self.arg.kd_temperature
        alpha = self.arg.kd_alpha

        student_log_prob = F.log_softmax(student_logits / t, dim=1)
        teacher_prob = F.softmax(teacher_logits / t, dim=1)
        kd = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (t * t)
        loss = (1.0 - alpha) * ce + alpha * kd
        return loss, ce.detach(), kd.detach()

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        
        loss_value, ce_value, kd_value = [], [], []
        
        for batch in loader:
            # 解析 batch (可能返回 dict 或 tensor)
            data_input, teacher_logits, label, _ = self._parse_batch(batch)
            
            # 处理多流/单流输入
            if isinstance(data_input, dict):
                # 多流: 分别移动到 device
                model_kwargs = {k: v.float().to(self.dev) for k, v in data_input.items()}
            else:
                # 单流: 保持原逻辑
                model_kwargs = {'data': data_input.float().to(self.dev)}
                
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)
            
            # 前向传播 (支持多返回值)
            output = self.model(**model_kwargs)
            if isinstance(output, tuple):
                fused_logits, branch_logits = output  # Late Fusion 模式
            else:
                fused_logits, branch_logits = output, None  # 单流模式
            
            # 计算 Loss (主 Loss + 可选辅助 Loss)
            loss, ce, kd = self._kd_loss(fused_logits, teacher_logits, label)
            bn_sparse = torch.tensor(0.0, device=loss.device)
            if self.arg.bn_l1_lambda > 0 and hasattr(self.model, 'bn_l1_loss'):
                bn_sparse = self.arg.bn_l1_lambda * self.model.bn_l1_loss()
                loss = loss + bn_sparse
            
            # 可选: 添加分支辅助 CE Loss (提升多流特征对齐)
            if branch_logits is not None and self.arg.use_aux_loss:
                aux_ce = sum(F.cross_entropy(bl, label) for bl in branch_logits) / len(branch_logits)
                loss = loss + self.arg.aux_loss_weight * aux_ce
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录日志
            self.iter_info.update({
                'loss': loss.item(),
                'ce_loss': ce.item(),
                'kd_loss': kd.item(),
                'bn_sparse': bn_sparse.item(),
                'lr': f'{self.lr:.6f}'
            })
            if branch_logits is not None:
                self.iter_info['aux_ce'] = aux_ce.item() if self.arg.use_aux_loss else 0.0
                
            loss_value.append(loss.item())
            ce_value.append(ce.item())
            kd_value.append(kd.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_ce_loss'] = np.mean(ce_value)
        self.epoch_info['mean_kd_loss'] = np.mean(kd_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        if 'test' not in self.data_loader:
            self.data_loader['test'] = self.data_loader['train']
            self.io.print_log("[WARN] No test dataset found, using train dataset for evaluation.")

        self.model.eval()
        loader = self.data_loader['test']

        loss_value = []
        result_frag = []
        label_frag = []

        for batch in loader:
            data_input, teacher_logits, label, _ = self._parse_batch(batch)

            # 🚀 修复：兼容多流(dict)和单流输入，与 train 方法保持一致
            if isinstance(data_input, dict):
                model_kwargs = {k: v.float().to(self.dev) for k, v in data_input.items()}
            else:
                model_kwargs = {'data': data_input.float().to(self.dev)}
                
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)

            with torch.no_grad():
                output = self.model(**model_kwargs)
                # 如果是多流返回元组，只取融合后的 logits
                if isinstance(output, tuple):
                    output = output[0]

            result_frag.append(output.data.cpu().numpy())

            if evaluation:
                loss, _, _ = self._kd_loss(output, teacher_logits, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()
            for k in self.arg.show_topk:
                self.show_topk(k)
    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network with KD')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        parser.add_argument('--kd_alpha', type=float, default=0.5, help='KD weighting coefficient')
        parser.add_argument('--kd_temperature', type=float, default=4.0, help='KD temperature')
        parser.add_argument('--use_aux_loss', type=str2bool, default=False, 
                            help='Use auxiliary CE loss for each stream branch')
        parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                            help='Weight for auxiliary branch loss')
        parser.add_argument('--bn_l1_lambda', type=float, default=0.0,
                    help='L1 regularization factor for BN gamma in network slimming')
        
        parser.add_argument('--train_feeder', type=str, default='feeder.feeder', help='train data loader class')
        parser.add_argument('--multi_stream', type=str2bool, default=False, help='use multi-stream input')
        parser.add_argument('--stream_names', type=str, default='joints,bones,motion', help='stream names')
        return parser

if __name__ == '__main__':
    p = REC_KD_Processor()
    p.start()
