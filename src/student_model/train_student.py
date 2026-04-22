#!/usr/bin/env python
# pylint: disable=W0201
import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path

# 🔧 路径修复
CURRENT_FILE = Path(__file__).resolve()
MODEL_ROOT = CURRENT_FILE.parents[2]
STUDENT_ROOT = CURRENT_FILE.parents[1]
if str(MODEL_ROOT) not in sys.path: sys.path.insert(0, str(MODEL_ROOT))
if str(STUDENT_ROOT) not in sys.path: sys.path.insert(0, str(STUDENT_ROOT))

# 🔧 Windows 编码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchlight
from torchlight import str2bool, DictAction, import_class
from distillation.processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    def __init__(self, argv=None):
        # 1. 解析命令行获取 --config
        parser = self.get_parser()
        if argv is None: argv = sys.argv[1:]
        self.arg = parser.parse_args(argv)

        # 2. 🔧 强制加载 YAML 并注入参数
        if getattr(self.arg, 'config', None):
            cfg_path = Path(self.arg.config)
            if not cfg_path.exists(): cfg_path = Path(os.getcwd()) / self.arg.config
            print(f"📂 加载配置: {cfg_path}")
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg_dict = yaml.safe_load(f)
                if cfg_dict:
                    for k, v in cfg_dict.items():
                        setattr(self.arg, k, v)
                    print(f"✅ 配置应用成功 | model={self.arg.model}")

        if not getattr(self.arg, 'model', None):
            print("❌ 错误: 'model' 未定义"); sys.exit(1)
            
        self.data_loader = {}
        
        # 3. 调用父类初始化 (传入空列表防止重复解析)
        super().__init__(argv=[])

    def load_arg(self, argv=None):
        """
        🔑 核心修复：覆盖父类 load_arg，跳过 YAML 二次校验
        因为我们已经在 __init__ 中完成了配置加载和参数注入
        """
        # 如果 self.arg 已存在且包含必要字段，直接使用
        if hasattr(self, 'arg') and getattr(self.arg, 'model', None):
            print("✅ 跳过重复配置加载，使用已注入的参数")
            return
        # 否则调用父类逻辑（兜底）
        super().load_arg(argv)

    def load_data(self):
        """加载数据集 (支持多流/单流)"""
        from src.student_model.dataset.multi_stream_dataset import MultiStreamSkeletonDataset
        
        def fix_path(rel_path, fallback_prefix="Lightweight-SRT-main"):
            """如果相对路径找不到文件，尝试添加前缀"""
            p = Path(rel_path)
            if p.exists():
                return rel_path
            fixed = Path(fallback_prefix) / rel_path
            if fixed.exists():
                print(f"✅ 自动修正路径: {rel_path} → {fixed}")
                return str(fixed)
            return rel_path  # 兜底返回原路径，让 Dataset 报明确错误
        
        index_path = fix_path(getattr(self.arg, 'index_path', 'processed/clean_indices.json'))
        data_dir = fix_path(getattr(self.arg, 'data_dir', 'processed/skeletons'))
        
        train_dataset = MultiStreamSkeletonDataset(
            index_path=index_path,
            data_dir=data_dir,
            topology=getattr(self.arg, 'topology', []),
            target_len=getattr(self.arg, 'target_len', 64),
            is_train=True,
            normalize_wrist=True
        )
            
        self.data_loader['train'] = DataLoader(
            train_dataset,
            batch_size=getattr(self.arg, 'batch_size', 16),
            shuffle=True,
            num_workers=getattr(self.arg, 'num_worker', 4),
            drop_last=True
        )
        
        self.data_loader['test'] = DataLoader(
            train_dataset,
            batch_size=getattr(self.arg, 'test_batch_size', 16),
            shuffle=False,
            num_workers=2
        )

    def load_model(self):
        model_args = getattr(self.arg, 'model_args', {}) or {}
        self.model = self.io.load_model(self.arg.model, **model_args)
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=getattr(self.arg, 'nesterov', True),
                weight_decay=getattr(self.arg, 'weight_decay', 0.0001))
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=getattr(self.arg, 'weight_decay', 0.0001))
        else:
            raise ValueError(f'Unsupported optimizer: {self.arg.optimizer}')

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and getattr(self.arg, 'step', None):
            lr = self.arg.base_lr * (0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        acc = 100 * sum(hit_top_k) / len(hit_top_k) if hit_top_k else 0
        self.io.print_log(f'\tTop{k}: {acc:.2f}%')

    def _to_stgcn_input(self, data):
        if data.dim() == 5: return data
        if data.dim() == 4: return data.unsqueeze(-1)
        if data.dim() == 3:
            n, t, f = data.shape
            c, v = (2, f//2) if f % 2 == 0 else (3, f//3)
            return data.view(n, t, v, c).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        raise ValueError(f'Unsupported data dim {data.dim()}')

    def _parse_batch_for_multistream(self, batch):
        data_dict, label = batch
        model_kwargs = {}
        for s in getattr(self.arg, 'stream_names', 'joints,bones,motion').split(','):
            if s in data_dict:
                model_kwargs[s] = data_dict[s].float().to(self.dev)
        return model_kwargs, label.long().to(self.dev)

    def _parse_batch_for_singlestream(self, batch):
        data, label = batch
        return {'data': self._to_stgcn_input(data.float().to(self.dev))}, label.long().to(self.dev)

    def _compute_loss(self, output, label):
        fused_logits, branch_logits = (output, None) if not isinstance(output, tuple) else output
        loss = self.loss(fused_logits, label)
        if getattr(self.arg, 'multi_stream', False) and getattr(self.arg, 'use_aux_loss', False) and branch_logits:
            aux = sum(self.loss(bl, label) for bl in branch_logits) / len(branch_logits)
            loss += getattr(self.arg, 'aux_loss_weight', 0.1) * aux
            self.iter_info['aux_ce'] = aux.item()
        return loss, fused_logits

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_val = []

        for batch in loader:
            kwargs, label = self._parse_batch_for_multistream(batch) if self.arg.multi_stream else self._parse_batch_for_singlestream(batch)
            loss, _ = self._compute_loss(self.model(**kwargs), label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter_info.update({'loss': loss.item(), 'lr': f'{self.lr:.6f}'})
            loss_val.append(loss.item())
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_val)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loss_val, res_frag, lab_frag = [], [], []

        for batch in loader:
            kwargs, label = self._parse_batch_for_multistream(batch) if self.arg.multi_stream else self._parse_batch_for_singlestream(batch)
            with torch.no_grad():
                out = self.model(**kwargs)
                out = out[0] if isinstance(out, tuple) else out
            res_frag.append(out.cpu().numpy())
            if evaluation:
                loss_val.append(self.loss(out, label).item())
                lab_frag.append(label.cpu().numpy())

        self.result = np.concatenate(res_frag)
        if evaluation:
            self.label = np.concatenate(lab_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_val)
            self.show_epoch_info()
            for k in getattr(self.arg, 'show_topk', [1, 5]):
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):
        """仅注册本项目新增参数，标准参数由父类提供"""
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help, parents=[parent_parser], description='ST-GCN Training')

        # 仅添加多流/蒸馏专属参数
        parser.add_argument('--multi_stream', type=str2bool, default=False)
        parser.add_argument('--stream_names', type=str, default='joints,bones,motion')
        parser.add_argument('--use_aux_loss', type=str2bool, default=False)
        parser.add_argument('--aux_loss_weight', type=float, default=0.1)
        
        return parser

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    REC_Processor().start()