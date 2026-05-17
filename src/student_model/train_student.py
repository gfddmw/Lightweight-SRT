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
        # 1. 获取解析器并解析参数
        parser = self.get_parser()
        if argv is None:
            argv = sys.argv[1:]

        # 2. 先解析命令行参数
        self.arg = parser.parse_args(argv)

        # 3. 加载配置文件（如果存在）
        if getattr(self.arg, 'config', None):
            cfg_path = Path(self.arg.config)
            if not cfg_path.exists():
                cfg_path = Path.cwd() / self.arg.config
            if cfg_path.exists():
                print(f"📂 加载配置: {cfg_path}")
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg_dict = yaml.safe_load(f)
                # 配置文件参数覆盖命令行（但保留命令行显式指定的）
                for k, v in cfg_dict.items():
                    if not hasattr(self.arg, k) or getattr(self.arg, k) == parser.get_default(k):
                        setattr(self.arg, k, v)
                print(f"✅ 配置应用成功 | model={self.arg.model}")

        # 4. 验证必要参数
        if not getattr(self.arg, 'model', None):
            print("❌ 错误: 'model' 未定义")
            sys.exit(1)

        os.makedirs(self.arg.work_dir, exist_ok=True)
        print(f"📁 工作目录: {self.arg.work_dir}")

        # 5. 初始化数据加载器和其他属性
        self.data_loader = {}
        self.dev = torch.device(f"cuda:{self.arg.device[0]}" if self.arg.device and torch.cuda.is_available() else "cpu")

        # 🔧 修复IO初始化
        try:
            self.io = torchlight.io.IO(
                self.arg.work_dir,
                save_log=self.arg.save_log,
                print_log=self.arg.print_log
            )
        except TypeError:
            self.io = torchlight.io.IO(self.arg.work_dir)

        self.model = None
        self.optimizer = None
        self.lr = self.arg.base_lr
        self.meta_info = {'epoch': 0, 'iter': 0, 'train_loss': [], 'test_loss': []}
        self.epoch_info = {}
        self.iter_info = {}

        # 6. 加载模型、优化器和数据
        self.load_model()
        self.load_optimizer()
        self.load_data()  # 改为方法调用而不是外部函数

    def load_model(self):
        model_args = getattr(self.arg, 'model_args', {}) or {}
        self.model = self.io.load_model(self.arg.model, **model_args)
        self.model.apply(weights_init)
        self.model.to(self.dev)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=getattr(self.arg, 'weight_decay', 0.0001))
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=getattr(self.arg, 'weight_decay', 0.0001))
        else:
            raise ValueError(f'Unsupported optimizer: {self.arg.optimizer}')

    def load_data(self):
        """加载数据集 - 使用绝对路径避免重复前缀问题"""
        from src.student_model.dataset.multi_stream_dataset import MultiStreamSkeletonDataset

        # 获取当前工作目录的绝对路径
        cwd = Path.cwd()

        # 定义可能的根目录
        possible_roots = [
            cwd,
            cwd / 'Lightweight-SRT',
            Path(__file__).parent.parent.parent,  # 向上3级到项目根目录
        ]

        def resolve_path(rel_path):
            """解析文件路径，返回绝对路径"""
            rel_path = str(rel_path)

            # 如果已经是绝对路径且存在，直接返回
            abs_path = Path(rel_path)
            if abs_path.is_absolute() and abs_path.exists():
                return str(abs_path)

            # 尝试在不同的根目录下查找
            for root in possible_roots:
                test_path = root / rel_path
                if test_path.exists():
                    print(f"✅ 找到文件: {test_path}")
                    return str(test_path.absolute())

                # 尝试去掉可能的重复前缀
                if 'Lightweight-SRT' in rel_path:
                    clean_path = rel_path.replace('Lightweight-SRT/', '').replace('Lightweight-SRT\\', '')
                    test_path = root / clean_path
                    if test_path.exists():
                        print(f"✅ 找到文件（清理后）: {test_path}")
                        return str(test_path.absolute())

            # 如果都没找到，返回相对于当前目录的路径（让后续报错）
            print(f"⚠️ 警告: 找不到文件 {rel_path}")
            return str(cwd / rel_path)

        # 获取配置中的路径
        data_dir_rel = getattr(self.arg, 'data_dir', 'processed/skeletons')
        train_index_rel = getattr(self.arg, 'train_index_path', 'processed/train_indices.json')
        test_index_rel = getattr(self.arg, 'test_index_path', 'processed/test_indices.json')

        # 解析为绝对路径
        data_dir = resolve_path(data_dir_rel)
        train_index_path = resolve_path(train_index_rel)
        test_index_path = resolve_path(test_index_rel)

        # 获取拓扑结构
        topology = getattr(self.arg, 'topology', None)
        if not topology:
            # 默认MediaPipe手部拓扑（21个关键点）
            topology = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
                       [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

        print(f"📁 数据目录: {data_dir}")
        print(f"📁 训练索引: {train_index_path}")
        print(f"📁 测试索引: {test_index_path}")

        # 检查文件是否存在
        if not os.path.exists(train_index_path):
            raise FileNotFoundError(f"训练索引文件不存在: {train_index_path}")
        if not os.path.exists(test_index_path):
            raise FileNotFoundError(f"测试索引文件不存在: {test_index_path}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 训练集
        train_dataset = MultiStreamSkeletonDataset(
            index_path=train_index_path,
            data_dir=data_dir,
            topology=topology,
            target_len=getattr(self.arg, 'target_len', 64),
            is_train=True,
            normalize_wrist=True
        )

        self.data_loader['train'] = DataLoader(
            train_dataset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=getattr(self.arg, 'num_worker', 4),
            drop_last=True
        )

        # 测试集
        test_dataset = MultiStreamSkeletonDataset(
            index_path=test_index_path,
            data_dir=data_dir,
            topology=topology,
            target_len=getattr(self.arg, 'target_len', 64),
            is_train=False,
            normalize_wrist=True
        )

        self.data_loader['test'] = DataLoader(
            test_dataset,
            batch_size=getattr(self.arg, 'test_batch_size', 256),
            shuffle=False,
            num_workers=2
        )

        print(f"✅ 数据加载完成 | 训练集: {len(train_dataset)} 样本 | 测试集: {len(test_dataset)} 样本")

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
        stream_names = getattr(self.arg, 'stream_names', 'joints,bones,motion').split(',')
        for s in stream_names:
            s = s.strip()
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

    def show_iter_info(self):
        """显示迭代信息"""
        info_str = f"Epoch {self.meta_info['epoch']}, Iter {self.meta_info['iter']}: "
        info_str += f"loss={self.iter_info.get('loss', 0):.4f}, lr={self.lr:.6f}"
        self.io.print_log(info_str)

    def show_epoch_info(self):
        """显示epoch信息"""
        info_str = f"Epoch {self.meta_info['epoch']} finished: "
        info_str += f"mean_loss={self.epoch_info.get('mean_loss', 0):.4f}"
        self.io.print_log(info_str)

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_val = []

        for batch in loader:
            # 解析batch
            if getattr(self.arg, 'multi_stream', False):
                kwargs, label = self._parse_batch_for_multistream(batch)
            else:
                kwargs, label = self._parse_batch_for_singlestream(batch)

            # 前向传播
            output = self.model(**kwargs)
            loss, _ = self._compute_loss(output, label)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录信息
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
            # 解析batch
            if getattr(self.arg, 'multi_stream', False):
                kwargs, label = self._parse_batch_for_multistream(batch)
            else:
                kwargs, label = self._parse_batch_for_singlestream(batch)

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

    def start(self):
        """主训练流程"""
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.meta_info['epoch'] = epoch
            self.train()

            # 定期评估
            if (epoch + 1) % self.arg.eval_interval == 0:
                self.test(evaluation=True)

            # 定期保存模型
            if (epoch + 1) % self.arg.save_interval == 0:
                save_path = os.path.join(self.arg.work_dir, f'epoch_{epoch+1}_model.pt')
                os.makedirs(self.arg.work_dir, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                self.io.print_log(f"Model saved to {save_path}")

    @staticmethod
    def get_parser(add_help=False):
        """完整的参数解析器，包含所有参数"""
        parser = argparse.ArgumentParser(add_help=add_help, description='ST-GCN Training')

        # 基础参数
        parser.add_argument('-w', '--work_dir', default='./workdirs/tmp', help='工作目录')
        parser.add_argument('-c', '--config', default=None, help='配置文件路径')
        parser.add_argument('--phase', default='train', help='训练或测试阶段')
        parser.add_argument('--save_result', type=str2bool, default=False, help='是否保存结果')
        parser.add_argument('--start_epoch', type=int, default=0, help='起始epoch')
        parser.add_argument('--num_epoch', type=int, default=50, help='总epoch数')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='是否使用GPU')
        parser.add_argument('--device', type=int, nargs='+', default=[0], help='GPU设备ID')
        parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
        parser.add_argument('--save_interval', type=int, default=10, help='保存模型间隔')
        parser.add_argument('--eval_interval', type=int, default=5, help='评估间隔')
        parser.add_argument('--save_log', type=str2bool, default=True, help='保存日志')
        parser.add_argument('--print_log', type=str2bool, default=True, help='打印日志')

        # 数据参数
        parser.add_argument('--feeder', default='src.student_model.dataset.multi_stream_dataset.MultiStreamSkeletonDataset', help='数据加载器')
        parser.add_argument('--num_worker', type=int, default=4, help='数据加载线程数')
        parser.add_argument('--train_feeder_args', action=DictAction, default={}, help='训练数据参数')
        parser.add_argument('--test_feeder_args', action=DictAction, default={}, help='测试数据参数')
        parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
        parser.add_argument('--test_batch_size', type=int, default=256, help='测试批次大小')
        parser.add_argument('--debug', action='store_true', help='调试模式')

        # 模型参数
        parser.add_argument('--model', default=None, help='模型类路径')
        parser.add_argument('--model_args', action=DictAction, default={}, help='模型参数')
        parser.add_argument('--weights', default=None, help='预训练权重')
        parser.add_argument('--ignore_weights', type=str, nargs='+', default=[], help='忽略的权重')

        # 优化器参数
        parser.add_argument('--base_lr', type=float, default=0.01, help='基础学习率')
        parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'], help='优化器')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
        parser.add_argument('--step', type=int, nargs='+', default=[], help='学习率下降步数')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='Nesterov动量')

        # 数据路径参数
        parser.add_argument('--data_dir', type=str, default='processed/skeletons', help='数据目录')
        parser.add_argument('--train_index_path', type=str, default='processed/train_indices.json', help='训练索引文件')
        parser.add_argument('--test_index_path', type=str, default='processed/test_indices.json', help='测试索引文件')
        parser.add_argument('--topology', type=list, default=None, help='骨架拓扑结构')
        parser.add_argument('--target_len', type=int, default=64, help='目标序列长度')

        # 多流专用参数
        parser.add_argument('--multi_stream', type=str2bool, default=False, help='是否使用多流训练')
        parser.add_argument('--stream_names', type=str, default='joints,bones,motion', help='流名称，用逗号分隔')
        parser.add_argument('--use_aux_loss', type=str2bool, default=False, help='是否使用辅助损失')
        parser.add_argument('--aux_loss_weight', type=float, default=0.1, help='辅助损失权重')
        parser.add_argument('--show_topk', type=int, nargs='+', default=[1, 5], help='显示TopK准确率')

        return parser

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    processor = REC_Processor()
    processor.start()
