#!/usr/bin/env python
# pylint: disable=W0201
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchlight import str2bool

from .processor import Processor


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
        self.model.apply(weights_init)
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

    def _parse_batch(self, batch):
        if len(batch) == 2:
            data, label = batch
            teacher_logits = None
            vid = None
        elif len(batch) == 4:
            data, teacher_logits, label, vid = batch
        else:
            raise ValueError(
                'Unsupported batch format. Expect 2 or 4 items, got {}'.format(len(batch))
            )
        return data, teacher_logits, label, vid

    def _to_stgcn_input(self, data):
        # Expected by ST-GCN: (N, C, T, V, M)
        if data.dim() == 5:
            return data

        # StudentDataset typically returns (N, T, F)
        if data.dim() == 3:
            n, t, f = data.shape
            if f % 3 != 0:
                raise ValueError('Flatten feature dim must be divisible by 3, got {}'.format(f))
            v = f // 3
            data = data.view(n, t, v, 3).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            return data

        raise ValueError('Unsupported data dim {} for ST-GCN input'.format(data.dim()))

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

        loss_value = []
        ce_value = []
        kd_value = []

        for batch in loader:
            data, teacher_logits, label, _ = self._parse_batch(batch)

            data = data.float().to(self.dev)
            data = self._to_stgcn_input(data)
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)

            output = self.model(data)
            loss, ce, kd = self._kd_loss(output, teacher_logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.item()
            self.iter_info['ce_loss'] = ce.item()
            self.iter_info['kd_loss'] = kd.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

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
        self.model.eval()
        loader = self.data_loader['test']

        loss_value = []
        result_frag = []
        label_frag = []

        for batch in loader:
            data, teacher_logits, label, _ = self._parse_batch(batch)

            data = data.float().to(self.dev)
            data = self._to_stgcn_input(data)
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)

            with torch.no_grad():
                output = self.model(data)

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

        return parser
