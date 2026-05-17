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


class FeatureAdapter(nn.Module):
    """
    Non-linear Multi-Layer Projector for cross-modal feature alignment.
    Bridges the gap between Skeleton (Student) and RGB (Teacher) spaces.
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, student_dim * 2),
            nn.LayerNorm(student_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(student_dim * 2, teacher_dim)
        )

    def forward(self, x):
        # x: (N, C)
        return self.net(x)


class SimilarityPreservingLoss(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation (SP).
    Ensures that the relative distances between samples in the student's 
    feature space match those in the teacher's space.
    """
    def __init__(self):
        super().__init__()

    def forward(self, f_s, f_t):
        # f_s: (N, C_s), f_t: (N, C_t)
        # 1. Compute Gram Matrices (N, N)
        g_s = torch.mm(f_s, f_s.t())
        g_t = torch.mm(f_t, f_t.t())
        
        # 2. Row-wise L2 Normalization
        g_s = F.normalize(g_s, p=2, dim=1)
        g_t = F.normalize(g_t, p=2, dim=1)
        
        # 3. Frobenius Norm of difference
        return torch.norm(g_s - g_t, p='fro') ** 2 / f_s.size(0)


class REC_MULTISTREAM_KD_HINT_Processor(Processor):
    """
    Advanced Multi-Stream Distillation Processor with SP Loss and MLP Adapters.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(weights_init)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.hint_loss = nn.MSELoss()
        self.sp_loss = SimilarityPreservingLoss()
        
        # Enhanced Adapter
        self.adapter = FeatureAdapter(
            student_dim=self.arg.student_feature_dim,
            teacher_dim=self.arg.teacher_feature_dim
        )
        self.adapter.apply(weights_init)

    def gpu(self):
        super().gpu()
        self.adapter = self.adapter.to(self.dev)

    def load_optimizer(self):
        # Include adapter parameters in optimization
        parameters = list(self.model.parameters()) + list(self.adapter.parameters())
        
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
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
        """
        Supports Multi-Stream and Single-Stream formats with Teacher Logits and Features.
        """
        if len(batch) == 2:
            data_or_dict, label = batch
            teacher_logits = None
            teacher_features = None
            vid = None
        elif len(batch) == 4:
            data_or_dict, teacher_logits, label, vid = batch
            teacher_features = None
        elif len(batch) == 5:
            data_or_dict, teacher_logits, teacher_features, label, vid = batch
        else:
            raise ValueError(f'Unsupported batch format, got {len(batch)} items')

        if isinstance(data_or_dict, dict):
            # Multi-stream mode
            streams = {}
            for k in ['joints', 'bones', 'motion']:
                if k in data_or_dict:
                    streams[k] = self._to_stgcn_input(data_or_dict[k])
            return streams, teacher_logits, teacher_features, label, vid
        else:
            # Single-stream mode
            data = self._to_stgcn_input(data_or_dict)
            return data, teacher_logits, teacher_features, label, vid

    def _to_stgcn_input(self, data):
        if data.dim() == 5:
            return data
        if data.dim() == 3:
            n, t, f = data.shape
            c = 2 if f % 2 == 0 else 3
            v = f // c
            return data.view(n, t, v, c).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        if data.dim() == 4:
            return data.unsqueeze(-1)
        raise ValueError(f'Unsupported data dim {data.dim()}')

    def _extract_student_features(self, data_input):
        """
        Extract concatenated features and logits from student model.
        Returns:
            logits: (N, num_class)
            pooled_feature: (N, C_total) where C_total = 3 * C_branch (e.g. 768)
            branch_logits: list of (N, num_class)
        """
        if isinstance(data_input, dict):
            model_kwargs = {k: v.float().to(self.dev) for k, v in data_input.items()}
            output, feature, raw_branches = self.model.extract_feature(**model_kwargs)
            branch_logits = [b.mean(dim=(2, 3, 4)) for b in raw_branches]
        else:
            data = data_input.float().to(self.dev)
            output, feature = self.model.extract_feature(data)
            branch_logits = None

        logits = output.mean(dim=(2, 3, 4))
        
        # 使用拼接后的全局平均池化特征进行蒸馏
        # feature shape: (N, C_branch*3, T, V, M)
        pooled_feature = feature.mean(dim=(2, 3, 4))
        
        return logits, pooled_feature, branch_logits

    def _get_dynamic_alpha(self):
        progress = float(self.meta_info['epoch']) / max(float(self.arg.num_epoch), 1.0)
        return self.arg.kd_alpha * max(0.0, 1.0 - progress)

    def _compute_hint_loss(self, student_feature, teacher_feature):
        """
        Aligned Hint Loss for concatenated features.
        student_feature: (N, C_student_total)
        teacher_feature: (N, C_teacher)
        """
        # Ensure teacher feature is (N, C)
        if teacher_feature.dim() == 3:
            teacher_feature = teacher_feature.mean(dim=1)
        elif teacher_feature.dim() == 1:
            teacher_feature = teacher_feature.unsqueeze(0)
            
        # 1. Non-linear Feature Alignment (MSE)
        adapted_s = self.adapter(student_feature)
        mse = self.hint_loss(adapted_s, teacher_feature)
        
        # 2. Similarity Preserving (SP) - Space-invariant
        sp = self.sp_loss(student_feature, teacher_feature)
            
        return mse + self.arg.sp_weight * sp

    def _compute_total_loss(self, logits, branch_logits, teacher_logits, teacher_features, student_feature, label):
        ce = self.ce_loss(logits, label)
        
        current_epoch = self.meta_info['epoch']
        if current_epoch < self.arg.warmup_epoch:
            zero = torch.tensor(0.0).to(self.dev)
            loss = ce
            aux_ce_val = 0.0
            if branch_logits is not None and self.arg.use_aux_loss:
                aux_ce = sum(F.cross_entropy(bl, label) for bl in branch_logits) / len(branch_logits)
                loss = loss + self.arg.aux_loss_weight * aux_ce
                aux_ce_val = aux_ce.item()
            return loss, ce.detach(), zero, zero, 0.0, aux_ce_val

        if teacher_logits is None:
            zero = torch.tensor(0.0).to(self.dev)
            return ce, ce.detach(), zero, zero, 0.0, 0.0
            
        t = self.arg.kd_temperature
        alpha = self._get_dynamic_alpha() if self.arg.dynamic_alpha else self.arg.kd_alpha
        
        # Logits KD
        student_log_prob = F.log_softmax(logits / t, dim=1)
        teacher_prob = F.softmax(teacher_logits / t, dim=1)
        kd = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (t * t)
        
        # Advanced HINT Loss (MSE + SP)
        hint = torch.tensor(0.0).to(self.dev)
        if teacher_features is not None:
            hint = self._compute_hint_loss(student_feature, teacher_features)
            
        loss = (1.0 - alpha) * ce + alpha * kd + self.arg.hint_weight * hint
        
        # Auxiliary Branch Loss
        aux_ce_val = 0.0
        if branch_logits is not None and self.arg.use_aux_loss:
            aux_ce = sum(F.cross_entropy(bl, label) for bl in branch_logits) / len(branch_logits)
            loss = loss + self.arg.aux_loss_weight * aux_ce
            aux_ce_val = aux_ce.item()
            
        return loss, ce.detach(), kd.detach(), hint.detach(), alpha, aux_ce_val

    def train(self):
        self.model.train()
        self.adapter.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        
        metrics = {'loss': [], 'ce': [], 'kd': [], 'hint': [], 'aux': []}
        
        for batch in loader:
            data_input, t_logits, t_feats, label, _ = self._parse_batch(batch)
            
            label = label.long().to(self.dev)
            if t_logits is not None:
                t_logits = t_logits.float().to(self.dev)
            if t_feats is not None:
                t_feats = t_feats.float().to(self.dev)
                
            # Forward & Feature Extraction
            s_logits, s_feats, b_logits = self._extract_student_features(data_input)
            
            # Loss Computation
            loss, ce, kd, hint, alpha, aux = self._compute_total_loss(
                s_logits, b_logits, t_logits, t_feats, s_feats, label
            )
            
            # BN Sparsity
            bn_sparse = torch.tensor(0.0).to(self.dev)
            if self.arg.bn_l1_lambda > 0 and hasattr(self.model, 'bn_l1_loss'):
                bn_sparse = self.arg.bn_l1_lambda * self.model.bn_l1_loss()
                loss = loss + bn_sparse
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            self.iter_info.update({
                'loss': loss.item(),
                'ce': ce.item(),
                'kd': kd.item(),
                'hint': hint.item(),
                'aux': aux,
                'alpha': alpha,
                'lr': f'{self.lr:.6f}'
            })
            
            metrics['loss'].append(loss.item())
            metrics['ce'].append(ce.item())
            metrics['kd'].append(kd.item())
            metrics['hint'].append(hint.item())
            metrics['aux'].append(aux)
            
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info.update({
            'mean_loss': np.mean(metrics['loss']),
            'mean_ce': np.mean(metrics['ce']),
            'mean_kd': np.mean(metrics['kd']),
            'mean_hint': np.mean(metrics['hint']),
            'mean_aux': np.mean(metrics['aux'])
        })
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        self.adapter.eval()
        loader = self.data_loader.get('test', self.data_loader.get('train'))

        loss_value = []
        result_frag = []
        label_frag = []

        for batch in loader:
            data_input, t_logits, t_feats, label, _ = self._parse_batch(batch)
            label = label.long().to(self.dev)
            if t_logits is not None:
                t_logits = t_logits.float().to(self.dev)
            if t_feats is not None:
                t_feats = t_feats.float().to(self.dev)

            with torch.no_grad():
                s_logits, s_feats, b_logits = self._extract_student_features(data_input)
                
                if evaluation:
                    loss, _, _, _, _, _ = self._compute_total_loss(
                        s_logits, b_logits, t_logits, t_feats, s_feats, label
                    )
                    loss_value.append(loss.item())
                    label_frag.append(label.data.cpu().numpy())

            result_frag.append(s_logits.data.cpu().numpy())

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
            description='Multi-Stream ST-GCN with Advanced Logit KD + HINT + SP Loss')

        # Training Params
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='Top K accuracy')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='LR decay steps')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
        parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epochs with CE only')

        # KD Params
        parser.add_argument('--kd_alpha', type=float, default=0.5, help='Logit KD alpha')
        parser.add_argument('--kd_temperature', type=float, default=4.0, help='KD temperature')
        parser.add_argument('--dynamic_alpha', type=str2bool, default=True, help='use dynamic alpha decay')
        
        # HINT/SP Params
        parser.add_argument('--hint_weight', type=float, default=0.1, help='HINT loss weight')
        parser.add_argument('--sp_weight', type=float, default=10.0, help='Similarity Preserving loss weight')
        parser.add_argument('--student_feature_dim', type=int, default=256, help='student feature dim')
        parser.add_argument('--teacher_feature_dim', type=int, default=1024, help='teacher feature dim')
        parser.add_argument('--hint_layer_weights', type=float, default=[1.0, 0.5], nargs='+', help='weights for hint layers')
        
        # Multi-Stream Aux Loss
        parser.add_argument('--use_aux_loss', type=str2bool, default=False, help='use branch aux CE loss')
        parser.add_argument('--aux_loss_weight', type=float, default=0.1, help='weight for aux loss')
        
        # Regularization
        parser.add_argument('--bn_l1_lambda', type=float, default=0.0, help='L1 for BN gamma')

        parser.add_argument('--train_feeder', type=str, default='feeder.feeder', help='train data loader')
        return parser

if __name__ == '__main__':
    p = REC_MULTISTREAM_KD_HINT_Processor()
    p.start()
