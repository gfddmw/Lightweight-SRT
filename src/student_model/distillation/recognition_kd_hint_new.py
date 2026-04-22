#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "student_model"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchlight import str2bool
from distillation.processor import Processor


def new_weights_init(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv1d") != -1:
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif class_name.find("Conv2d") != -1:
        module.weight.data.normal_(0.0, 0.02)
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif class_name.find("BatchNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class NewFeatureAdapter(nn.Module):
    """
    1x1 Conv adapter for feature-space alignment.
    Input:  (N, C_student)
    Output: (N, C_teacher)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projector = nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.projector(x)
        return x.squeeze(-1)


class REC_KD_HINT_Processor_New(Processor):
    def new_load_model_and_losses(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model.apply(new_weights_init)
        self.ce_loss = nn.CrossEntropyLoss()
        self.hint_loss = nn.MSELoss()
        self.adapter = NewFeatureAdapter(
            student_dim=self.arg.student_feature_dim,
            teacher_dim=self.arg.teacher_feature_dim,
        )
        self.adapter.apply(new_weights_init)

    def load_model(self):
        self.new_load_model_and_losses()

    def gpu(self):
        super().gpu()
        self.adapter = self.adapter.to(self.dev)

    def new_build_trainable_parameters(self):
        return list(self.model.parameters()) + list(self.adapter.parameters())

    def load_optimizer(self):
        parameters = self.new_build_trainable_parameters()
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(parameters, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == "SGD" and self.arg.step:
            lr = self.arg.base_lr * (0.1 ** np.sum(self.meta_info["epoch"] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [label in rank[index, -k:] for index, label in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log("\tTop{}: {:.2f}%".format(k, 100 * accuracy))

    def new_parse_batch(self, batch):
        if len(batch) == 2:
            data, label = batch
            teacher_logits = None
            teacher_features = None
            vid = None
        elif len(batch) == 4:
            data, teacher_logits, label, vid = batch
            teacher_features = None
        elif len(batch) == 5:
            data, teacher_logits, teacher_features, label, vid = batch
        else:
            raise ValueError("Unsupported batch format. Expect 2, 4 or 5 items, got {}".format(len(batch)))
        return data, teacher_logits, teacher_features, label, vid

    def new_to_stgcn_input(self, data):
        if data.dim() == 5:
            return data
        if data.dim() == 3:
            batch_size, timesteps, feature_dim = data.shape
            if feature_dim % 3 != 0:
                raise ValueError("Flatten feature dim must be divisible by 3, got {}".format(feature_dim))
            num_nodes = feature_dim // 3
            return data.view(batch_size, timesteps, num_nodes, 3).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        raise ValueError("Unsupported data dim {} for ST-GCN input".format(data.dim()))

    def new_extract_student_feature_vector(self, data):
        if not hasattr(self.model, "extract_feature"):
            raise AttributeError("Current student model does not expose extract_feature()")

        output, feature = self.model.extract_feature(data)
        # logits: (N, num_class, T, V, M) -> (N, num_class)
        logits = output.sum(dim=(2, 3, 4))
        # two levels for hint loss:
        # - level 1: global pooled feature
        # - level 2: temporal pooled feature then node/person pooling
        global_feature = feature.mean(dim=(2, 3, 4))
        temporal_feature = feature.mean(dim=2).mean(dim=(2, 3))
        return logits, [global_feature, temporal_feature]

    def new_get_dynamic_alpha(self):
        progress = float(self.meta_info["epoch"]) / max(float(self.arg.num_epoch), 1.0)
        return self.arg.kd_alpha * max(0.0, 1.0 - progress)

    def new_compute_hint_loss(self, student_feature_vectors, teacher_features):
        if teacher_features.dim() == 2:
            teacher_features = teacher_features.unsqueeze(1)

        hint_losses = []
        hint_weights = self.arg.hint_layer_weights
        if len(hint_weights) == 0:
            hint_weights = [1.0] * len(student_feature_vectors)

        if len(hint_weights) < len(student_feature_vectors):
            hint_weights = hint_weights + [hint_weights[-1]] * (len(student_feature_vectors) - len(hint_weights))
        elif len(hint_weights) > len(student_feature_vectors):
            hint_weights = hint_weights[: len(student_feature_vectors)]

        for layer_index, student_feature in enumerate(student_feature_vectors):
            adapted_student_feature = self.adapter(student_feature)
            teacher_index = min(layer_index, teacher_features.size(1) - 1)
            teacher_feature = teacher_features[:, teacher_index, :]
            hint_losses.append(hint_weights[layer_index] * self.hint_loss(adapted_student_feature, teacher_feature))

        return sum(hint_losses) / max(len(hint_losses), 1)

    def new_compute_total_loss(self, student_logits, teacher_logits, teacher_features, student_feature_vectors, label):
        ce = self.ce_loss(student_logits, label)

        if teacher_logits is None:
            zero = torch.tensor(0.0, device=ce.device)
            return ce, ce.detach(), zero, zero, 0.0

        temperature = self.arg.kd_temperature
        dynamic_alpha = self.new_get_dynamic_alpha()

        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)

        hint = torch.tensor(0.0, device=ce.device)
        if teacher_features is not None:
            hint = self.new_compute_hint_loss(student_feature_vectors, teacher_features)

        loss = (1.0 - dynamic_alpha) * ce + dynamic_alpha * kd + self.arg.hint_weight * hint
        return loss, ce.detach(), kd.detach(), hint.detach(), dynamic_alpha

    def train(self):
        self.model.train()
        self.adapter.train()
        self.adjust_lr()
        loader = self.data_loader["train"]

        loss_value = []
        ce_value = []
        kd_value = []
        hint_value = []

        for batch in loader:
            data, teacher_logits, teacher_features, label, _ = self.new_parse_batch(batch)

            data = data.float().to(self.dev)
            data = self.new_to_stgcn_input(data)
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)
            if teacher_features is not None:
                teacher_features = teacher_features.float().to(self.dev)

            student_logits, student_feature_vectors = self.new_extract_student_feature_vector(data)
            loss, ce, kd, hint, dynamic_alpha = self.new_compute_total_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                teacher_features=teacher_features,
                student_feature_vectors=student_feature_vectors,
                label=label,
            )

            bn_sparse = torch.tensor(0.0, device=loss.device)
            if self.arg.bn_l1_lambda > 0 and hasattr(self.model, "bn_l1_loss"):
                bn_sparse = self.arg.bn_l1_lambda * self.model.bn_l1_loss()
                loss = loss + bn_sparse

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info["loss"] = loss.item()
            self.iter_info["ce_loss"] = ce.item()
            self.iter_info["kd_loss"] = kd.item()
            self.iter_info["hint_loss"] = hint.item()
            self.iter_info["dynamic_alpha"] = dynamic_alpha
            self.iter_info["bn_sparse"] = bn_sparse.item()
            self.iter_info["lr"] = "{:.6f}".format(self.lr)

            loss_value.append(loss.item())
            ce_value.append(ce.item())
            kd_value.append(kd.item())
            hint_value.append(hint.item())

            self.show_iter_info()
            self.meta_info["iter"] += 1

        self.epoch_info["mean_loss"] = np.mean(loss_value)
        self.epoch_info["mean_ce_loss"] = np.mean(ce_value)
        self.epoch_info["mean_kd_loss"] = np.mean(kd_value)
        self.epoch_info["mean_hint_loss"] = np.mean(hint_value)
        self.epoch_info["dynamic_alpha"] = self.new_get_dynamic_alpha()
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        self.adapter.eval()
        loader = self.data_loader["test"]

        loss_value = []
        result_frag = []
        label_frag = []

        for batch in loader:
            data, teacher_logits, teacher_features, label, _ = self.new_parse_batch(batch)

            data = data.float().to(self.dev)
            data = self.new_to_stgcn_input(data)
            label = label.long().to(self.dev)
            if teacher_logits is not None:
                teacher_logits = teacher_logits.float().to(self.dev)
            if teacher_features is not None:
                teacher_features = teacher_features.float().to(self.dev)

            with torch.no_grad():
                student_logits, student_feature_vectors = self.new_extract_student_feature_vector(data)

            result_frag.append(student_logits.data.cpu().numpy())

            if evaluation:
                loss, _, _, _, _ = self.new_compute_total_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    teacher_features=teacher_features,
                    student_feature_vectors=student_feature_vectors,
                    label=label,
                )
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info["mean_loss"] = np.mean(loss_value)
            self.show_epoch_info()
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description="Spatial Temporal Graph Convolution Network with KD + Hint Loss",
        )

        parser.add_argument("--show_topk", type=int, default=[1, 5], nargs="+")
        parser.add_argument("--base_lr", type=float, default=0.01)
        parser.add_argument("--step", type=int, default=[], nargs="+")
        parser.add_argument("--optimizer", default="SGD")
        parser.add_argument("--nesterov", type=str2bool, default=True)
        parser.add_argument("--weight_decay", type=float, default=0.0001)

        parser.add_argument("--kd_alpha", type=float, default=0.5)
        parser.add_argument("--kd_temperature", type=float, default=4.0)
        parser.add_argument("--hint_weight", type=float, default=0.1)
        parser.add_argument("--student_feature_dim", type=int, default=256)
        parser.add_argument("--teacher_feature_dim", type=int, default=1024)
        parser.add_argument("--adapter_hidden_dim", type=int, default=512)
        parser.add_argument("--hint_layer_weights", type=float, default=[1.0, 0.5], nargs="+")
        parser.add_argument("--bn_l1_lambda", type=float, default=0.0)

        return parser


if __name__ == "__main__":
    processor = REC_KD_HINT_Processor_New()
    processor.start()
