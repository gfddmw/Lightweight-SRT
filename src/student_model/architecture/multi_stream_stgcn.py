# 采用 Late Fusion (Logits 加权平均)。移动端部署时，Late Fusion 可并行推理三个极小分支，且便于知识蒸馏对齐。
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.student_model.architecture.st_gcn import Model as STGCN  # 您的基础轻量 ST-GCN

class MultiStreamSTGCN(nn.Module):
    def __init__(self, num_class, in_channels, graph_args, edge_importance_weighting=True, 
                 num_person=1, **kwargs):
        super().__init__()
        
        # ✅ 关键修复：确保 graph_args 是字典且包含 num_point
        if isinstance(graph_args, dict):
            graph_config = graph_args
        else:
            graph_config = {'num_point': 21, 'strategy': 'spatial'}
        
        # 实例化三个轻量级分支
        joint_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_class', 'graph_args', 'edge_importance_weighting', 'channel_cfg']}
        bone_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_class', 'graph_args', 'edge_importance_weighting', 'channel_cfg']}
        motion_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_class', 'graph_args', 'edge_importance_weighting', 'channel_cfg']}

        if 'joint_channel_cfg' in kwargs:
            joint_kwargs['channel_cfg'] = kwargs['joint_channel_cfg']
        elif 'channel_cfg' in kwargs:
            joint_kwargs['channel_cfg'] = kwargs['channel_cfg']

        if 'bone_channel_cfg' in kwargs:
            bone_kwargs['channel_cfg'] = kwargs['bone_channel_cfg']
        elif 'channel_cfg' in kwargs:
            bone_kwargs['channel_cfg'] = kwargs['channel_cfg']

        if 'motion_channel_cfg' in kwargs:
            motion_kwargs['channel_cfg'] = kwargs['motion_channel_cfg']
        elif 'channel_cfg' in kwargs:
            motion_kwargs['channel_cfg'] = kwargs['channel_cfg']

        self.joint_net = STGCN(
            in_channels=in_channels,
            num_class=num_class,
            graph_args=graph_config,
            edge_importance_weighting=edge_importance_weighting,
            num_person=num_person,
            **joint_kwargs
        )
        self.bone_net = STGCN(
            in_channels=in_channels,
            num_class=num_class,
            graph_args=graph_config,
            edge_importance_weighting=edge_importance_weighting,
            num_person=num_person,
            **bone_kwargs
        )
        self.motion_net = STGCN(
            in_channels=in_channels,
            num_class=num_class,
            graph_args=graph_config,
            edge_importance_weighting=edge_importance_weighting,
            num_person=num_person,
            **motion_kwargs
        )
        
        # 可学习的融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3))
        self.num_class = num_class

    def forward(self, joints=None, bones=None, motion=None, data=None):
        # 单流兼容
        if data is not None:
            return self.joint_net(data)
        
        # 多流前向
        out_j = self.joint_net(joints)
        out_b = self.bone_net(bones)
        out_m = self.motion_net(motion)
        
        # Late Fusion
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * out_j + weights[1] * out_b + weights[2] * out_m
        
        return fused, (out_j, out_b, out_m)  # 返回分支 logits 用于辅助 Loss

    def extract_feature(self, joints=None, bones=None, motion=None, data=None):
        if data is not None:
            return self.joint_net.extract_feature(data)
        
        # 多流特征提取
        out_j, feat_j = self.joint_net.extract_feature(joints)
        out_b, feat_b = self.bone_net.extract_feature(bones)
        out_m, feat_m = self.motion_net.extract_feature(motion)
        
        # 使用与 forward 相同的 Fusion 权重用于 Logits
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_output = weights[0] * out_j + weights[1] * out_b + weights[2] * out_m
        
        # 【重要改进】特征蒸馏使用拼接特征，避免 Late Fusion 导致的特征消融
        # feat_j shape: (N, C, T, V, M)
        concatenated_feature = torch.cat([feat_j, feat_b, feat_m], dim=1)
        
        return fused_output, concatenated_feature, (out_j, out_b, out_m)
