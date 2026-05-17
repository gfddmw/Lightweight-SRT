# 高级骨架增强：包含随机空间旋转与时间剪裁
import numpy as np
import random

class RandomRotateSkeleton:
    """
    随机空间旋转增强 (针对手语识别优化)
    🔑 核心改进：默认绕腕部(索引0)旋转，避免绕图像原点(0,0)旋转导致手部飞出合理坐标域。
    """
    def __init__(self, max_angle=15, is_3d=False, center_on_root=True):
        self.max_angle = max_angle
        self.is_3d = is_3d
        self.center_on_root = center_on_root

    def __call__(self, skeleton):
        # skeleton shape: (T, K, C)
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        rad = np.deg2rad(angle)

        # 1. 中心化：绕腕部旋转 (MediaPipe 索引0为腕部)
        root_pos = None
        if self.center_on_root and skeleton.shape[1] > 0:
            # 🔑 修复广播错误：确保 root_pos 的通道数与 skeleton 一致
            root_pos = skeleton[:, 0:1, :].copy()    # (T, 1, C)
            root_pos[..., 2:] = 0                    # 仅平移 x, y，保留 z/conf
            skeleton = skeleton - root_pos           # 平移至原点

        # 2. 旋转矩阵乘法
        if not self.is_3d:
            R = np.array([[np.cos(rad), -np.sin(rad)],
                          [np.sin(rad),  np.cos(rad)]])
            skeleton[..., :2] = skeleton[..., :2] @ R.T
        else:
            # ⚠️ 注意：MediaPipe 的 z 是相对深度，非真实物理尺度。
            # 3D旋转会扭曲深度语义，手语识别强烈建议保持 is_3d=False
            Rz = np.array([[np.cos(rad), -np.sin(rad), 0],
                           [np.sin(rad),  np.cos(rad), 0],
                           [0, 0, 1]])
            skeleton[..., :3] = skeleton[..., :3] @ Rz.T

        # 3. 还原平移
        if self.center_on_root:
            skeleton += root_pos
            
        return skeleton

class TemporalCropOrPad:
    """时间轴剪裁或填充至固定长度 L"""
    def __init__(self, target_len=64, padding_mode='zero', random=False):
        self.L = target_len
        self.padding_mode = padding_mode  # 'zero' (推荐) | 'repeat_first'
        self.random = random

    def __call__(self, skeleton):
        T = skeleton.shape[0]
        if T == self.L:
            return skeleton
        if T > self.L:
            if self.random:
                # 随机起始帧裁剪
                start = random.randint(0, T - self.L)
            else:
                # 固定中心裁剪 (测试集/验证集推荐)
                start = (T - self.L) // 2
            return skeleton[start:start + self.L]
        else:
            # 填充策略
            pad_len = self.L - T
            if self.padding_mode == 'zero':
                # ST-GCN 标准做法：零填充不影响图卷积的邻接聚合
                pad_shape = (pad_len,) + skeleton.shape[1:]
                padding = np.zeros(pad_shape, dtype=skeleton.dtype)
            else:
                # 备用：重复首帧（比末帧更符合动作起始先验）
                padding = np.repeat(skeleton[0:1], pad_len, axis=0)
            return np.concatenate([skeleton, padding], axis=0)

class SkeletonCompose:
    """标准变换管道"""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton
