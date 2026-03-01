import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *


class Feature_fusion(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()

        # 先做通道压缩（比先上采样更省算力）
        self.reduce0 = nn.Conv2d(192, out_channels, 1, bias=False)
        self.reduce1 = nn.Conv2d(384, out_channels, 1, bias=False)
        self.reduce2 = nn.Conv2d(384, out_channels, 1, bias=False)

        # 轻量 refine（depthwise separable）
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def fuse_one_side(self, features):
        # BHWC → BCHW
        f0 = features[0].permute(0, 3, 1, 2)  # 32x64
        f1 = features[1].permute(0, 3, 1, 2)  # 16x32
        f2 = features[2].permute(0, 3, 1, 2)  # 16x32

        # 1️⃣ 先降维（比先上采样省 FLOPs）
        f0 = self.reduce0(f0)
        f1 = self.reduce1(f1)
        f2 = self.reduce2(f2)

        # 2️⃣ 上采样到最高分辨率
        target_size = f0.shape[-2:]

        f1 = F.interpolate(f1, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)

        # 3️⃣ 加法融合（避免 cat）
        fused = f0 + f1 + f2

        # 4️⃣ 轻量 refine
        fused = self.refine(fused)

        return fused.permute(0, 2, 3, 1)

    def forward(self, left_features, right_features):

        left_output = self.fuse_one_side(left_features)
        right_output = self.fuse_one_side(right_features)

        return left_output, right_output