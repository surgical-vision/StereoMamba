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

        # 2x 上采样（depthwise 反卷积，参数和显存开销更低）
        self.up2x = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=out_channels,
            bias=False,
        )

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

        # 2️⃣ ConvTranspose2d 上采样到最高分辨率
        f1 = self.up2x(f1)
        f2 = self.up2x(f2)

        # 若输入分辨率存在微小偏差，做一次安全对齐
        if f1.shape[-2:] != f0.shape[-2:]:
            f1 = F.interpolate(f1, size=f0.shape[-2:], mode="nearest")
        if f2.shape[-2:] != f0.shape[-2:]:
            f2 = F.interpolate(f2, size=f0.shape[-2:], mode="nearest")

        # 3️⃣ 低显存融合（避免 cat 产生 3 倍通道临时张量）
        fused = f0 + f1 + f2

        # 4️⃣ 轻量 refine
        fused = self.refine(fused)

        return fused.permute(0, 2, 3, 1)

    def forward(self, left_features, right_features):

        left_output = self.fuse_one_side(left_features)
        right_output = self.fuse_one_side(right_features)

        return left_output, right_output