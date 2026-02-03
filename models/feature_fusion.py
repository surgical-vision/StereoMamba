import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *


class Feature_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Level 2: [B, 16, 32, 384] -> [B, 64, 128, 128]
        self.upsample_level2 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: [B, 16, 32, 384] -> [B, 64, 128, 128]
        self.upsample_level1 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=4, stride=4, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Level 0: [B, 32, 64, 192] -> [B, 64, 128, 128]
        self.upsample_level0 = nn.Sequential(
            nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fusion of all levels: 3 * 128 = 384 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
    
    def forward(self, left_features, right_features):
        # Convert from BHWC to BCHW for convolution operations
        # Process left features
        left_feat0 = self.upsample_level0(left_features[0].permute(0, 3, 1, 2))
        left_feat1 = self.upsample_level1(left_features[1].permute(0, 3, 1, 2))
        left_feat2 = self.upsample_level2(left_features[2].permute(0, 3, 1, 2))
        
        # Process right features
        right_feat0 = self.upsample_level0(right_features[0].permute(0, 3, 1, 2))
        right_feat1 = self.upsample_level1(right_features[1].permute(0, 3, 1, 2))
        right_feat2 = self.upsample_level2(right_features[2].permute(0, 3, 1, 2))

        # Concatenate all levels
        left_fused = torch.cat([left_feat0, left_feat1, left_feat2], dim=1)
        right_fused = torch.cat([right_feat0, right_feat1, right_feat2], dim=1)
        
        # Final fusion
        left_output = self.fusion_conv(left_fused).permute(0, 2, 3, 1)  # Back to BHWC
        right_output = self.fusion_conv(right_fused).permute(0, 2, 3, 1)  # Back to BHWC
        
        return left_output, right_output