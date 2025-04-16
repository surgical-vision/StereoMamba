import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *

# class Feature_fusion(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         # Upsampling and channel adjustment for level 3 features
#         # From [8, 8, 16, 768] to [8, 64, 128, 384]
#         self.upsample_level3 = nn.Sequential(
#             nn.Linear(768, 384),  # Reduce channels
#             nn.ReLU(inplace=True)
#         )
        
#         # Upsampling and channel adjustment for level 1 features (corrected)
#         # From [8, 16, 32, 384] to [8, 64, 128, 384]
#         self.upsample_level1 = nn.Sequential(
#             nn.Linear(384, 384),  # Maintain channels
#             nn.ReLU(inplace=True)
#         )
        
#         # Upsampling and channel adjustment for level 0 features (corrected)
#         # From [8, 32, 64, 192] to [8, 64, 128, 384]
#         self.upsample_level0 = nn.Sequential(
#             nn.Linear(192, 384),  # Increase channels
#             nn.ReLU(inplace=True)
#         )
        
#         # Feature fusion components
#         self.fusion_conv = nn.Sequential(
#             nn.Linear(384, 384),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, left_features, right_features):
#         # Process level 3 features [8, 8, 16, 768] -> [8, 64, 128, 384]
#         left_level3 = self.upsample_level3(left_features[3])
#         right_level3 = self.upsample_level3(right_features[3])
    
#         # Spatial upsampling for level 3
#         left_level3 = F.interpolate(
#             left_level3.permute(0, 3, 1, 2),  # [B, C, H, W]
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)  # Back to [B, H, W, C]
        
#         right_level3 = F.interpolate(
#             right_level3.permute(0, 3, 1, 2),
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)
        
#         # Process level 1 features [8, 16, 32, 384] -> [8, 64, 128, 384]
#         left_level1 = self.upsample_level1(left_features[1])
#         right_level1 = self.upsample_level1(right_features[1])
        
#         # Spatial upsampling for level 1
#         left_level1 = F.interpolate(
#             left_level1.permute(0, 3, 1, 2),
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)
        
#         right_level1 = F.interpolate(
#             right_level1.permute(0, 3, 1, 2),
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)
        
#         # Process level 0 features [8, 32, 64, 192] -> [8, 64, 128, 384]
#         left_level0 = self.upsample_level0(left_features[0])
#         right_level0 = self.upsample_level0(right_features[0])
        
#         # Spatial upsampling for level 0
#         left_level0 = F.interpolate(
#             left_level0.permute(0, 3, 1, 2),
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)
        
#         right_level0 = F.interpolate(
#             right_level0.permute(0, 3, 1, 2),
#             size=(64, 128),
#             mode='bilinear',
#             align_corners=False
#         ).permute(0, 2, 3, 1)
        
#         # Fuse features from different levels (excluding level 2)
#         left_fused = left_level3 + left_level1 + left_level0
#         right_fused = right_level3 + right_level1 + right_level0
        
#         # Final processing
#         left_output = self.fusion_conv(left_fused)
#         right_output = self.fusion_conv(right_fused)
        
#         return left_output, right_output

class Feature_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Level 3: [8, 768, 8, 16] -> [8, 384, 64, 128]
        self.upsample_level3 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=4, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: [8, 384, 16, 32] -> [8, 384, 64, 128]
        self.upsample_level1 = nn.Sequential(
            nn.ConvTranspose2d(384, 384, kernel_size=4, stride=4, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Level 0: [8, 192, 32, 64] -> [8, 384, 64, 128]
        self.upsample_level0 = nn.Sequential(
            nn.ConvTranspose2d(384, 384, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_concat = nn.Sequential(
            nn.Conv2d(576, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion component
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, left_features, right_features):
        # Process features (now using BCHW format)
        left_level3 = self.upsample_level3(left_features[3].permute(0, 3, 1, 2))
        right_level3 = self.upsample_level3(right_features[3].permute(0, 3, 1, 2))
        
        left_level1 = self.upsample_level1(left_features[1].permute(0, 3, 1, 2))
        right_level1 = self.upsample_level1(right_features[1].permute(0, 3, 1, 2))
        
        left_fuse_level3_feat0 = self.conv_concat(torch.cat([left_level3, left_features[0].permute(0, 3, 1, 2)], dim=1))
        right_fuse_level3_feat0 = self.conv_concat(torch.cat([right_level3, right_features[0].permute(0, 3, 1, 2)], dim=1))

        left_level0 = self.upsample_level0(left_fuse_level3_feat0)
        right_level0 = self.upsample_level0(right_fuse_level3_feat0)

        # Fuse features
        left_fused = torch.cat([left_level1, left_level0], dim=1)
        right_fused = torch.cat([right_level1, right_level0], dim=1)
        # right_fused = right_level1 + right_level0
        
        # Final processing
        left_output = self.fusion_conv(left_fused).permute(0, 2, 3, 1)  # Back to BHWC
        right_output = self.fusion_conv(right_fused).permute(0, 2, 3, 1)  # Back to BHWC
        
        return left_output, right_output