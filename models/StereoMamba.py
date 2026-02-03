from .gwcnet import *
from .vmamba import VSSM, SS2D
from .feature_fusion import Feature_fusion
from .cross_attention import CrossAttn

# import torch
import torch.nn as nn
# import torch.nn.functional as F
# from typing import Callable, List, Optional
import time


class StereoMamba(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        # num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        _SS2D=SS2D,
        # =========================
        max_disparity=192, 
        use_concat_volume=False,
        cross_attn=True,
        d_model=64,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=8,
        ngroups=1,
        **kwargs,):
        super().__init__()

        self.cross_attn = cross_attn

        self.feature_extractor = VSSM(**kwargs)
        self.feature_fusion = Feature_fusion()
        self.CrossAttn = CrossAttn(d_model, d_state, d_conv, expand, headdim, ngroups)
        self.disparity_head = GwcNet(max_disparity, use_concat_volume)


    def forward(self, img_left, img_right):
        disparity_scales = None

        left_features = self.feature_extractor(img_left)
        right_features = self.feature_extractor(img_right)
        
        left_features, right_features = self.feature_fusion(left_features, right_features)
        
        if self.cross_attn:
            left_features, right_features = self.CrossAttn(left_features, right_features)
            
        else:
            left_features = left_features.permute(0,3,1,2)
            right_features = right_features.permute(0,3,1,2)
        
        disparity_scales = self.disparity_head((img_left, left_features),
                                               (img_right, right_features))
        
        return disparity_scales