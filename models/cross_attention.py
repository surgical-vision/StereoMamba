# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        # self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Modify Conv2d downsampling layer
        self.downsample = nn.Conv2d(
            in_channels=3,  # Changed from d_model to 3 for RGB input
            out_channels=self.d_model,  # Changed from d_model to 3 for RGB output
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=0,
            **factory_kwargs
        )


    def forward(self, left_img, right_img, seq_idx=None):
        """
        left_img: (B, C, H, W) = (14, 3, 256, 512)
        Returns: same shape as left_img but downsampled
        """
       
        # Apply downsampling using Conv2d
        left_img_downsampled = self.downsample(left_img)   # Output: [14, 3, 64, 128]
        right_img_downsampled = self.downsample(right_img) # Output: [14, 3, 64, 128]
        batch, channels, height, width = left_img_downsampled.shape
        # Reshape from [B,C,H,W] to [B,H*W,C]
        left_img_flat = left_img_downsampled.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        right_img_flat = right_img_downsampled.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        
        seqlen = height * width  # 8192
        
        zxbcdt_left = self.in_proj(left_img_flat)  # (B, L, d_in_proj)
        zxbcdt_right = self.in_proj(right_img_flat)
        A_left = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        A_right = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        if self.use_mem_eff_path: # False
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt_left,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A_left,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z_left, xBC_left, dt_left = torch.split(
                zxbcdt_left, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt_left = F.softplus(dt_left + self.dt_bias)  # (B, L, nheads)
            z_right, xBC_right, dt_right = torch.split(
                zxbcdt_right, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt_right = F.softplus(dt_right + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC_left = self.act(
                    self.conv1d(xBC_left.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC_left = xBC_left[:, :seqlen, :]
            else:
                xBC_left = causal_conv1d_fn(
                    x=xBC_left.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x_left, B_left, C_left = torch.split(xBC_left, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            x_right, B_right, C_right = torch.split(xBC_right, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y_left = mamba_chunk_scan_combined(
                rearrange(x_right, "b l (h p) -> b l h p", p=self.headdim),
                dt_left,
                A_left,
                rearrange(B_right, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C_left, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y_right = mamba_chunk_scan_combined(
                rearrange(x_left, "b l (h p) -> b l h p", p=self.headdim),
                dt_right,
                A_right,
                rearrange(B_left, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C_right, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y_left = rearrange(y_left, "b l h p -> b l (h p)")
            y_right = rearrange(y_right, "b l h p -> b l (h p)")
            # Multiply "gate" branch and apply extra normalization layer
            y_left = self.norm(y_left, z_left)
            y_right = self.norm(y_right, z_right)
            out_left = self.out_proj(y_left)
            out_right = self.out_proj(y_right)
            out_left = out_left.reshape(batch, height, width, channels)#.permute(0, 3, 1, 2)
            out_right = out_right.reshape(batch, height, width, channels)#.permute(0, 3, 1, 2)
        return out_left, out_right


class CrossAttn(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim, ngroups):
        super(CrossAttn, self).__init__()
        self.cross_attn = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=None,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="swish",
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=False,
            layer_idx=None,  # Absorb kwarg for general module
            device=None,
            dtype=None,
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(d_model+384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, img_left, img_right, left_features, right_features):
        
        left_attn, right_attn = self.cross_attn(img_left, img_right)

        left_features = self.conv_concat(torch.cat([left_attn.permute(0,3,1,2), left_features.permute(0,3,1,2)], dim=1))
        right_features = self.conv_concat(torch.cat([right_attn.permute(0,3,1,2), right_features.permute(0,3,1,2)], dim=1))

        return left_features, right_features