"""
conditioning.py
----------------
FiLM / conditional normalization layers for nnFormer-based denoiser.

Provides:
- AdaLayerNorm: Conditional LayerNorm for transformer token streams.
- AdaGroupNorm3d: Conditional GroupNorm for 3D convolutional feature maps.

Both layers are **identity-initialized** so that immediately after construction
(or when loading pretrained nnFormer weights), they behave like their vanilla
counterparts (no conditioning effect) until trained.

Usage:
  from nnformer.conditioning import AdaLayerNorm, AdaGroupNorm3d
  # In transformer blocks:
  self.norm1 = AdaLayerNorm(dim, d_cond=1)
  x = self.norm1(x, cond)  # cond shape: [B, d_cond]

  # In conv blocks:
  self.norm = AdaGroupNorm3d(num_groups=8, num_channels=C, d_cond=1)
  x = self.norm(x, cond)

Notes on shapes:
- AdaLayerNorm expects inputs whose **last dimension** is the feature size (D).
  Typical shapes: [B, N, D] or [*, D]. The conditioning vector is [B, d_cond].
- AdaGroupNorm3d expects inputs in [B, C, D, H, W] and cond in [B, d_cond].

Author: ChatGPT (2025)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["AdaLayerNorm", "AdaGroupNorm3d"]


class AdaLayerNorm(nn.Module):
    """
    Conditional LayerNorm (FiLM on normalized features).

    - Performs standard LayerNorm (elementwise_affine=False).
    - Computes per-sample FiLM parameters (gamma, beta) from a conditioning
      vector via a small MLP, and applies them to the normalized features.
    - Identity-initialized so it starts with no effect on the base LayerNorm.

    Args:
        normalized_shape: feature dimension (int) or tuple passed to LayerNorm.
        d_cond: dimension of the conditioning vector.
        eps: numerical epsilon for LayerNorm.
        hidden: hidden size in the FiLM MLP.

    Forward:
        x: tensor with last dim == normalized_shape (e.g., [B, N, D]).
        cond: [B, d_cond]; will be broadcast across sequence/spatial dims.
    """
    def __init__(
        self,
        normalized_shape: int | Tuple[int, ...],
        d_cond: int,
        eps: float = 1e-6,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, int) else int(normalized_shape[-1])
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)

        self.mlp = nn.Sequential(
            nn.Linear(d_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.normalized_shape),
        )
        # Identity init: FiLM starts as no-op (gamma=0, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Optional base scale/bias to aid optimization (start as identity)
        self.register_parameter("bias_base", nn.Parameter(torch.zeros(1, self.normalized_shape)))
        self.register_parameter("scale_base", nn.Parameter(torch.ones(1, self.normalized_shape)))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [..., D]
        cond: [B, d_cond]  (B must match x.shape[0])
        """
        y = self.ln(x)  # same shape as x
        B = y.shape[0]
        if cond.dim() != 2 or cond.shape[0] != B:
            raise ValueError(f"cond must be [B, d_cond] with B={B}, got {tuple(cond.shape)}")

        gamma, beta = self.mlp(cond).chunk(2, dim=-1)  # [B, D] each

        # Reshape for broadcasting across all non-batch dims except channel/feat
        expand_shape = [B] + [1] * (y.ndim - 2) + [self.normalized_shape]
        gamma = gamma.view(*expand_shape)
        beta = beta.view(*expand_shape)

        scale = self.scale_base
        bias = self.bias_base
        # expand base parameters to match feature rank (excluding batch dim)
        for _ in range(y.ndim - 2):
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)

        return (scale * (1.0 + gamma)) * y + (bias + beta)


class AdaGroupNorm3d(nn.Module):
    """
    Conditional GroupNorm for 3D tensors.

    - Applies GroupNorm (affine=False).
    - Generates per-sample FiLM (gamma, beta) from cond via an MLP.
    - Identity-initialized so output equals vanilla GroupNorm at start.

    Args:
        num_groups: number of groups for GroupNorm.
        num_channels: channel dim C.
        d_cond: dimension of conditioning vector.
        eps: epsilon for GroupNorm.
        hidden: hidden size in the FiLM MLP.

    Forward:
        x: [B, C, D, H, W]
        cond: [B, d_cond]
    """
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        d_cond: int,
        eps: float = 1e-6,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)

        self.gn = nn.GroupNorm(self.num_groups, self.num_channels, affine=False, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.num_channels),
        )
        # Identity init for FiLM head
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Base scale/bias
        self.register_parameter("bias_base", nn.Parameter(torch.zeros(1, self.num_channels, 1, 1, 1)))
        self.register_parameter("scale_base", nn.Parameter(torch.ones(1, self.num_channels, 1, 1, 1)))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"AdaGroupNorm3d expects x of shape [B,C,D,H,W], got {tuple(x.shape)}")
        B = x.shape[0]
        if cond.dim() != 2 or cond.shape[0] != B:
            raise ValueError(f"cond must be [B, d_cond] with B={B}, got {tuple(cond.shape)}")

        y = self.gn(x)
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)  # [B, C] each

        # Reshape FiLM params to [B,C,1,1,1]
        gamma = gamma.view(B, self.num_channels, 1, 1, 1)
        beta = beta.view(B, self.num_channels, 1, 1, 1)

        return (self.scale_base * (1.0 + gamma)) * y + (self.bias_base + beta)


if __name__ == "__main__":
    # --- Minimal smoke tests ---
    torch.manual_seed(0)

    # AdaLayerNorm: [B,N,D]
    B, N, D = 2, 7, 16
    x = torch.randn(B, N, D)
    cond = torch.randn(B, 3)
    aln = AdaLayerNorm(D, d_cond=3)
    y = aln(x, cond)
    assert y.shape == x.shape, "AdaLayerNorm shape mismatch"

    # AdaGroupNorm3d: [B,C,D,H,W]
    B, C, Dd, H, W = 2, 8, 6, 10, 12
    x3 = torch.randn(B, C, Dd, H, W)
    cond3 = torch.randn(B, 2)
    agn = AdaGroupNorm3d(num_groups=4, num_channels=C, d_cond=2)
    y3 = agn(x3, cond3)
    assert y3.shape == x3.shape, "AdaGroupNorm3d shape mismatch"

    print("conditioning.py smoke tests: OK")
