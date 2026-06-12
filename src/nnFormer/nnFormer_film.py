"""
nnFormer_film.py — FiLM-conditioned nnFormer for low-dose PET denoising.

The vanilla model at ``nnFormer.nnFormer_seg.nnFormer`` is kept untouched. This
file builds parallel ``Cond*`` classes that accept a conditioning vector
``cond \\in R^{B x d_cond}`` and inject it via ``AdaLayerNorm`` (FiLM) **only at
the transformer-block norms** (``norm1`` and ``norm2`` inside SwinTransformerBlock
and SwinTransformerBlock_kv).

LayerNorms outside the transformer blocks — PatchEmbed/PatchMerging/
Patch_Expanding/Encoder output norms/final_patch_expanding — stay vanilla, so
the cond signal is restricted to the attention/MLP path of each block.

Initialization is identity: every ``AdaLayerNorm.mlp[-1]`` is zero so the model
output is bit-identical to a vanilla nnFormer until the FiLM heads learn a
non-zero modulation.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple

from nnFormer.conditioning import AdaLayerNorm
from nnFormer.nnFormer_seg import (
    Mlp,
    WindowAttention,
    WindowAttention_kv,
    PatchMerging,
    Patch_Expanding,
    PatchEmbed,
    final_patch_expanding,
    window_partition,
    window_reverse,
    identity_function,
)
from nnFormer.neural_network import SegmentationNetwork


__all__ = ["nnFormerFiLM", "drf_to_cond", "make_cond_fn", "COND_TRANSFORM"]


# ---------------------------------------------------------------------------
# Conditioning encoding
# ---------------------------------------------------------------------------

COND_TRANSFORM = {
    "kind": "log_div",
    "log": "natural",
    "divisor": 10.0,
    "version": 1,
}


def drf_to_cond(drf: float) -> float:
    """Map a Dose Reduction Factor to the FiLM conditioning scalar using the
    currently-installed transform (`COND_TRANSFORM`).

    ``cond = log(drf) / 10`` (natural log) per FILM_IMPLEMENTATION.md sec 4.2.1.
    """
    return math.log(float(drf)) / 10.0


def make_cond_fn(transform):
    """Return a ``drf -> cond`` callable for an arbitrary ``COND_TRANSFORM`` dict.

    Lets evaluation scripts honor the exact transform that was used at training
    time (recovered from the checkpoint's ``config['COND_TRANSFORM']``), instead
    of assuming the module-global default.

    Supported kinds:
      - ``log_div``: ``cond = log_<log>(drf) / divisor`` for ``log in {natural, 10}``.

    Unknown shapes fall back to ``drf_to_cond`` with a warning.
    """
    if not isinstance(transform, dict):
        return drf_to_cond
    kind = transform.get("kind")
    if kind == "log_div":
        log_kind = transform.get("log", "natural")
        divisor = float(transform.get("divisor", 10.0))
        if log_kind == "natural":
            return lambda drf: math.log(float(drf)) / divisor
        if log_kind in ("10", "log10"):
            return lambda drf: math.log10(float(drf)) / divisor
    import warnings
    warnings.warn(f"Unknown COND_TRANSFORM {transform!r}; falling back to drf_to_cond")
    return drf_to_cond


# ---------------------------------------------------------------------------
# Conditional Swin transformer blocks
# ---------------------------------------------------------------------------


class CondSwinBlock(nn.Module):
    """SwinTransformerBlock with AdaLayerNorm for norm1/norm2.

    Mirrors ``nnFormer.nnFormer_seg.SwinTransformerBlock`` byte-for-byte except
    that the two LayerNorms are replaced with FiLM-conditioned variants and
    ``forward`` accepts ``cond``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.ReLU,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = AdaLayerNorm(dim, d_cond=d_cond, hidden=film_hidden)
        self.attn = WindowAttention(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaLayerNorm(dim, d_cond=d_cond, hidden=film_hidden)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, mask_matrix, cond: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x, cond)
        x = x.view(B, S, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,) * 3, dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 3, C)
        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=None)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,) * 3, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x, cond)))
        return x


class CondSwinBlock_kv(nn.Module):
    """SwinTransformerBlock_kv with AdaLayerNorm for norm1/norm2.

    Mirrors ``nnFormer.nnFormer_seg.SwinTransformerBlock_kv`` with FiLM-aware
    norms and a ``cond`` parameter in ``forward``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.ReLU,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = AdaLayerNorm(dim, d_cond=d_cond, hidden=film_hidden)
        self.attn = WindowAttention_kv(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaLayerNorm(dim, d_cond=d_cond, hidden=film_hidden)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        mask_matrix,
        skip: torch.Tensor,
        x_up: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        skip = self.norm1(skip, cond)
        x_up = self.norm1(x_up, cond)

        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        x = x.view(B, S, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size
        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = skip.shape

        if self.shift_size > 0:
            skip = torch.roll(skip, shifts=(-self.shift_size,) * 3, dims=(1, 2, 3))
            x_up = torch.roll(x_up, shifts=(-self.shift_size,) * 3, dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            attn_mask = None

        skip = window_partition(skip, self.window_size).view(-1, self.window_size ** 3, C)
        x_up = window_partition(x_up, self.window_size).view(-1, self.window_size ** 3, C)
        attn_windows = self.attn(skip, x_up, mask=attn_mask, pos_embed=None)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,) * 3, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x, cond)))
        return x


# ---------------------------------------------------------------------------
# Conditional encoder / decoder layers
# ---------------------------------------------------------------------------


class CondBasicLayer(nn.Module):
    """BasicLayer using CondSwinBlock and threading cond through forward."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path=0.0,
        downsample=True,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            CondSwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                d_cond=d_cond,
                film_hidden=film_hidden,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            # PatchMerging keeps its vanilla LayerNorm (no FiLM here).
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def _build_attn_mask(self, x: torch.Tensor, S: int, H: int, W: int) -> torch.Tensor:
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        s_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        h_slices = s_slices
        w_slices = s_slices
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, S, H, W, cond):
        attn_mask = self._build_attn_mask(x, S, H, W)
        for blk in self.blocks:
            x = blk(x, attn_mask, cond)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        return x, S, H, W, x, S, H, W


class CondBasicLayer_up(nn.Module):
    """BasicLayer_up using CondSwinBlock_kv (first block) + CondSwinBlock."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path=0.0,
        upsample=True,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList()
        self.blocks.append(
            CondSwinBlock_kv(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                d_cond=d_cond,
                film_hidden=film_hidden,
            )
        )
        for i in range(depth - 1):
            self.blocks.append(
                CondSwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    d_cond=d_cond,
                    film_hidden=film_hidden,
                )
            )

        # Patch_Expanding keeps its vanilla LayerNorm.
        self.Upsample = upsample(dim=2 * dim, norm_layer=nn.LayerNorm)

    def _build_attn_mask(self, x: torch.Tensor, S: int, H: int, W: int) -> torch.Tensor:
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)
        s_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        h_slices = s_slices
        w_slices = s_slices
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, skip, S, H, W, cond):
        x_up = self.Upsample(x, S, H, W)
        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        attn_mask = self._build_attn_mask(x, S, H, W)
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up, cond=cond)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask, cond)
        return x, S, H, W


# ---------------------------------------------------------------------------
# Conditional encoder / decoder
# ---------------------------------------------------------------------------


class CondEncoder(nn.Module):
    def __init__(
        self,
        pretrain_img_size,
        patch_size=4,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        window_size: Sequence[int] = (4, 4, 8, 4),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = tuple(out_indices)

        # PatchEmbed keeps vanilla LayerNorm — no FiLM at the conv stem.
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CondBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer,
                    pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer,
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                d_cond=d_cond,
                film_hidden=film_hidden,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Per-stage skip norms remain vanilla LayerNorms (no FiLM here).
        for i_layer in self.out_indices:
            self.add_module(f"norm{i_layer}", nn.LayerNorm(num_features[i_layer]))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        down: List[torch.Tensor] = []

        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww, cond)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                down.append(out)
        return down


class CondDecoder(nn.Module):
    def __init__(
        self,
        pretrain_img_size,
        embed_dim: int,
        patch_size=4,
        depths: Sequence[int] = (2, 2, 2),
        num_heads: Sequence[int] = (24, 12, 6),
        window_size: Sequence[int] = (4, 8, 4),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = CondBasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths) - i_layer - 1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths) - i_layer - 1),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                upsample=Patch_Expanding,
                d_cond=d_cond,
                film_hidden=film_hidden,
            )
            self.layers.append(layer)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor], cond: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)

        for idx in range(self.num_layers)[::-1]:
            layer = self.layers[idx]
            skip = skips[idx].flatten(2).transpose(1, 2).contiguous()
            x, S, H, W = layer(x, skip, S, H, W, cond)
            outs.append(x.view(-1, S, H, W, self.num_features[idx]))
        return outs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class nnFormerFiLM(SegmentationNetwork):
    """FiLM-conditioned nnFormer.

    Forward signature: ``forward(x, cond)`` where ``cond`` is a ``[B, d_cond]``
    float tensor (with ``d_cond=1`` and ``cond = log(DRF)/10`` for the PET
    low-dose application).
    """

    def __init__(
        self,
        crop_size: Sequence[int] = (96, 96, 96),
        embedding_dim: int = 64,
        input_channels: int = 1,
        num_classes: int = 1,
        conv_op=nn.Conv3d,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (8, 16, 32, 64),
        patch_size: Sequence[int] = (2, 4, 4),
        window_size: Sequence[int] = (4, 4, 8, 4),
        deep_supervision: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        d_cond: int = 1,
        film_hidden: int = 128,
    ):
        super().__init__()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.aleatoric_uncertainty = False
        self.d_cond = int(d_cond)

        self.upscale_logits_ops = [identity_function]

        self.model_down = CondEncoder(
            pretrain_img_size=crop_size,
            window_size=window_size,
            embed_dim=embedding_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            in_chans=input_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            d_cond=d_cond,
            film_hidden=film_hidden,
        )
        self.decoder = CondDecoder(
            pretrain_img_size=crop_size,
            embed_dim=embedding_dim,
            window_size=tuple(window_size)[::-1][1:],
            patch_size=patch_size,
            num_heads=tuple(num_heads)[::-1][1:],
            depths=tuple(depths)[::-1][1:],
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            d_cond=d_cond,
            film_hidden=film_hidden,
        )
        self.final = final_patch_expanding(embedding_dim, num_classes, patch_size=patch_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)
        skips = self.model_down(x, cond)
        neck = skips[-1]
        decoder_out = self.decoder(neck, skips, cond)
        if isinstance(self.final, nn.ModuleList):
            out = self.final[-1](decoder_out[-1])
        else:
            out = self.final(decoder_out[-1])
        return out


# ---------------------------------------------------------------------------
# Convenience smoke test (importable)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _selftest(device: str = "cpu") -> None:
    model = nnFormerFiLM(
        crop_size=(96, 96, 96),
        embedding_dim=64,
        num_heads=(8, 16, 32, 64),
        depths=(2, 2, 2, 2),
        d_cond=1,
    ).to(device).eval()
    x = torch.randn(2, 1, 96, 96, 96, device=device)
    cond = torch.tensor([[drf_to_cond(4)], [drf_to_cond(100)]], device=device)
    y = model(x, cond)
    assert y.shape == (2, 1, 96, 96, 96), y.shape
    assert torch.isfinite(y).all().item()
    print(f"_selftest OK on {device}: output shape {tuple(y.shape)}")


if __name__ == "__main__":
    _selftest("cuda" if torch.cuda.is_available() else "cpu")
