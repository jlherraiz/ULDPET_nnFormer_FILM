"""
infer_nifti.py — Denoise low-dose PET NIfTI volumes with the FiLM-conditioned
nnFormer model.

A single conditioned model handles all dose-reduction factors (DRFs); the DRF is
passed at inference time and turned into the FiLM conditioning scalar.

Pipeline (matches the training/evaluation protocol exactly):
  1. read NIfTI (low-dose input), keep geometry (spacing/origin/direction)
  2. min-max percentile normalize to ~[0,1] with the scanner's GLOBAL percentiles
  3. sliding-window inference (96^3 ROI, overlap 0.5, gaussian), conditioned on DRF
  4. clamp to >= 0, de-normalize back to the input intensity domain (e.g. Bq/mL)
  5. write NIfTI with the original geometry

Example:
  python infer_nifti.py \
      --ckpt FILM_QUADRA_ep40_best.pt \
      --input low_dose.nii.gz --drf 100 \
      --percentiles global_percentiles_QUADRA.json \
      --output denoised.nii.gz

Run inference in fp32 (default) or bf16 (--bf16 on Ampere+). NEVER fp16: the
attention softmax overflows fp16's range and produces NaNs.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch
import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer

# make the bundled model package (../src) importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from nnFormer.nnFormer_film import nnFormerFiLM, make_cond_fn, COND_TRANSFORM

PATCH_SIZE = (96, 96, 96)
VALUE_ABS_MAX = 1e8  # neutralize finite-but-implausible voxels (corrupt chunks)


def minmax_percentile(x: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """(x - p_low) / (p_high - p_low), with garbage/non-finite voxels -> p_low.
    No [0,1] clip (matches training)."""
    scale = float(p_high - p_low)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x = x.astype(np.float32)
    bad = ~np.isfinite(x) | (np.abs(x) > VALUE_ABS_MAX)
    if bad.any():
        x = np.where(bad, np.float32(p_low), x)
    y = (x - float(p_low)) / scale
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _build_from_cfg(cfg, sd, device):
    crop = tuple(cfg.get("patch_size", PATCH_SIZE))
    emb = cfg.get("embedding_dim", cfg.get("EMBEDDING_DIM", 64))
    heads = tuple(cfg.get("num_heads", cfg.get("NUM_HEADS", (8, 16, 32, 64))))
    depths = tuple(cfg.get("depths", cfg.get("DEPTHS", (2, 2, 2, 2))))
    d_cond = cfg.get("d_cond", 1)
    transform = cfg.get("COND_TRANSFORM", COND_TRANSFORM)
    model = nnFormerFiLM(crop_size=crop, embedding_dim=emb, num_heads=heads, depths=depths, d_cond=d_cond).to(device)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model, make_cond_fn(transform), cfg


def load_model(ckpt_path: str, device: torch.device):
    """Build nnFormerFiLM from a checkpoint and return (model, cond_fn, cfg).

    Supports:
      - .safetensors  : weights + JSON 'config' in the file metadata (recommended
                        for distribution; no code execution on load).
      - .pt training checkpoint: dict with 'model_state' + 'config'.
      - .pt bare state_dict: falls back to default hyperparameters.
    A sidecar '<ckpt>.config.json' is used if a bare file carries no config.
    """
    import json as _json, os as _os
    if ckpt_path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file
        cfg = {}
        with safe_open(ckpt_path, framework="pt") as f:   # read embedded metadata
            md = f.metadata() or {}
            if "config" in md:
                cfg = _json.loads(md["config"])
        if not cfg:
            side = ckpt_path.rsplit(".safetensors", 1)[0] + ".config.json"
            if _os.path.isfile(side):
                cfg = _json.load(open(side))
        return _build_from_cfg(cfg, load_file(ckpt_path), device)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        return _build_from_cfg(state["config"], state["model_state"], device)
    side = ckpt_path + ".config.json"
    cfg = _json.load(open(side)) if _os.path.isfile(side) else {}
    return _build_from_cfg(cfg, state, device)


def main() -> int:
    ap = argparse.ArgumentParser(description="FiLM nnFormer low-dose PET denoising (single model, all DRFs).")
    ap.add_argument("--ckpt", required=True, help="FiLM checkpoint (.pt)")
    ap.add_argument("--input", required=True, help="Low-dose input NIfTI (.nii/.nii.gz)")
    ap.add_argument("--drf", type=int, required=True, help="Dose reduction factor of the input (e.g. 4,10,20,50,100)")
    ap.add_argument("--percentiles", default=None, help="global_percentiles_*.json (p_low,p_high). "
                    "Optional for .safetensors that embed p_low/p_high in config.")
    ap.add_argument("--output", required=True, help="Output NIfTI path")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (Ampere+). Default fp32. NEVER use fp16.")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cond_fn, cfg = load_model(a.ckpt, device)

    if a.percentiles:
        pct = json.load(open(a.percentiles))
        p_low, p_high = float(pct["p_low"]), float(pct["p_high"])
    elif "p_low" in cfg and "p_high" in cfg:
        p_low, p_high = float(cfg["p_low"]), float(cfg["p_high"])   # from safetensors metadata
    else:
        raise SystemExit("No percentiles: pass --percentiles or use a checkpoint with embedded p_low/p_high.")
    scale = float(max(1e-6, p_high - p_low))
    cond = torch.tensor([[float(cond_fn(a.drf))]], dtype=torch.float32, device=device)
    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=a.overlap, mode="gaussian")

    img = sitk.ReadImage(a.input)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    vol_mm = minmax_percentile(vol, p_low, p_high)

    amp_dtype = torch.bfloat16 if a.bf16 else torch.float32
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=a.bf16):
        xt = torch.from_numpy(vol_mm).float()[None, None].to(device)
        pred_mm = inferer(xt, lambda x: model(x, cond.expand(x.shape[0], -1)))
    pred_mm = pred_mm.float().clamp_(0.0, None)[0, 0].cpu().numpy()
    pred = pred_mm * scale + p_low  # de-normalize to input intensity domain

    out = sitk.GetImageFromArray(pred.astype(np.float32))
    out.SetSpacing(img.GetSpacing()); out.SetOrigin(img.GetOrigin()); out.SetDirection(img.GetDirection())
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    sitk.WriteImage(out, a.output)
    print(f"DRF={a.drf} cond={float(cond_fn(a.drf)):.4f} | in[max]={vol.max():.1f} -> out[min={pred.min():.3f} max={pred.max():.1f}] | wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
