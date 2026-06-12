# FiLM-nnFormer — DRF-conditioned low-dose PET denoising

A single **FiLM-conditioned nnFormer** that denoises low-dose PET across the full
range of dose-reduction factors (DRFs) with **one model per scanner**, instead of
training a separate model for every dose level.

Two pretrained models are released, one per scanner:

| Model | Scanner | Training patients | DRFs covered | Best epoch | Val loss\* |
|-------|---------|-------------------|--------------|-----------|-----------|
| `FILM_QUADRA_ep40_best.pt`   | Siemens Biograph **Vision Quadra** | 377 | 4, 10, 20, 50, 100 | 40 / 40 | 0.00181 |
| `FILM_EXPLORER_ep20_best.pt` | United Imaging **uEXPLORER**       | 994 | 4, 10, 20, 50, 100 | 20 / 50 | 0.00214 |

\* aggregate `0.95·L1 + 0.05·(1−SSIM)` on the held-out validation set, normalized [0,1] domain.

> Pretrained weights ship **in this repo** under `weights/` (fp32 safetensors, 74 MB
> each) and are also mirrored as
> **[release assets](https://github.com/jlherraiz/ULDPET_nnFormer_FILM/releases/latest)**.
> See **Pretrained weights** below.

---

## 1. What this is, and how it differs from the baseline

**Baseline (what we compare against):** a *separate* nnFormer denoiser trained for
**each DRF** — 5 models per scanner (DRF 4/10/20/50/100), i.e. **10 models** total.
Each model only ever sees one dose level.

**This model (FiLM):** a *single* network per scanner, **conditioned on the DRF**.
The DRF is encoded as a scalar `cond = ln(DRF)/10` and injected through
**FiLM-modulated LayerNorms** (`AdaLayerNorm`) inside every Swin transformer block.
The same weights handle all dose levels; you tell it the DRF at inference time.

| | Baseline | FiLM (this repo) |
|---|---|---|
| Models per scanner | 5 (one per DRF) | **1** |
| DRF handling | implicit (separate weights) | explicit conditioning `ln(DRF)/10` |
| Add a new DRF | retrain a new model | condition on the new value (interpolates) |
| Storage / deployment | 5 checkpoints | 1 checkpoint |
| Architecture | nnFormer | nnFormer + FiLM `AdaLayerNorm` |
| Training data, normalization, patch size, inferer | **identical** | **identical** |

**Result:** on a like-for-like evaluation (same patients, same preprocessing, same
sliding-window inferer; only the model differs), the single FiLM model **matches the
per-DRF specialists to within ±0.0002 L1 / ±0.0007 SSIM** — no meaningful quality
loss from collapsing 5 models into 1. See **Results**.

---

## 2. Datasets

**Data origin.** Both scanners' data comes from the **UDPET Challenge** (Ultra-Low
Dose PET Imaging Challenge), <https://udpet-challenge.github.io/>. The full challenge
dataset is **1,447** whole-body **¹⁸F-FDG** total-body PET subjects — **Siemens
Biograph Vision Quadra (n=387)** and **United Imaging uEXPLORER (n=1,060)** — each with
a standard-dose acquisition and simulated low-dose versions. Low-dose data is generated
from **list-mode** acquisitions by rebinning the counts of a time window resampled at
the middle of the scan with correspondingly reduced time, at five **dose-reduction
factors (DRF 4, 10, 20, 50, 100)** plus full dose. Images are NIfTI in **Bq/mL** with a
CSV of DICOM-header metadata (patient weight, injected activity, acquisition/injection
time offsets, isotope half-life). Access requires a signed **Data Transfer Agreement**;
the raw patient data is **not** redistributed here. The patient counts below are the
subsets used to train these particular FiLM models.

Two scanners, trained independently. Each patient contributes a **full-dose target**
and several **simulated low-dose inputs** at fixed DRFs (the input at DRF *N* uses
≈1/*N* of the counts).

| Scanner | Manufacturer / model | Patients | DRFs used for training | Voxel intensity unit |
|---|---|---|---|---|
| QUADRA   | Siemens Biograph Vision Quadra | 377 | 4, 10, 20, 50, 100 | Bq/mL |
| EXPLORER | United Imaging uEXPLORER        | 994 | 4, 10, 20, 50, 100 | Bq/mL |

- **Split:** 95% train / 5% validation, deterministic (`seed=42`). QUADRA → 359 train /
  18 val; EXPLORER → 945 train / 49 val. Splits are by **patient**, never mixing a
  patient's volumes across train/val.
- **Storage:** volumes are kept as chunked `zarr` stores named
  `<patient>__full.zarr` (target) and `<patient>__drf{N}.zarr` (inputs).
- **Normalization:** global min–max percentile scaling to ~[0,1] using the **0.1 and
  99.9 percentiles of the training targets** (shipped in
  `global_percentiles_{QUADRA,EXPLORER}.json`):
  - QUADRA: `p_low=0`, `p_high=14973.4794921875`
  - EXPLORER: `p_low=0`, `p_high=14296.7197265625`
  These constants are part of the inference contract — outputs are de-normalized back
  to Bq/mL with the same constants.

> The raw patient PET data is **not** redistributed here (clinical data). This repo
> provides the model, code, normalization constants, and protocol to reproduce
> training/inference on equivalently-formatted data.

---

## 3. Architecture

`nnFormerFiLM` (`src/nnFormer/nnFormer_film.py`) = nnFormer encoder–decoder with
FiLM-conditioned normalization.

| Hyperparameter | Value |
|---|---|
| Input patch | 96 × 96 × 96 (single channel) |
| Embedding dim | 64 |
| Stage channels / heads | (8, 16, 32, 64) |
| Depths | (2, 2, 2, 2) |
| Window size | 7 |
| Conditioning dim `d_cond` | 1 |
| Conditioning transform | `cond = ln(DRF) / 10` (`COND_TRANSFORM={"kind":"log_div","log":"natural","divisor":10.0,"version":1}`) |
| FiLM layer | `AdaLayerNorm`: LayerNorm(affine=False) + MLP(`d_cond`→128→`2·D`, SiLU), **identity-initialized** |

**FiLM mechanism.** Each conditioned LayerNorm predicts per-sample `(γ, β)` from the
conditioning scalar and applies `out = scale·(1+γ)·LN(x) + (bias+β)`. The MLP is
zero-initialized so the model starts identical to vanilla nnFormer and *learns* the
dose conditioning. The cond→scalar map is stored **in each checkpoint's config**, so
evaluation always reconstructs the exact transform used at training time
(`make_cond_fn`).

The DRF→cond values actually used:

| DRF | 4 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| cond = ln(DRF)/10 | 0.1386 | 0.2303 | 0.2996 | 0.3912 | 0.4605 |

---

## 4. Training protocol

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-5` |
| Loss | `0.95 · L1 + 0.05 · (1 − SSIM)` (per-sample, masked) |
| Patch / stride | 96³ / 80³ |
| DRF sampling | weighted, target mix `{4:5%, 10:15%, 20:20%, 50:25%, 100:35%}` |
| Validation | sliding-window, ROI 96³, **overlap 0.5**, gaussian; full per-DRF val every 10 epochs |
| Epochs | QUADRA 40, EXPLORER 50 |
| Precision | **fp32** on V100 / **bf16** on A100 — *never fp16* (see warning) |
| Effective batch | QUADRA 8 (4×V100, per-GPU 2), EXPLORER 9 (3×A100, per-GPU 3) |
| Seed | 42 |

> ### ⚠️ Precision warning — do not use fp16
> The Swin attention softmax overflows fp16's range (max 65504) after ~15 epochs,
> flooding the loss with NaNs and freezing training. Train and infer in **fp32**
> (any GPU) or **bf16** (Ampere+ / A100). The conditioning, data, and optimizer are
> all fp16-agnostic; the failure is purely numerical range in the forward pass.

---

## 5. Pretrained weights

The released checkpoints are **weights-only safetensors** (`model_state` + an embedded
`config`). They are **self-describing** — `infer_nifti.py` rebuilds the architecture
from `config` automatically — and contain everything needed for inference and
fine-tuning.

The weights ship **in this repo** under `weights/`, so a plain `git clone` already
gives you everything — no extra download step. They are also mirrored as
**[release assets](https://github.com/jlherraiz/ULDPET_nnFormer_FILM/releases/latest)**
if you'd rather grab a single file.

### The files (recommended format: safetensors)

Stripped, fp32, weights-only **safetensors** — 74 MB each (vs 221 MB for the full
`.pt`), and they load with **no code execution** (unlike pickle `.pt`). The
architecture config and normalization constants are embedded in the file metadata
(and mirrored in a `*.config.json` sidecar), so they are fully self-describing.

| Model | Scanner | Best epoch | In repo | Release mirror |
|-------|---------|-----------|---------|----------------|
| FiLM-nnFormer Quadra    | Siemens Biograph Vision Quadra | 40 / 40 | `weights/FILM_QUADRA_ep40_best.safetensors` | [download](https://github.com/jlherraiz/ULDPET_nnFormer_FILM/releases/download/v1.0.0/FILM_QUADRA_ep40_best.safetensors) |
| FiLM-nnFormer uEXPLORER | United Imaging uEXPLORER        | 20 / 50 | `weights/FILM_EXPLORER_ep20_best.safetensors` | [download](https://github.com/jlherraiz/ULDPET_nnFormer_FILM/releases/download/v1.0.0/FILM_EXPLORER_ep20_best.safetensors) |

To pull just one file (e.g. without cloning) use the GitHub CLI:

```bash
mkdir -p weights && gh release download v1.0.0 \
    -R jlherraiz/ULDPET_nnFormer_FILM \
    -p '*.safetensors' -p '*.config.json' -D weights/
```

or `curl`:

```bash
curl -L -o weights/FILM_QUADRA_ep40_best.safetensors \
  https://github.com/jlherraiz/ULDPET_nnFormer_FILM/releases/download/v1.0.0/FILM_QUADRA_ep40_best.safetensors
```

`infer_nifti.py` accepts `.safetensors` or `.pt` transparently. With safetensors you
don't even need `--percentiles` (p_low/p_high come from the embedded config):

```bash
python scripts/infer_nifti.py \
    --ckpt   weights/FILM_QUADRA_ep40_best.safetensors \
    --input  low_dose_scan.nii.gz --drf 100 \
    --output denoised.nii.gz
```

Load the weights yourself in a few lines:

```python
import json, torch
from safetensors import safe_open
from safetensors.torch import load_file
from nnFormer.nnFormer_film import nnFormerFiLM, make_cond_fn

path = "weights/FILM_QUADRA_ep40_best.safetensors"
with safe_open(path, framework="pt") as f:
    cfg = json.loads(f.metadata()["config"])          # arch + p_low/p_high + cond transform
model = nnFormerFiLM(crop_size=tuple(cfg["patch_size"]), embedding_dim=cfg["embedding_dim"],
                     num_heads=tuple(cfg["num_heads"]), depths=tuple(cfg["depths"]), d_cond=cfg["d_cond"])
model.load_state_dict(load_file(path)); model.eval()
cond_fn = make_cond_fn(cfg["COND_TRANSFORM"])         # cond_fn(DRF) -> conditioning scalar
```

The safetensors hold the model weights and are all you need for inference or
fine-tuning. The original full `.pt` checkpoints additionally carry optimizer and RNG
state — needed only to *resume training* from the exact stopping point — but they are
**not distributed here** (221 MB each, over GitHub's file limit). If you need them for
that purpose, contact the authors.

---

## 6. Inference

```bash
pip install -r requirements.txt

python scripts/infer_nifti.py \
    --ckpt   weights/FILM_QUADRA_ep40_best.safetensors \
    --input  low_dose_scan.nii.gz \
    --drf    100 \
    --output denoised.nii.gz
```

- `--drf` is the dose-reduction factor of the **input** (4/10/20/50/100; intermediate
  values also work — the conditioning interpolates).
- With **safetensors** the normalization percentiles come from the embedded config, so
  `--percentiles` is not needed. For a raw `.pt` checkpoint, add
  `--percentiles global_percentiles_QUADRA.json` (use the file **matching** the scanner
  the model was trained on).
- Output is de-normalized to the input intensity domain (Bq/mL), geometry preserved.
- Add `--bf16` on Ampere+; default is fp32. (Never fp16.)

The script reproduces the exact protocol: global-percentile normalization →
sliding-window (96³, overlap 0.5, gaussian) conditioned on the DRF → clamp ≥ 0 →
de-normalize → write with original spacing/origin/direction.

---

## 7. Results (FiLM vs per-DRF baseline)

Held-out validation, normalized [0,1] domain. Lower L1 / higher SSIM is better.

**QUADRA — FiLM (ep40) full validation (18 patients):**

| DRF | L1 | SSIM |
|---|---|---|
| 4   | 0.0010 | 0.9954 |
| 10  | 0.0012 | 0.9935 |
| 20  | 0.0014 | 0.9919 |
| 50  | 0.0017 | 0.9890 |
| 100 | 0.0020 | 0.9855 |

**QUADRA — paired FiLM vs per-DRF baseline** (identical pipeline, same 18 patients,
same overlap 0.5; only the model differs):

| DRF | FiLM L1 / SSIM | Baseline L1 / SSIM | Verdict |
|---|---|---|---|
| 4   | 0.0011 / 0.9955 | 0.0009 / 0.9957 | ≈ (baseline +0.0002 L1) |
| 10  | 0.0013 / 0.9933 | 0.0012 / 0.9932 | tie |
| 20  | 0.0014 / 0.9922 | 0.0014 / 0.9919 | tie |
| 50  | 0.0018 / 0.9888 | 0.0017 / 0.9888 | tie |
| 100 | 0.0021 / 0.9853 | 0.0020 / 0.9860 | ≈ (baseline +0.0007 SSIM) |

→ **One conditioned model ≈ five specialist models**, within measurement noise.

**EXPLORER — FiLM (ep20) full validation (≈49 patients):**

| DRF | L1 | SSIM |
|---|---|---|
| 4   | 0.0011 | 0.9943 |
| 10  | 0.0013 | 0.9920 |
| 20  | 0.0016 | 0.9893 |
| 50  | 0.0018 | 0.9859 |
| 100 | 0.0024 | 0.9796 |

(EXPLORER was trained to epoch 20 of a planned 50; it was still improving, with the
high-DRF tail closing toward the per-DRF baseline.)

---

## 8. Reproducibility checklist

- **Code:** `src/nnFormer/` (model) + `scripts/infer_nifti.py` (inference). Training
  uses the `film_train.py` pipeline (zarr data, weighted DRF sampler, DDP); see §4
  for every hyperparameter.
- **Determinism:** `seed=42` controls the train/val split and the per-epoch sampler.
- **Normalization constants:** shipped (`global_percentiles_*.json`).
- **Dependencies:** pinned in `requirements.txt`. Note `scipy` and `batchgenerators`
  are dragged in only by the nnFormer base class at import time and are unused at
  inference.
- **Environment used:** Python 3.12, torch 2.10.0 (cu126/cu128), MONAI 1.5.2,
  trained on 4×V100-32GB (QUADRA, fp32) and 3×A100-80GB (EXPLORER, bf16).
- **Clean-room check:** the model reconstructs and loads (`strict=True`) from only
  the bundled `src/nnFormer/` package + a checkpoint, with no dependency on the
  training repo.

---

## 9. Repository layout

```
.
├── README.md
├── MODEL_CARD.md
├── LICENSE.md
├── requirements.txt
├── global_percentiles_QUADRA.json      # p_low/p_high for Vision Quadra
├── global_percentiles_EXPLORER.json    # p_low/p_high for uEXPLORER
├── scripts/
│   └── infer_nifti.py                  # self-contained NIfTI inference
└── src/
    └── nnFormer/                       # model package (nnFormerFiLM + FiLM layers)
```

## 10. Citation / attribution

Built on nnFormer (https://github.com/282857341/nnFormer). If you use these models,
please cite nnFormer and this repository. See `MODEL_CARD.md` for intended use and
limitations.

**Data.** The training data is from the **UDPET Challenge** dataset
(<https://udpet-challenge.github.io/>). If you use it, please cite the dataset paper:

```bibtex
@InProceedings{XueSon_UDPET_MICCAI2025,
  author    = {Xue, Song and Wang, Hanzhong and Chen, Yizhou and Liu, Fanxuan and Zhu, Hong and Viscione, Marco and Guo, Rui and Rominger, Axel and Li, Biao and Shi, Kuangyu},
  title     = {UDPET: Ultra-low Dose PET Imaging Challenge Dataset},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  series    = {LNCS},
  volume    = {15972},
  pages     = {616--623},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-05169-1_59}
}
```
