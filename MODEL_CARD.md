# Model Card — FiLM-nnFormer low-dose PET denoiser

## Overview
A single FiLM-conditioned nnFormer per scanner that denoises low-dose PET across
dose-reduction factors (DRFs) 4–100, replacing a family of per-DRF specialist
models with one DRF-conditioned model. Two checkpoints are released: one for
Siemens Biograph Vision Quadra, one for United Imaging uEXPLORER.

## Intended use
- **Primary:** research on dose-reduction / denoising of static FDG-PET volumes from
  the Vision Quadra or uEXPLORER scanners, for the DRFs the model was trained on
  (4, 10, 20, 50, 100) and reasonable intermediate values (conditioning interpolates).
- **Inputs:** a 3D PET volume (NIfTI) in the same intensity domain (Bq/mL) as
  training, plus its DRF.
- **Outputs:** a denoised 3D volume de-normalized to Bq/mL, original geometry preserved.

## Out of scope / cautions
- Not a clinical device; **not for diagnostic use**. No regulatory clearance.
- Scanner-specific: use the QUADRA model on Quadra data and the EXPLORER model on
  uEXPLORER data, each with its own normalization constants. Cross-scanner use is
  untested.
- Trained on FDG static images; other tracers, dynamic frames, or very different
  reconstruction settings are out of distribution.
- Absolute quantification: outputs track the per-DRF baseline to within ~±3% in
  total activity / mean intensity. Treat absolute SUV/total-activity readouts as
  approximate.
- DRFs far outside 4–100 are extrapolation and unvalidated.

## Training data
- QUADRA: 377 patients (359 train / 18 val); EXPLORER: 994 patients (945 / 49).
- Patient-level 95/5 split, seed 42. Full-dose target + simulated low-dose inputs at
  DRF 4/10/20/50/100 per patient.
- Global min–max percentile normalization (0.1/99.9 of training targets); constants
  shipped with the repo. Raw clinical data is not redistributed.

## Evaluation
- Protocol: sliding-window (96³, overlap 0.5, gaussian), metrics L1 and SSIM
  (data_range=1) on the normalized [0,1] domain, on the held-out val set.
- Paired comparison vs per-DRF baseline (same patients, same pipeline, only the model
  differs) shows parity within ±0.0002 L1 / ±0.0007 SSIM on QUADRA. See README §7.

## Architecture & precision
- nnFormerFiLM, patch 96³, embedding 64, heads (8,16,32,64), depths (2,2,2,2),
  FiLM via identity-initialized `AdaLayerNorm`; cond = ln(DRF)/10.
- **Run in fp32 or bf16. Never fp16** — the attention softmax overflows fp16 range
  and produces NaNs.

## Limitations summary
Single-tracer, scanner-specific, research-only; EXPLORER checkpoint is from epoch 20
of a planned 50 (still improving at high DRF). Quantitative SUV use should be
validated by the user.

## Attribution
Derived from nnFormer (https://github.com/282857341/nnFormer). License: see LICENSE.md.
