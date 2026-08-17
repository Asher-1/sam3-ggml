# Model Zoo — models/

This directory holds ready-to-run **GGUF** models for [sam3.cpp](../README.md).
Each file name encodes three things:

```
<family>_<backbone/size>_<precision>.gguf
```

| Part | Meaning |
|------|---------|
| `sam3` / `sam3-visual` | **SAM 3** — ViT-32 backbone + text encoder + DETR detector (850M params) |
| `sam2` / `sam2.1` | **SAM 2 / SAM 2.1** — Meta's Hiera-backbone segmentation models (visual only) |
| `edgetam` | **EdgeTAM** — RepViT-M1 + Perceiver decoder, the mobile/embedded variant of the SAM 2 family |
| `tiny` / `small` / `base_plus` / `large` | Backbone size (39M / 46M / 81M / 224M params) |
| `f32` / `f16` / `q8_0` / `q4_1` / `q4_0` | Weight precision (see [Precision guide](#precision-guide)) |

> **Architecture lineage**: this directory covers **3 architectures** —
> **SAM 3** (`sam3-*`, `sam3-visual-*`), the **SAM 2 family** (`sam2*`, `sam2.1*`,
> `edgetam*`), and **no SAM 1** checkpoints (SAM 1 / ViT-B/L/H is a separate
> architecture not shipped by this project). The SAM 2 family is visual-only
> (points/box + tracking); **SAM 3 full adds text-prompted detection (PCS)**:
> type `"cat"` and get every cat in the image.

## Quick pick

| You want… | Pick |
|-----------|------|
| **Text-prompted detection** ("type *cat*, get every cat") | `sam3-f16.gguf` (1.8 GB) or `sam3-q8_0.gguf` (1.1 GB) |
| Best visual quality-to-speed balance on GPU | `sam2.1_hiera_base_plus_f16.gguf` (156 MB) |
| Fastest interactive point/box segmentation on any device | `sam2.1_hiera_tiny_q4_0.gguf` (23 MB) |
| Fastest on mobile/embedded / lowest memory | `edgetam_q4_0.gguf` (16 MB) |
| Best segmentation quality | `sam2.1_hiera_large_f16.gguf` (431 MB) or `_q8_0` (231 MB) |
| Debugging / numerical reference (never for deployment) | `sam2.1_hiera_tiny_f32.gguf` |

## Model files

Sizes below are the actual `.gguf` files in this directory. Latency is a
single-image PVS run (encode + segment) at 1008×1008 on **RTX 3060 CUDA**,
point (315,250) on `tests/cat.jpg`, 2-run min — measured with the same
`bench_pvs` tool across every model. `score` = mask IoU confidence.

### SAM 3 (850M params — ViT-32 backbone + text encoder + DETR decoder)

| File | Size | Load | Encode | Segment | Total | score |
|------|------|-----:|-------:|--------:|------:|------:|
| `sam3-f32.gguf` | 3.3 GB | 3.3 s | 4.2 s | 0.21 s | 7.7 s | 0.953 |
| `sam3-f16.gguf` | 1.8 GB | 1.6 s | 3.6 s | 0.22 s | 5.4 s | 0.952 |
| `sam3-q8_0.gguf` | 1.1 GB | 1.4 s | 3.5 s | 0.20 s | 5.1 s | 0.953 |
| `sam3-q4_1.gguf` | 730 MB | 1.1 s | 3.6 s | 0.21 s | 4.9 s | 0.937 |
| `sam3-q4_0.gguf` | 707 MB | 1.5 s | 3.6 s | 0.25 s | 5.3 s | 0.915 |

**Best for**: text-prompted detection (PCS) + point/box segmentation (PVS) +
video tracking in one model. The full SAM 3 is the only family here that
supports text prompts; the visual path matches `sam3-visual` exactly.

### SAM 3 Visual (no text encoder — PVS + tracking only)

| File | Size | Load | Encode | Segment | Total | score |
|------|------|-----:|-------:|--------:|------:|------:|
| `sam3-visual-f16.gguf` | 902 MB | 1.5 s | 2.3 s | 0.22 s | 4.0 s | 0.952 |
| `sam3-visual-q8_0.gguf` | 494 MB | 0.7 s | 2.2 s | 0.22 s | 3.1 s | 0.953 |
| `sam3-visual-q4_1.gguf` | 303 MB | 0.7 s | 2.2 s | 0.21 s | 3.1 s | 0.937 |
| `sam3-visual-q4_0.gguf` | 276 MB | 0.6 s | 2.2 s | 0.23 s | 3.0 s | 0.915 |

**Best for**: SAM 3-quality segmentation without the text encoder — half the
size and ~40% faster than full SAM 3. Same PVS + tracking capabilities as
`sam2.1_hiera_base_plus` but with the stronger SAM 3 backbone.

### EdgeTAM (SAM 2 family — mobile variant, RepViT-M1 backbone)

| File | Size | Load | Encode | Segment | Total | score |
|------|------|-----:|-------:|--------:|------:|------:|
| `edgetam_f16.gguf` | 28 MB | 0.63 s | 0.54 s | 0.19 s | 1.4 s | 0.495 |
| `edgetam_q8_0.gguf` | 20 MB | 0.60 s | 0.50 s | 0.21 s | 1.3 s | 0.495 |
| `edgetam_q4_0.gguf` | 16 MB | 0.57 s | 0.51 s | 0.17 s | 1.2 s | 0.439 |

**Best for**: embedded / mobile / battery-powered devices, video tracking with
tight memory budgets. ~7× faster than SAM 2 Tiny on GPU. Note: EdgeTAM's mask
scores on single-point prompts are inherently lower (0.44–0.50) and its mask
contains more salt noise than Hiera models — this matches the official
PyTorch EdgeTAM behavior, not a porting defect (verified against
facebookresearch/EdgeTAM with the same prompt).

### SAM 2 (Hiera backbone, visual only)

| File | Size | Load | Encode | Segment | Total | score |
|------|------|-----:|-------:|--------:|------:|------:|
| `sam2_hiera_tiny_f16.gguf` | 76 MB | 0.79 s | 1.43 s | 0.37 s | 2.6 s | 0.959 |
| `sam2_hiera_tiny_f32.gguf` | 149 MB | 0.90 s | 0.98 s | 0.20 s | 2.1 s | 0.959 |
| `sam2_hiera_tiny_q8_0.gguf` | 41 MB | 0.39 s | 0.82 s | 0.16 s | 1.4 s | 0.959 |
| `sam2_hiera_tiny_q4_1.gguf` | 25 MB | 0.35 s | 0.81 s | 0.16 s | 1.3 s | 0.930 |
| `sam2_hiera_tiny_q4_0.gguf` | 23 MB | 0.43 s | 0.83 s | 0.17 s | 1.4 s | 0.933 |
| `sam2_hiera_base_plus_f16.gguf` | 156 MB | 0.82 s | 1.22 s | 0.20 s | 2.2 s | 0.957 |
| `sam2_hiera_base_plus_f32.gguf` | 309 MB | 0.97 s | 1.22 s | 0.17 s | 2.4 s | 0.957 |
| `sam2_hiera_base_plus_q8_0.gguf` | 84 MB | 0.93 s | 1.57 s | 0.28 s | 2.8 s | 0.955 |
| `sam2_hiera_base_plus_q4_1.gguf` | 51 MB | 0.73 s | 1.16 s | 0.18 s | 2.1 s | 0.954 |
| `sam2_hiera_base_plus_q4_0.gguf` | 46 MB | 0.65 s | 1.19 s | 0.22 s | 2.1 s | 0.952 |
| `sam2_hiera_large_f16.gguf` | 430 MB | 1.42 s | 1.47 s | 0.25 s | 3.1 s | 0.909 |

### SAM 2.1 (improved SAM 2, same Hiera architecture)

| File | Size | Load | Encode | Segment | Total | score |
|------|------|-----:|-------:|--------:|------:|------:|
| `sam2.1_hiera_tiny_f32.gguf` | 149 MB | 0.78 s | 1.11 s | 0.21 s | 2.1 s | 0.943 |
| `sam2.1_hiera_tiny_f16.gguf` | 76 MB | 0.60 s | 1.03 s | 0.23 s | 1.9 s | 0.943 |
| `sam2.1_hiera_tiny_q8_0.gguf` | 41 MB | 0.62 s | 1.11 s | 0.24 s | 2.0 s | 0.945 |
| `sam2.1_hiera_tiny_q4_1.gguf` | 25 MB | 0.62 s | 0.95 s | 0.19 s | 1.8 s | 0.956 |
| `sam2.1_hiera_tiny_q4_0.gguf` | 23 MB | 0.73 s | 1.13 s | 0.25 s | 2.1 s | 0.927 |
| `sam2.1_hiera_small_f32.gguf` | 176 MB | 0.69 s | 0.88 s | 0.18 s | 1.8 s | 0.945 |
| `sam2.1_hiera_small_f16.gguf` | 90 MB | 0.75 s | 0.99 s | 0.19 s | 1.9 s | 0.945 |
| `sam2.1_hiera_small_q8_0.gguf` | 48 MB | 0.61 s | 0.96 s | 0.19 s | 1.8 s | 0.944 |
| `sam2.1_hiera_small_q4_1.gguf` | 30 MB | 0.72 s | 1.27 s | 0.19 s | 2.2 s | 0.947 |
| `sam2.1_hiera_small_q4_0.gguf` | 27 MB | 0.62 s | 1.11 s | 0.19 s | 1.9 s | 0.949 |
| `sam2.1_hiera_base_plus_f32.gguf` | 309 MB | 0.99 s | 1.25 s | 0.21 s | 2.5 s | 0.953 |
| `sam2.1_hiera_base_plus_f16.gguf` | 156 MB | 0.71 s | 1.09 s | 0.18 s | 2.0 s | 0.953 |
| `sam2.1_hiera_base_plus_q8_0.gguf` | 84 MB | 0.79 s | 1.40 s | 0.21 s | 2.4 s | 0.954 |
| `sam2.1_hiera_base_plus_q4_1.gguf` | 51 MB | 0.74 s | 1.44 s | 0.24 s | 2.4 s | 0.944 |
| `sam2.1_hiera_base_plus_q4_0.gguf` | 46 MB | 0.79 s | 1.34 s | 0.23 s | 2.4 s | 0.936 |
| `sam2.1_hiera_large_f32.gguf` | 857 MB | 1.42 s | 1.57 s | 0.18 s | 3.2 s | 0.940 |
| `sam2.1_hiera_large_f16.gguf` | 431 MB | 1.00 s | 1.29 s | 0.17 s | 2.5 s | 0.940 |
| `sam2.1_hiera_large_q8_0.gguf` | 231 MB | 0.76 s | 1.44 s | 0.19 s | 2.4 s | 0.938 |
| `sam2.1_hiera_large_q4_1.gguf` | 138 MB | 0.75 s | 1.68 s | 0.23 s | 2.7 s | 0.928 |
| `sam2.1_hiera_large_q4_0.gguf` | 124 MB | 0.77 s | 1.49 s | 0.23 s | 2.5 s | 0.900 |

## Charts

- [Latency chart — ALL 43 models (RTX 3060 CUDA)](../media/benchmark_latency_all_models.png)
- [Effect grid — ALL 43 models on cat.jpg](../media/benchmark_effect_all_models.png)

## Precision guide

| Precision | Relative size | Quality | Use |
|-----------|---------------|---------|-----|
| `f32` | 1.0× | reference | Debugging, numerical checks only — never deploy |
| `f16` | 0.5× | ≈ f32 | **Recommended default** — near-lossless, half the size |
| `q8_0` | 0.25× | very close to f16 | Big models (large/`sam3`) when f16 is too big |
| `q4_1` | ~0.14× | good (retains scale + offset) | Aggressive size cuts with better fidelity than q4_0 |
| `q4_0` | ~0.13× | acceptable for interactive use | Smallest files; quality gap is visible on thin structures |

## Size selection guide

| Need | SAM 3 | SAM 3 Visual | base_plus | tiny | EdgeTAM |
|------|-------|--------------|-----------|------|---------|
| Text prompts (PCS) | Yes | - | - | - | - |
| PVS + tracking | Yes | Yes | Yes | Yes | Yes |
| Encode latency (RTX 3060) | 3.5–4.2 s | 2.2 s | ~1.1–1.6 s | ~0.8–1.1 s | ~0.5 s |
| Size (f16) | 1.8 GB | 902 MB | 156 MB | 76 MB | 28 MB |

- **SAM 2 vs SAM 2.1**: prefer **2.1** for new projects (better training data and
  tracking; same architecture, same speed, same sizes).
- **Video tracking**: tiny or EdgeTAM are the only practical choices for
  interactive playback on CPU; larger backbones work well on GPU.
- **Point/box (PVS) + tracking** work on every model here; **text-prompted
  detection (PCS) requires a SAM 3 checkpoint** (the `sam3-*` files above).
