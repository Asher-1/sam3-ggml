# sam3.cpp

State-of-the-art image and video segmentation in portable C/C++

![SAM3 Image Segmentation Demo](media/image_demo.gif)
![SAM3 Video Segmentation Demo](media/video_demo.gif)

---

## Why sam3.cpp?

Running Meta's Segment Anything models typically requires Python, PyTorch, and a CUDA GPU. **sam3.cpp** eliminates all of that. It's a single C++ library that runs SAM 2, SAM 2.1, and SAM 3 inference on CPU, CUDA, Vulkan and Apple Metal. No Python runtime, no heavyweight dependencies. Just compile and segment.

- **3 supported releases**: SAM 2, SAM 2.1 (Hiera), and SAM 3 (ViT + text detection)
- **4-bit quantization**: SAM 2.1 Tiny in **22 MB** at ~1 fps on Metal, SAM 3 down to 673 MB
- **CUDA / Vulkan / Apple Metal GPU acceleration** for the full backbone and transformer decoder
- **Text-prompted detection** (SAM 3 only): type `"cat"` and get every cat in the image, no clicks needed
- **Point/box segmentation + video tracking** with memory bank across all models
- **Single-file library**: `sam3.cpp` + `sam3.h`, C++14, no exceptions, no inheritance
- **Standard GGUF model format** — every model loads through `gguf_init_from_file`, the same interface as face-detect-ggml, free-splatter.cpp, OpenPCDet-GGML and the rest of the ggml ecosystem
- **Zero dependencies** beyond [ggml](https://github.com/ggerganov/ggml) and [stb](https://github.com/nothings/stb)

## Quick Start

```bash
# Clone
# 1. switch the ggml submodule to the official ggml-org/ggml @ v0.21.0
#    (the fork URL in .gitmodules is updated; run `git submodule update --init`)
git clone --recursive https://github.com/PABannier/sam3.cpp
cd sam3.cpp

# Build (GPU backends are opt-in)
mkdir build && cd build
cmake ..                       # CPU only
cmake .. -DSAM3_CUDA=ON        # + NVIDIA CUDA
cmake .. -DSAM3_VULKAN=ON      # + Vulkan (AMD/Intel/NVIDIA)
make -j

# Download a model (SAM 2.1 Tiny, 75 MB) — auto-repacked to .gguf
bash ../scripts/download_models.sh --filter tiny
# or download + repack manually:
#   curl -L -o ../models/sam2.1_hiera_tiny_f16.ggml \
#     https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_tiny_f16.ggml
#   python3 ../scripts/convert_ggml_to_gguf.py ../models/sam2.1_hiera_tiny_f16.ggml

# Segment an image interactively (requires SDL2)
./examples/sam3_image --model ../models/sam2.1_hiera_tiny_f16.gguf --image ../data/test_image.jpg

# Track objects in a video interactively (requires SDL2)
./examples/sam3_video --model ../models/sam2.1_hiera_tiny_f16.gguf --video ../data/test_video.mp4
```

The interactive apps use SDL2 + ImGui. If SDL2 isn't found, only the benchmark and quantize tools are built.

## Benchmarks

Video object tracking latency on **Apple M4 Pro (24 GB)**, 5 frames at 1008x1008 resolution, 4 threads. Each run is isolated in a forked subprocess.

### SAM 3 (Full: text detection + visual tracking)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam3-f32 | 3.2 GB | - | 40.5 | - | 200.4 |
| sam3-f16 | 1.7 GB | **7.7** | 23.8 | 38.1 | 117.5 |
| sam3-q8_0 | 1.0 GB | **7.8** | 23.3 | 38.7 | 115.2 |
| sam3-q4_1 | 756 MB | - | 24.5 | - | 120.9 |
| sam3-q4_0 | 673 MB | **7.8** | 23.9 | 38.7 | 117.7 |

### SAM 3 Visual-Only (no text encoder, tracking only)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam3-visual-f16 | 901 MB | **6.6** | 22.6 | 32.7 | 111.2 |
| sam3-visual-q8_0 | 493 MB | **6.7** | 22.0 | 33.0 | 108.4 |
| sam3-visual-q4_1 | 318 MB | - | 23.1 | - | 113.9 |
| sam3-visual-q4_0 | 275 MB | **6.7** | 22.3 | 33.0 | 110.0 |

### SAM 2 / SAM 2.1 (Hiera backbone)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam2_hiera_tiny_f16 | 75 MB | **0.9** | 2.7 | 4.0 | 12.6 |
| sam2_hiera_tiny_q8_0 | 40 MB | **0.9** | 2.5 | 4.0 | 11.7 |
| sam2_hiera_tiny_q4_0 | **22 MB** | **0.9** | 2.5 | 4.0 | 11.7 |
| sam2_hiera_small_f16 | 89 MB | **0.9** | 2.9 | 4.1 | 13.7 |
| sam2_hiera_small_q8_0 | 47 MB | **0.9** | 2.7 | 4.1 | 12.5 |
| sam2_hiera_small_q4_0 | 26 MB | **0.9** | 2.7 | 4.1 | 12.7 |
| sam2_hiera_base_plus_f16 | 155 MB | **1.0** | 4.2 | 4.7 | 20.2 |
| sam2_hiera_base_plus_q8_0 | 83 MB | - | 3.9 | - | 18.9 |
| sam2_hiera_large_f16 | 429 MB | - | 8.4 | - | 40.9 |
| sam2_hiera_large_q8_0 | 230 MB | - | 7.6 | - | 37.1 |
| | | | | | |
| sam2.1_hiera_tiny_f16 | 75 MB | **0.8** | 2.6 | 4.0 | 12.3 |
| sam2.1_hiera_tiny_q8_0 | 40 MB | **0.9** | 2.4 | 4.0 | 11.4 |
| sam2.1_hiera_tiny_q4_0 | **22 MB** | **0.9** | 2.5 | 4.0 | 11.5 |
| sam2.1_hiera_small_f16 | 89 MB | **0.9** | 2.9 | 4.1 | 13.5 |
| sam2.1_hiera_small_q8_0 | 47 MB | **0.9** | 2.7 | 4.1 | 12.5 |
| sam2.1_hiera_small_q4_0 | 26 MB | **0.9** | 2.7 | 4.1 | 12.6 |
| sam2.1_hiera_base_plus_f16 | 155 MB | **1.0** | 4.2 | 4.7 | 20.1 |
| sam2.1_hiera_base_plus_q8_0 | 83 MB | - | 3.9 | - | 18.6 |
| sam2.1_hiera_large_f16 | 430 MB | - | 8.5 | - | 41.4 |
| sam2.1_hiera_large_q8_0 | 230 MB | - | 7.7 | - | 37.7 |

### CUDA / Vulkan (RTX 3060, Linux)

SAM 2.1 Tiny family, 2 frames at 1024x1024, measured on this repo's CI box.
CUDA and Vulkan run the full backbone + decoder on the GPU; the reported
`Track/frame` is the steady-state per-frame cost (encode amortized).

| Model | Size | Track/fr CUDA (ms) | Track/fr Vulkan (ms) | Track/fr CPU (ms) |
|-------|------|--------------------|----------------------|-------------------|
| sam2.1_hiera_tiny_f32 | 148 MB | 695 | 883 | 3969 |
| sam2.1_hiera_tiny_f16 | 75 MB | 651 | 1062 | 4182 |
| sam2.1_hiera_tiny_q8_0 | 40 MB | 648 | 906 | 4091 |
| sam2.1_hiera_tiny_q4_0 | 22 MB | 626 | 1003 | 4405 |

CUDA is the fastest backend on NVIDIA hardware; Vulkan offers a vendor-neutral
alternative (AMD/Intel/NVIDIA) at roughly 1.5x the CUDA latency.

### SAM 3 PVS CUDA latency (RTX 3060, Linux)

Full `sam3-f16.gguf`, 1008x1008 input, point `(315,250)` on `tests/data/cat.jpg`,
2 warmups followed by 7 timed runs. Model loading is excluded.

| Encode p50 | Segment p50 | Inference p50 | Mask score |
|-----------:|------------:|--------------:|-----------:|
| 566.167 ms | 31.925 ms | **598.092 ms** | 0.9527 |

Point/box-only callers should use `sam3_encode_image_pvs()`. It skips the
detector neck for full SAM 3 checkpoints while keeping the visual features
consumed by PVS and tracking unchanged.


<details>
<summary><b>Reproduce these benchmarks</b></summary>

```bash
# Full benchmark (all models, both backends)
./build/examples/sam3_benchmark

# GPU only, all models
./build/examples/sam3_benchmark --gpu-only

# Quick iteration (tiny models, 3 frames)
./build/examples/sam3_benchmark --filter tiny --n-frames 3 --gpu-only

# CPU only, specific model
./build/examples/sam3_benchmark --cpu-only --filter sam2.1_hiera_small
```

Options: `--models-dir <path>`, `--video <path>`, `--n-frames <n>`, `--n-threads <n>`, `--filter <substr>`, `--cpu-only`, `--gpu-only`

</details>

## Model Zoo

All models ship in the standard **GGUF** format. The SAM 3 family (text-prompted
PCS + tracking) is published ready-to-run on
**[huggingface.co/Asher-1/sam3-gguf](https://huggingface.co/Asher-1/sam3-gguf)**
— download straight from there (see [models/README.md](models/README.md)).
The legacy upstream repo (PABannier/sam3.cpp, `.ggml` +
`scripts/download_models.sh` repack, or `scripts/convert_ggml_to_gguf.py`) also
serves SAM 2 variants; note its SAM 3 `.ggml` files predate the tokenizer merge
table fix (6 `#`-first merges missing) and the tracker-alignment hparams —
prefer the Asher-1 GGUFs, or run `scripts/fix_gguf_merges.py` on repacked ones.
For converting from the original `sam3.pt`, use
`scripts/download_model.sh` (fetches the official BPE vocab from GitHub and
embeds the complete tokenizer via `--bpe-gz`).

The supported model set contains 49 GGUF variants across the SAM 2 and SAM 3
architectures, with multiple sizes and up to five precisions.

For per-file sizes, use-case guidance and measured GPU latency of every model
(including the SAM 3 family), see **[models/README.md](models/README.md)**.

### SAM 3 (850M params, ViT-32 backbone + text encoder + DETR decoder)

| Variant | Precision | Size | Features |
|---------|-----------|------|----------|
| sam3 | f32 | 3.4 GB | Text detection (PCS) + point/box segmentation (PVS) + video tracking |
| sam3 | f16 | 1.7 GB | Same |
| sam3 | q8_0 | 1.0 GB | Same |
| sam3 | q4_1 | 756 MB | Same |
| sam3 | q4_0 | 707 MB | Same |
| sam3-visual | f16 | 946 MB | Point/box segmentation (PVS) + video tracking (no text) |
| sam3-visual | q8_0 | 517 MB | Same |
| sam3-visual | q4_1 | 318 MB | Same |
| sam3-visual | q4_0 | 289 MB | Same |

### SAM 2 / SAM 2.1 (Hiera backbone, 4 sizes)

| Family | Size | Params | f32 | f16 | q8_0 | q4_1 | q4_0 |
|--------|------|--------|-----|-----|------|------|------|
| SAM 2 | Tiny | 39M | 156 MB | 79 MB | 43 MB | 26 MB | 24 MB |
| SAM 2 | Small | 46M | 184 MB | 94 MB | 50 MB | 30 MB | 28 MB |
| SAM 2 | Base+ | 81M | 323 MB | 163 MB | 88 MB | 53 MB | 48 MB |
| SAM 2 | Large | 224M | 898 MB | 451 MB | 241 MB | 144 MB | 130 MB |
| SAM 2.1 | Tiny | 39M | 156 MB | 79 MB | 43 MB | 26 MB | 24 MB |
| SAM 2.1 | Small | 46M | 184 MB | 94 MB | 50 MB | 30 MB | 28 MB |
| SAM 2.1 | Base+ | 81M | 323 MB | 163 MB | 88 MB | 53 MB | 48 MB |
| SAM 2.1 | Large | 224M | 898 MB | 451 MB | 241 MB | 144 MB | 130 MB |

### Feature Matrix

| Capability | SAM 3 | SAM 3 Visual | SAM 2 / 2.1 |
|-----------|-------|-------------|-------------|
| Text-prompted detection (PCS) | Yes | - | - |
| Point/box segmentation (PVS) | Yes | Yes | Yes |
| Multi-mask output | Yes | Yes | Yes |
| Video tracking (memory bank) | Yes | Yes | Yes |
| Interactive refinement | Yes | Yes | Yes |
| Offline result visualization (PNG) | Yes | Yes | Yes |
| IoM mask dedup (agent-aligned) | Yes | Yes | Yes |
| Soft mask logits output (opt-in) | Yes | Yes | Yes |
| Image mask prompt (SAM1-task style) | - | Yes | Yes |
| Session edit: remove tracked instance | - | Yes | Yes |
| Session edit: rewind to earlier frame | - | Yes | Yes |
| Session edit: clear frame prompt | - | Yes | Yes |
| Quantization (Q4/Q8) | Yes | Yes | Yes |
| Metal GPU | Yes | Yes | Yes |
| CUDA / Vulkan GPU | Yes | Yes | Yes |

## Building from Source

### Prerequisites

- C++14 compiler (Clang, GCC, MSVC)
- CMake 3.14+
- (Optional) SDL2 for the interactive image/video examples
- (Optional) ffmpeg for video frame decoding

### Build

```bash
git clone --recursive https://github.com/PABannier/sam3.cpp
cd sam3.cpp
mkdir build && cd build
cmake ..
make -j
```

By default only the CPU backend is built. Enable the GPU backends you need:

```bash
cmake .. -DSAM3_CUDA=ON      # NVIDIA CUDA (needs CUDA toolkit)
cmake .. -DSAM3_VULKAN=ON    # Vulkan (needs Vulkan SDK; works on AMD/Intel/NVIDIA)
```

The backends are additive — you can enable **CUDA + Vulkan + CPU in one build**
so a single binary can switch backends at runtime. Use the helper script (it
also auto-detects your GPU's CUDA arch and builds SDL2 from source if the dev
package is missing):

```bash
# CPU + CUDA + Vulkan in one binary (build-all/)
bash scripts/build_multi_backend.sh

# Subsets / options
bash scripts/build_multi_backend.sh --cuda-only
bash scripts/build_multi_backend.sh --vulkan-only
bash scripts/build_multi_backend.sh --cpu-only
bash scripts/build_multi_backend.sh --jobs 8 --build-dir build-all
```

The GUI examples (`sam3_image`, `sam3_video`) then expose a **Devices** combo
(Auto / CPU / CUDA / Vulkan) in the top bar. **Auto** probes the ggml backend
registry at load time in the order CUDA → Vulkan → CPU and uses the first one
that initialises. `--device auto|cpu|cuda|vulkan` does the same from the
command line; the combo re-loads the model onto the selected backend on the
fly. The `Backend:` label next to it shows which device the model is actually
running on.

On macOS, Metal is enabled automatically (`SAM3_METAL=OFF` disables it).
The ggml submodule is pinned to the official `ggml-org/ggml` at v0.21.0; the
local patches in `ggml-patches/` (Metal/CUDA attention support plus CUDA
F16 matmul output, fused RoPE/window layout and fused QKV layout+RoPE) are applied automatically by CMake at configure time via
`scripts/apply_ggml_patches.sh`.

To build tests:

```bash
cmake .. -DSAM3_BUILD_TESTS=ON
make -j
```

## Usage

### Image Segmentation (Interactive GUI)

```bash
# Point/box segmentation with any model
./sam3_image --model models/sam2.1_hiera_tiny_f16.gguf --image photo.jpg

# Text-prompted detection (SAM 3 only)
./sam3_image --model models/sam3.1-f16.gguf --image photo.jpg
# → Type "cat" in the text field, click [Segment]
```

**Controls:**
- **Left-click**: add positive point
- **Right-click**: add negative point
- **Drag**: draw bounding box
- **Text field + Segment**: detect all instances matching the text prompt (SAM 3 only)
- **Export**: save masks as PNG

### Video Tracking (Interactive GUI)

```bash
# Visual tracking (SAM 2/2.1/3)
./sam3_video --model models/sam2.1_hiera_small_f16.gguf --video input.mp4

# Text-prompted tracking (SAM 3 only)
./sam3_video --model models/sam3.1-f16.gguf --video input.mp4
```

**Controls:**
- Click a point or draw a box on a paused frame to add an instance
- Click on an existing tracked mask to refine it
- Play/Pause/Step for playback
- Export per-frame mask PNGs

### C++ API

```cpp
#include "sam3.h"

// Load model
sam3_params params;
params.model_path = "models/sam2.1_hiera_tiny_f16.gguf";
params.use_gpu    = true;
params.n_threads  = 4;

auto model = sam3_load_model(params);
auto state = sam3_create_state(*model, params);

// Encode PVS/tracking features (call once, reuse for multiple prompts)
auto image = sam3_load_image("photo.jpg");
sam3_encode_image_pvs(*state, *model, image);

// Segment with a point click
sam3_pvs_params pvs;
pvs.pos_points.push_back({315.0f, 250.0f});
sam3_result result = sam3_segment_pvs(*state, *model, pvs);

for (auto& det : result.detections) {
    sam3_save_mask(det.mask, "mask.png");
}
```

```cpp
// Text-prompted detection (SAM 3 full model only)
sam3_pcs_params pcs;
pcs.text_prompt     = "yellow school bus";
pcs.score_threshold = 0.5f;
sam3_result result = sam3_segment_pcs(*state, *model, pcs);
// → result.detections contains every matching instance
```

```cpp
// Video tracking
auto tracker = sam3_create_visual_tracker(*model, {});

// Frame 0: encode + add instance with a click
sam3_encode_image(*state, *model, frame0);
sam3_pvs_params pvs;
pvs.pos_points.push_back({315.0f, 250.0f});
sam3_tracker_add_instance(*tracker, *state, *model, pvs);

// Subsequent frames: propagate masks
for (int f = 1; f < n_frames; f++) {
    sam3_result result = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
    // result.detections[i].mask - tracked mask for each instance
}

// Reverse propagation (official propagate_in_video(reverse=True) alignment):
// walks decreasing frame indices, never re-processes the prompt frame
// (official range(start-1, ..., -1)), and is rejected at frame 0. Memory and
// object pointers on the reverse side participate with tracking-order
// temporal positions. Pass the frame whose index equals
// (tracker.frame_index - 1).
sam3_result back = sam3_propagate_frame(*tracker, *state, *model, frames[f - 1],
                                       /*reverse=*/true);
```

```cpp
// Seed a tracked instance from an existing mask instead of points/box.
// The mask is written straight into the memory bank, so propagation tracks
// exactly the supplied mask (e.g. one produced by an earlier PVS/PCS call,
// or an external segmentation).
auto tracker = sam3_create_visual_tracker(*model, {});
sam3_encode_image(*state, *model, frame0);
sam3_mask seed = /* 0/255 binary mask, any resolution */;
sam3_tracker_add_instance_from_mask(*tracker, *state, *model, seed);
// then sam3_propagate_frame(...) on subsequent frames as above

// Session editing: drop a mis-detected / stale instance by ID (official
// remove_object alignment). Propagation skips it from the next frame on.
sam3_tracker_remove_instance(*tracker, instance_id);

// Rewind to an earlier frame to add a prompt there (official
// add_prompt-at-earlier-frame + propagate_in_video(start_frame_idx=...)
// alignment): memory recorded after that frame is dropped, instances that
// first appeared after it are removed. Then re-prompt and replay frames.
sam3_tracker_rewind(*tracker, frame_index);
sam3_tracker_add_instance(*tracker, *state, *model, pvs);
// ... sam3_propagate_frame(...) from frame_index onward

// Clear one instance's prompt/conditioning at one frame (official
// clear_all_points_in_frame alignment).
sam3_tracker_clear_instance_frame(*tracker, instance_id, frame_index);
```

```cpp
// Batch image inference (official Sam3Processor::set_image_batch alignment):
// encode a batch once, then query it with one text prompt. Results are
// per-image and identical to encode-and-query-each separately (the detector
// attends within each image only). SAM3 full checkpoints only.
sam3_image imgs[2] = {img_a, img_b};
sam3_encode_image_batch(*state, *model, imgs, 2);
sam3_result results[2];
sam3_segment_pcs_batch(*state, *model, pcs, results, 2);
```

#### Numerical alignment notes

The tracker mirrors the official SAM3 video predictor's memory behavior:
conditioning frames are selected with the official `select_closest_cond_frames`
cap (4), non-conditioning memory and object pointers only participate from the
tracking-order past side, and (SAM3 default, `use_memory_selection=1`) memory
frames are filtered per frame by the official `frame_filter` (`eff_iou_score >
0.01` plus the must-include adjacent frame). Pre-alignment checkpoints are
repaired in place by `scripts/fix_gguf_merges.py` (merge table + these
hparams; idempotent).

### Offline Visualization (no GUI required)

`sam3_vis` runs segmentation and renders the official-style full-scene PNG:
per-instance colored mask overlay + box outlines + `id score` labels.

```bash
# Text-prompted detection (SAM 3)
./examples/sam3_vis --model models/sam3-f16.gguf --image photo.jpg \
    --prompt "cat" --out vis.png

# Point/box segmentation (any model)
./examples/sam3_vis --model models/sam2.1_hiera_tiny_f16.gguf \
    --image photo.jpg --point 320,240 --out vis.png

# Optional greedy IoM mask dedup (official agent default 0.3)
./examples/sam3_vis --model models/sam3-f16.gguf --image photo.jpg \
    --prompt "cat" --iom 0.3 --out vis.png

# Video: render every tracked frame to vis_frames/frame_XXXXX.png
./examples/sam3_vis --model models/sam3-f16.gguf --video input.mp4 \
    --prompt "cat"
```

The same rendering primitives are exposed in the C++ API
(`sam3_render_result`, `sam3_overlay_mask`, `sam3_draw_box`,
`sam3_save_image`, `sam3_remove_overlapping_masks` in `sam3.h`).

### Mask Prompt & Soft Logits (API)

```cpp
// Iterative refinement (SAM 1-task style): feed a previous prediction back.
sam3_pvs_params p1 = p0;
p1.return_logits = true;                      // get raw logits per detection
sam3_result r1 = sam3_segment_pvs(*state, *model, p1);

sam3_pvs_params p2 = p0;                       // same point prompt
p2.use_mask_prompt = true;                     // low-res mask prompt branch
p2.mask_prompt = r1.detections[0].mask;        // binary 0/255 convenience input
// or: sam3_resize_mask_logits(logits, w, h, low, 256, 256);
//     p2.mask_prompt_logits.assign(low, low + 256*256);  // official raw-logits path
sam3_result r2 = sam3_segment_pvs(*state, *model, p2);
```

`ggml_upgrade_ab.sh` now A/B-fingerprints the PCS (text detection) and
video-tracking paths in addition to PVS, so future ggml upgrades verify every
inference path bit-for-bit, not just point prompts.

### Quantization

Convert F32/F16 weights to smaller quantized formats (standard GGUF in/out):

```bash
./sam3_quantize models/sam3-f16.gguf models/sam3-q4_0.gguf q4_0
# Supported types: q4_0, q4_1, q8_0
```

## Converting Weights

Convert official PyTorch checkpoints straight to GGUF:

```bash
# SAM 3
uv run python convert_sam3_to_ggml.py \
    --model sam3.pt \
    --output models/sam3-f16.gguf \
    --ftype 1 \
    --tokenizer /path/to/tokenizer

# SAM 3 visual-only (no text encoder, smaller file)
uv run python convert_sam3_to_ggml.py \
    --model sam3.pt \
    --output models/sam3-visual-f16.gguf \
    --ftype 1 \
    --visual-only

# SAM 2 / SAM 2.1
uv run python convert_sam2_to_ggml.py \
    --model sam2.1_hiera_large.pt \
    --config sam2.1_hiera_l.yaml \
    --output models/sam2.1_hiera_large_f16.gguf \
    --ftype 1

```

`--ftype 0` = float32, `--ftype 1` = float16 (recommended). Then quantize with `sam3_quantize`.

To migrate an existing legacy `.ggml` file (no PyTorch checkpoint needed):

```bash
python3 scripts/convert_ggml_to_gguf.py models/old_model.ggml models/old_model.gguf
# or repack every model in a directory:
python3 scripts/convert_ggml_to_gguf.py --dir models/
```

## Acknowledgments

- [Meta AI Research](https://github.com/facebookresearch) for SAM, SAM 2, and SAM 3
- [ggml](https://github.com/ggerganov/ggml), the tensor computation library that makes this possible
- [sam.cpp](https://github.com/YavorGIvanov/sam.cpp), the original SAM 1 C++ port that inspired this project's architecture

## License

MIT
