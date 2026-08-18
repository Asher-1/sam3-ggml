# CLAUDE.md

## Project

sam3.cpp — a C++14 port of Meta's SAM 3 (Segment Anything Model 3) using ggml for inference on CPU, CUDA, Vulkan and Apple Metal.

## Architecture

- **One library**: `sam3.cpp` (implementation) + `sam3.h` (public API).
- **Structs and free functions only**. No classes, no inheritance, no virtual dispatch, no polymorphism.
- **C++14 idioms**: `std::unique_ptr`, `std::shared_ptr`, `std::make_unique`, move semantics, lambdas, `auto`. Use them.
- **Speed is a first-class citizen**. Avoid unnecessary copies, prefer in-place ggml ops (`_inplace` variants), minimize allocations in hot paths. Always use the fastest available ggml kernels: prefer `ggml_flash_attn_ext` over manual Q·K^T→softmax→V when the backend supports it, use fused ops where ggml provides them, and check `ggml/examples/` for the most up-to-date patterns. Profile before over-engineering.

## ggml graph isolation (CRITICAL)

**Each logical pipeline stage MUST run in its own ggml sub-graph** with its own `ggml_context`, `ggml_cgraph`, and `ggml_gallocr`. Data flows between stages as CPU-side `std::vector<float>` buffers.

**Why:** The ggml graph allocator (`ggml_gallocr`) reuses intermediate tensor buffers once their consumers have executed. In a single large graph spanning multiple transformer stages, the allocator overwrites buffers that downstream stages still need. This produces silently wrong numerical results — not crashes, just garbage outputs that are extremely hard to debug.

**Concrete rules:**

1. **One sub-graph per transformer block/stage.** The text encoder, geometry encoder, fusion encoder, DETR decoder, segmentation head, memory encoder, and memory attention each get their own `ggml_context` + `ggml_gallocr`. Build → allocate → set inputs → compute → read outputs → free.

2. **NEVER use state tensors as graph operands.** Tensors from `state.neck_trk[*]`, `state.neck_det[*]`, or any previous graph's output MUST NOT appear as arguments to `ggml_add`, `ggml_reshape`, `ggml_permute`, or any graph builder function. `ggml_build_forward_expand` traces the entire dependency tree — using a state tensor pulls in ALL its ancestors (the full ViT + neck recomputation: 2500+ nodes, ~40 seconds). Instead, create a fresh input tensor and copy data via CPU:
   ```cpp
   // WRONG — pulls in entire ViT recomputation:
   auto* x = ggml_reshape_3d(ctx, state.neck_trk[2], D, N, 1);

   // CORRECT — isolated input, no dependency chain:
   auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N, 1);
   ggml_set_name(x, "input"); ggml_set_input(x);
   // after ggml_gallocr_alloc_graph:
   std::vector<float> buf(D * N);
   ggml_backend_tensor_get(state.neck_trk[2], buf.data(), 0, buf.size() * sizeof(float));
   ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
   ```

3. **Model weight tensors are safe.** The model's weight tensors (in `model.xxx.weight`) live in a separate persistent buffer and are never managed by the graph allocator. They can be referenced directly in graph ops (e.g., `ggml_mul_mat(ctx, model.layer.weight, x)`).

**Functions that follow this pattern:** `sam3_segment_pcs` (5 sub-graphs), `sam3_segment_pvs`, `sam3_propagate_single`, `sam3_encode_memory`.

## Implementation plan

All work follows the phased plan in `PLAN.md`. Read it before starting any phase. Each phase has concrete steps, verification criteria, and the exact structs/functions to implement.

## Reference implementations

When lost on how to structure the ggml forward pass, how to build graphs, or how to load weights:

1. **sam.cpp** (https://github.com/YavorGIvanov/sam.cpp) — the original SAM 1 port to C++/ggml. Study `sam.cpp` and `sam.h` for patterns: graph construction, two-pass measure+compute, `ggml_backend_tensor_set`, window partition, attention with relative position, mask decoder upscaling. Our code follows the same conventions.

2. **ggml examples** (`ggml/examples/` in the submodule) — canonical, up-to-date examples of how to use ggml APIs. Check these for: backend init, graph allocation (`ggml_gallocr`), tensor creation, `ggml_backend_graph_compute`, Metal usage. The ggml API evolves; the submodule examples are always correct for our pinned version.

3. **SAM 3 official repo** (https://github.com/facebookresearch/sam3) — the ground truth for the forward pass. When in doubt about tensor shapes, operation order, activation functions, or any architectural detail, read the Python source. The paper is in `sam3.pdf`.

## Code style

- Prefix all internal (static) functions with `sam3_`.
- ggml graph-building functions take `ggml_context *` as first arg and return `ggml_tensor *`.
- Weight structs hold raw `ggml_tensor *` pointers (owned by the model's ggml context).
- Use `fprintf(stderr, ...)` for diagnostics, not `std::cerr`.
- No exceptions. Check return values. Functions that can fail return `bool` or `nullptr`.

## Dependencies

Only: ggml (submodule), stb_image/stb_image_write (vendored in `stb/`), C++14 standard library. Nothing else in the library. SDL2/ImGui are example-only.

## ggml integration (CRITICAL)

The `ggml/` submodule tracks the **official upstream repo** (`https://github.com/ggml-org/ggml`) pinned to tag **v0.18.1**. NEVER edit files inside `ggml/` by hand:

1. Every local change lives in the consolidated patch file in `ggml-patches/`.
2. `scripts/apply_ggml_patches.sh` applies them idempotently (safe to re-run).
3. CMake calls the script automatically at configure time — a fresh clone builds the same patched source without manual steps. The script fails the configure with diagnostics if a patch stops applying (e.g. after an upstream bump), instead of silently building stale code.

To update ggml: bump the submodule tag, then regenerate/adjust the consolidated patch against the new tree (`cd ggml && git apply --check ../ggml-patches/0001-sam3-ggml-combined.patch`), and update the pin in this file + README.

Current patch set:
`0001-sam3-ggml-combined.patch` — the single consolidated patch containing the Metal head-dimension extensions, CUDA flash-attention head-dimension support, CUDA/Vulkan conv2d-transpose fast paths, CUDA F16 matmul output, fused custom-frequency/F16-to-F32 RoPE, native window partition, and fused QKV layout plus Q/K RoPE changes.

## Model format (GGUF, CRITICAL)

All model files are **standard GGUF v3** — same container the rest of the ggml ecosystem (face-detect-ggml, free-splatter.cpp, OpenPCDet-GGML) loads via `gguf_init_from_file`. There is NO custom binary format anymore; legacy `.ggml` files are migrated with `scripts/convert_ggml_to_gguf.py` (byte-identical tensor data).

Load path in `sam3_load_model`:
1. `gguf_init_from_file(path, {no_alloc=true, &gguf_ctx})` — metadata + tensor table only, data streamed later so a 5 GB model never sits twice in RAM.
2. All KV reads go through a single `gguf_kv` reader (err-accumulating, `required` flag) — the same interface as free-splatter.cpp / OpenPCDet-GGML's `model_file`. The standard `general.architecture` key (`sam3`/`sam2`) dispatches to `sam3_load_hparams` / `sam2_load_hparams`; legacy files with only `sam3.arch` still load (fallback). Every hparam is a `sam3.hparams.<field>` KV (int32 or int32 array).
3. Tensors are registered as before (register_* macros), then `sam3_load_tensors_from_gguf` streams each blob from `gguf_get_data_offset() + gguf_get_tensor_offset()` and uploads via `ggml_backend_tensor_set`. Registered type wins: file f16→registered f32 etc. is converted on load (1x1 convs are registered F32 even in F16 files); element-count (not per-dim shape) is enforced because legacy files store e.g. `sam_pe.pe_gaussian` as [128,2] while registered as [2,128].
4. Tokenizer (SAM3 only) comes from `sam3.tokenizer.vocab` (string array indexed by token id) + `sam3.tokenizer.merges` ("a b" strings in rank order).

Writers: `convert_sam3_to_ggml.py` and `convert_sam2_to_ggml.py` emit GGUF directly (hparams→KV, tokenizer→string-array KV, plus the standard `general.architecture` key; `sam3.arch` is written too for backwards compatibility); `examples/quantize.cpp` is GGUF in/out via `gguf_init_empty` + `gguf_add_tensor` + `gguf_write_to_file` and copies all KV through with `gguf_set_kv`. The quantize decision MUST mirror the register_* macros: quantize when `ne[0] % blk == 0` and the name is not an embedding / not `.bias` / not `norm` (GGUF n_dims cannot distinguish a [256,1] Linear output from a [256] bias). Quantized output goes through a pre-reserved pool — `gguf_set_tensor_data` stores a pointer dereferenced only at write time, so it must never point into a reallocating buffer.

## Python

`uv` is the package manager. Use `uv run python` for all Python execution (scripts, tests, weight conversion). Never use bare `python` or `pip` — always `uv run python` and `uv pip install`.

## Build

```bash
cd build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

Backends are forwarded to ggml via CMake options:

```bash
cmake .. -DSAM3_CUDA=ON                 # NVIDIA CUDA backend
cmake .. -DSAM3_VULKAN=ON               # Vulkan backend (needs Vulkan SDK / glslc)
cmake .. -DSAM3_METAL=OFF               # disable Metal (on by default on macOS)
cmake .. -DSAM3_HIPBLAS=ON              # AMD HIP backend
```

The active backend is discovered at runtime through the ggml backend registry (`ggml_backend_dev_by_type(GPU)`), so no sam3 code differs per backend. `sam3_backend_name(model)` reports which one a model actually runs on.

Tests: `cmake .. -DSAM3_BUILD_TESTS=ON`

## Benchmarking

`sam3_benchmark` tracks an object across video frames and reports latency for every model × backend combination (CPU + whichever GPU backend the build includes). Each run is forked into a subprocess so a crash does not kill the suite. The `Backend` column shows the real backend name (CUDA/Vulkan/Metal/CPU), not a guess.

```bash
# Full benchmark (all 49 models × Metal + CPU):
./build/examples/sam3_benchmark

# Quick iteration (e.g. testing an optimization) — 4 runs, ~30 s:
./build/examples/sam3_benchmark --filter tiny --n-frames 3 --filter-prec f16,q4_0

# GPU only (CUDA/Vulkan/Metal, whichever the build has):
./build/examples/sam3_benchmark --gpu-only

# CPU only:
./build/examples/sam3_benchmark --cpu-only
```

**Quick-iteration recipe:** when profiling or testing optimizations, `--filter tiny --n-frames 3` limits to the SAM2/2.1 tiny models on both Metal and CPU in f16 and q4_0 — just 4 runs total, enough to see whether a change helps without waiting for the full suite.

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir <path>` | `models/` | Directory containing `.gguf`/`.ggml` files |
| `--video <path>` | `data/test_video.mp4` | Video file |
| `--point-x <f>` | `315.0` | X coordinate of the tracking point |
| `--point-y <f>` | `250.0` | Y coordinate of the tracking point |
| `--n-frames <n>` | `10` | Number of frames to track |
| `--n-threads <n>` | `4` | CPU thread count |
| `--cpu-only` | | Skip Metal runs |
| `--gpu-only` | | Skip CPU runs |
| `--filter <substr>` | | Only run models whose filename contains `<substr>` |

Output columns: model name, file size, backend, load time, init time (frame 0 encode + add instance), average per-frame tracking time, total pipeline time, detection count, status. Diagnostics go to stderr; the final table goes to stdout (pipe-friendly: `./build/examples/sam3_benchmark 2>/dev/null > results.txt`).

## Weights

PyTorch checkpoint → `convert_sam3_to_ggml.py` → `.gguf` (GGUF v3, standard). The conversion stores every tensor. The C++ loader registers them and streams the blobs into the backend buffer. Quantize with `sam3_quantize <in.gguf> <out.gguf> <q4_0|q4_1|q8_0>`.
