# tests/ — Layout Guide

Every C/C++ test links against `libsam3` and is registered in `CMakeLists.txt`
(`-DSAM3_BUILD_TESTS=ON` to build). Python tooling is not built; it requires
the PyTorch environment from `pyproject.toml` and is only needed when
regenerating reference data or debugging numerical alignment.

```
tests/
├── CMakeLists.txt        # all C++ test targets
├── test_utils.h          # shared test helpers
├── reports/              # regression baselines (the source of truth for
│                         # "did a model/code change drift quality or speed")
│   ├── regression_report_perf.md     # perf + VRAM baselines, RTX 3060
│   └── regression_report_q8_0.md     # quantization quality baselines + cull data
├── data/                 # test assets (images, prompt/case tables)
├── python/               # C++ ↔ PyTorch numerical-alignment tooling
│                         # (dump_* regenerate reference tensors, compare_*/
│                         #  check_*/analyze_*/trace_* diff against them)
└── backend/              # backend-specific tests & microbenchmarks
    ├── test_metal_*.cpp  # Metal/CUDA/Vulkan stage-comparison harnesses
    └── bench_im2col_tiled_ab.cu  # standalone A/B: tiled vs scalar im2col
                                  # (nvcc -O3 -arch=sm_86; see perf report §17)
```

Root-level `test_*.cpp` are the library tests (unit/phase checks and
end-to-end PVS/PCS pipelines). Two scripts are not registered in CMake and
are built ad hoc: `diff_tokenizer.cpp` (tokenizer differential vs the
official BPE, `--text` for clean-layer expectations) and
`ab_forward_probe.cpp` / `upgrade_probe.cpp` (ggml upgrade probes, used by
`scripts/ggml_upgrade_ab.sh`).

## Review notes (2026-09-28)

- No file was redundant: every Python script has a distinct role (verified by
  content diff, zero byte-level duplicates), and every `test_*.cpp` except the
  three named above is a CMake target.
- Grouping rationale: reports/data/python are non-build artifacts that were
  interleaved with buildable C++ tests; `backend/` collects the 16
  Metal-prefixed harnesses that share the stage-comparison pattern.
- Update paths in CMakeLists.txt (`backend/…`) and cross-document links when
  adding files; the regression reports reference each other by section
  numbers only, so they survive moves.
