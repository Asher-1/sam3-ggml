#!/bin/bash
#
# build_multi_backend.sh
#
# One-shot build of sam3.cpp with CPU + CUDA + Vulkan backends compiled into a
# single binary (any combination). The resulting sam3_image / sam3_video GUI
# examples expose a "Devices" combo (Auto / CPU / CUDA / Vulkan) — pick a
# backend at runtime, or leave it on Auto which probes CUDA -> Vulkan -> CPU
# and falls back to CPU.
#
# Usage:
#   bash scripts/build_multi_backend.sh                 # CPU + CUDA + Vulkan
#   bash scripts/build_multi_backend.sh --jobs 8        # limit parallel jobs
#   bash scripts/build_multi_backend.sh --cpu-only      # CPU backend only
#   bash scripts/build_multi_backend.sh --cuda-only     # CPU + CUDA
#   bash scripts/build_multi_backend.sh --vulkan-only   # CPU + Vulkan
#   bash scripts/build_multi_backend.sh --no-gui        # skip SDL2 GUI examples
#   bash scripts/build_multi_backend.sh --build-dir build-all
#
# Prerequisites:
#   - CUDA toolkit (nvcc) for the CUDA backend
#   - Vulkan SDK (glslc / glslangValidator) for the Vulkan backend
#   - SDL2 development files for the GUI examples. If SDL2 is not found by
#     CMake, the script builds SDL2 from source (git clone, no root needed)
#     into <project>/../tools/sdl2-install and points SDL2_DIR at it.
#
# Output: <project>/<build-dir>/examples/sam3_image, sam3_video, sam3_benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="build-all"
JOBS="$(nproc 2>/dev/null || echo 4)"
WANT_CUDA=1
WANT_VULKAN=1
WANT_GUI=1
CUDA_ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)        JOBS="$2"; shift 2 ;;
        --build-dir)   BUILD_DIR="$2"; shift 2 ;;
        --cuda-arch)   CUDA_ARCH="$2"; shift 2 ;;
        --cpu-only)    WANT_CUDA=0; WANT_VULKAN=0; shift ;;
        --cuda-only)   WANT_CUDA=1; WANT_VULKAN=0; shift ;;
        --vulkan-only) WANT_CUDA=0; WANT_VULKAN=1; shift ;;
        --no-gui)      WANT_GUI=0; shift ;;
        -h|--help)
            sed -n '2,26p' "$0"
            exit 0 ;;
        *) echo "error: unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "== sam3.cpp multi-backend build =="
echo "  build dir : ${BUILD_DIR}"
echo "  jobs      : ${JOBS}"
echo "  cuda      : $([ ${WANT_CUDA} -eq 1 ] && echo ON || echo OFF)"
echo "  vulkan    : $([ ${WANT_VULKAN} -eq 1 ] && echo ON || echo OFF)"
echo "  gui       : $([ ${WANT_GUI} -eq 1 ] && echo ON || echo OFF)"

# ── Toolchain checks ─────────────────────────────────────────────────────────
if [[ ${WANT_CUDA} -eq 1 ]] && ! command -v nvcc >/dev/null 2>&1; then
    echo "error: --cuda requested but nvcc not found (install CUDA toolkit)" >&2
    exit 1
fi
if [[ ${WANT_VULKAN} -eq 1 ]] && ! command -v glslc >/dev/null 2>&1 \
    && ! command -v glslangValidator >/dev/null 2>&1; then
    echo "error: --vulkan requested but neither glslc nor glslangValidator found" >&2
    exit 1
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
[[ ${WANT_CUDA} -eq 1 ]]   && CMAKE_ARGS+=(-DSAM3_CUDA=ON)
[[ ${WANT_VULKAN} -eq 1 ]] && CMAKE_ARGS+=(-DSAM3_VULKAN=ON)

# CUDA arch: prefer an explicit --cuda-arch, then detect from the GPU via
# nvidia-smi (compute cap "8.6" -> "86"), otherwise fall back to sm_86.
# NOTE: keep sm_90 out of the fallback — ggml's PDL code path needs CUDA 12+
# to compile for Hopper, which breaks older toolchains (e.g. CUDA 11.8).
if [[ ${WANT_CUDA} -eq 1 && -z "${CUDA_ARCH}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    [[ -n "${CAP}" ]] && CUDA_ARCH="${CAP//./}"
fi
if [[ ${WANT_CUDA} -eq 1 ]]; then
    if [[ -n "${CUDA_ARCH}" ]]; then
        echo "  cuda arch : ${CUDA_ARCH} (detected from GPU)"
    else
        CUDA_ARCH="86"
        echo "  cuda arch : ${CUDA_ARCH} (fallback; set --cuda-arch to override)"
    fi
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
fi

# ── SDL2: honor SDL2_DIR, else known local builds, else system, else source ─
if [[ ${WANT_GUI} -eq 1 ]]; then
    SDL2_INSTALL="${PROJECT_ROOT}/../tools/sdl2-install"
    SDL2_CMAKE=""
    if [[ -n "${SDL2_DIR:-}" ]] && [[ -f "${SDL2_DIR}/SDL2Config.cmake" ]]; then
        SDL2_CMAKE="${SDL2_DIR}"
        echo "Using SDL2 from \$SDL2_DIR: ${SDL2_DIR}"
    else
        # Reuse an existing local build if present (skip the source build).
        for cand in "${PROJECT_ROOT}/../tools/sdl2-install" \
                    "${PROJECT_ROOT}/tools/sdl2-install" \
                    "${HOME}/develop/tools/sdl2-install"; do
            if [[ -d "${cand}/lib/cmake/SDL2" ]]; then
                SDL2_INSTALL="${cand}"
                SDL2_CMAKE="${cand}/lib/cmake/SDL2"
                echo "Using local SDL2 at ${SDL2_INSTALL}"
                break
            fi
        done
    fi
    if [[ -z "${SDL2_CMAKE}" ]] && pkg-config --exists sdl2 2>/dev/null; then
        echo "Using system SDL2"
    elif [[ -z "${SDL2_CMAKE}" ]]; then
        # No usable SDL2 dev files anywhere — build SDL2 from source (no root needed).
        SDL2_TAG="release-2.30.9"
        SDL2_SRC="$(mktemp -d)/SDL"
        echo "SDL2 not found; building SDL2 ${SDL2_TAG} from source..."
        git clone --depth 1 --branch "${SDL2_TAG}" \
            https://github.com/libsdl-org/SDL.git "${SDL2_SRC}"
        cmake -S "${SDL2_SRC}" -B "${SDL2_SRC}/build" \
            -DCMAKE_INSTALL_PREFIX="${SDL2_INSTALL}" -DCMAKE_BUILD_TYPE=Release \
            >/dev/null
        cmake --build "${SDL2_SRC}/build" -j"${JOBS}" >/dev/null
        cmake --install "${SDL2_SRC}/build" >/dev/null
        echo "SDL2 installed to ${SDL2_INSTALL}"
        SDL2_CMAKE="${SDL2_INSTALL}/lib/cmake/SDL2"
    fi

    if [[ -n "${SDL2_CMAKE}" ]]; then
        CMAKE_ARGS+=(-DSDL2_DIR="${SDL2_CMAKE}")
    fi
fi

# ── Configure + build ────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
cmake -S . -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo
echo "== Build finished =="
if [[ -f "${BUILD_DIR}/examples/sam3_image" ]]; then
    echo "  GUI:    ${BUILD_DIR}/examples/sam3_image"
    echo "          ${BUILD_DIR}/examples/sam3_video"
    echo "  run:    ${BUILD_DIR}/examples/sam3_image --model models/sam2.1_hiera_tiny_f16.gguf --image data/test_image.jpg"
    echo "  device: add --device auto|cpu|cuda|vulkan (or use the Devices combo in the UI)"
else
    echo "  tools:  ${BUILD_DIR}/examples/sam3_benchmark  ${BUILD_DIR}/examples/sam3_quantize"
fi
