#!/bin/bash
#
# run_video.sh
#
# One-click launcher for the sam3_video interactive video tracking GUI
# example (built from examples/main_video.cpp).
#
# Usage:
#   ./run_video.sh                              # model only (pick a video via --video)
#   ./run_video.sh data/test_video.mp4          # positional arg = video path
#   ./run_video.sh --model models/sam3-visual-f16.gguf clip.mp4
#   ./run_video.sh --device cuda --threads 8 data/test_video.mp4
#
# Environment overrides:
#   MODEL=models/sam3-f16.gguf    default model when --model is not given
#   VIDEO=data/test_video.mp4     default video when no positional arg / --video
#   DEVICE=auto|cpu|cuda|vulkan   appended as --device unless already given
#   THREADS=4                     appended as --threads unless already given
#   BUILD_DIR=build-all           which build dir to use (auto-detected otherwise)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Locate the binary ────────────────────────────────────────────────────────
BIN_NAME="sam3_video"
if [[ -n "${BUILD_DIR:-}" ]]; then
    SEARCH_DIRS=("$BUILD_DIR")
else
    SEARCH_DIRS=(build-all build-cuda build-vulkan build-cpu2 build-cpu build)
fi

BIN=""
for d in "${SEARCH_DIRS[@]}"; do
    if [[ -x "$ROOT/$d/examples/$BIN_NAME" ]]; then
        BIN="$ROOT/$d/examples/$BIN_NAME"
        break
    fi
done

if [[ -z "$BIN" ]]; then
    echo "Error: $BIN_NAME not found in: ${SEARCH_DIRS[*]}" >&2
    echo "Build it first, e.g.:" >&2
    echo "  bash scripts/build_multi_backend.sh          # CPU + CUDA + Vulkan" >&2
    echo "  bash scripts/build_multi_backend.sh --cpu-only" >&2
    exit 1
fi

# ── Assemble arguments ───────────────────────────────────────────────────────
# Precedence: explicit flag > positional arg > env var > default.
# User args are passed through untouched; defaults are only prepended.
MODEL="${MODEL:-models/sam3-f16.gguf}"

VALUED_FLAGS=" --model --image --video --device --threads --encode-img-size "

has_flag() {  # has_flag <flag> [args...]
    local f="$1" skip=0 a
    shift
    for a in "$@"; do
        (( skip )) && { skip=0; continue; }
        [[ "$a" == "$f" ]] && return 0
        [[ "$a" == -* && "$VALUED_FLAGS" == *" $a "* ]] && skip=1
    done
    return 1
}

positional_arg() {  # first non-flag arg that is not a flag's value
    local skip=0 a
    for a in "$@"; do
        (( skip )) && { skip=0; continue; }
        if [[ "$a" == -* ]]; then
            [[ "$VALUED_FLAGS" == *" $a "* ]] && skip=1
            continue
        fi
        printf '%s' "$a"
        return 0
    done
    return 1
}

ARGS=()
if ! has_flag --model "$@"; then
    ARGS+=(--model "$MODEL")
fi
POS="$(positional_arg "$@" || true)"
if ! has_flag --video "$@"; then
    if [[ -n "$POS" ]]; then
        ARGS+=(--video "$POS")
    elif [[ -n "${VIDEO:-}" ]]; then
        ARGS+=(--video "$VIDEO")
    fi
fi
if [[ -n "${DEVICE:-}" ]] && ! has_flag --device "$@"; then
    ARGS+=(--device "$DEVICE")
fi
if [[ -n "${THREADS:-}" ]] && ! has_flag --threads "$@"; then
    ARGS+=(--threads "$THREADS")
fi

# Pass-through: user args minus the positional arg already promoted to --video.
PASS=()
skip=0
consumed_pos=0
for a in "$@"; do
    (( skip )) && { skip=0; PASS+=("$a"); continue; }  # flag's value: keep
    if [[ "$a" == -* ]]; then
        [[ "$VALUED_FLAGS" == *" $a "* ]] && skip=1
        PASS+=("$a")
        continue
    fi
    (( consumed_pos )) && PASS+=("$a") || consumed_pos=1
done

echo ">> $BIN ${ARGS[*]} ${PASS[*]:-}"
exec "$BIN" "${ARGS[@]}" "${PASS[@]}"
