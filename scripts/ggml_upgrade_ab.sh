#!/usr/bin/env bash
#
# ggml_upgrade_ab.sh
#
# Steady-state performance A/B for a ggml submodule upgrade, paired with
# tests/upgrade_probe.cpp (numerical fingerprint + per-stage timing).
#
# Compile the probe once per build tree (see tests/upgrade_probe.cpp header),
# then run this script with the two build dirs to compare:
#   - mask FNV hash  (numerical parity, must be identical)
#   - encode/pvs median latency over N in-process repetitions
#     (eliminates the fork/cold-start noise that makes single-shot
#      benchmark runs incomparable across builds)
#
# Usage:
#   bash scripts/ggml_upgrade_ab.sh <baseline-build-dir> <candidate-build-dir> [repeat]
#
set -euo pipefail

BASE_BUILD=${1:?usage: ggml_upgrade_ab.sh <baseline-build> <candidate-build> [repeat]}
CAND_BUILD=${2:?usage: ggml_upgrade_ab.sh <baseline-build> <candidate-build> [repeat]}
REPEAT=${3:-5}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="$ROOT/data/test_image.jpg"
PX=315; PY=250

MODELS=(
    sam3-f16.gguf          # full SAM3, f16
    sam3-q8_0.gguf         # quantized full model
    sam3-visual-f16.gguf   # visual-only
    sam2.1_hiera_tiny_f16.gguf
    sam2.1_hiera_tiny_q4_0.gguf
)

for model in "${MODELS[@]}"; do
    for dev in cuda cpu; do
        echo "── $model @ $dev (n=$REPEAT)"
        echo -n "  base: "
        "$BASE_BUILD/upgrade_probe"  "$ROOT/models/$model" "$IMG" "$dev" $PX $PY "$REPEAT" 2>/dev/null \
            | grep -E "timing|det\[" | tr '\n' ' '; echo
        echo -n "  cand: "
        "$CAND_BUILD/upgrade_probe"  "$ROOT/models/$model" "$IMG" "$dev" $PX $PY "$REPEAT" 2>/dev/null \
            | grep -E "timing|det\[" | tr '\n' ' '; echo
    done
done
