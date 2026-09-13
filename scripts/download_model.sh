#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/models"

mkdir -p "${MODEL_DIR}"

echo "=== Downloading SAM3 checkpoint from HuggingFace ==="
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

python3 -c "
from huggingface_hub import hf_hub_download
import os
dst = '${MODEL_DIR}'
print('Downloading sam3.pt ...')
hf_hub_download('facebook/sam3', 'sam3.pt', local_dir=dst)
print(f'Saved to {dst}/sam3.pt')
"

echo ""
echo "=== Downloading the official BPE vocab (tokenizer source) ==="
# The converter embeds the tokenizer into the GGUF from this single
# authoritative asset (the official SimpleTokenizer input). The
# facebook/sam3 HF repo is gated; this GitHub asset is public. The converter
# takes the official 48894-row slice verbatim, so all merges — including the
# 6 that start with '#' — land in the file.
BPE_GZ="${MODEL_DIR}/bpe_simple_vocab_16e6.txt.gz"
if [ ! -f "${BPE_GZ}" ]; then
    curl -L --fail -o "${BPE_GZ}" \
        "https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
fi
ls -lh "${BPE_GZ}"

echo ""
echo "=== Converting to GGUF ==="
python3 "${PROJECT_DIR}/convert_sam3_to_ggml.py" \
    --model "${MODEL_DIR}/sam3.pt" \
    --output "${MODEL_DIR}/sam3-f16.gguf" \
    --ftype 1 \
    --bpe-gz "${BPE_GZ}"

echo ""
echo "Done. Model saved to ${MODEL_DIR}/sam3-f16.gguf"
ls -lh "${MODEL_DIR}/sam3-f16.gguf"
