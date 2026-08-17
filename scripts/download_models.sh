#!/bin/bash
#
# download_models.sh
#
# Download every model of the sam3.cpp Hugging Face repo into the local
# models/ directory, then repack each legacy .ggml file into the standard
# .gguf format (scripts/convert_ggml_to_gguf.py) and delete the .ggml source.
# The file list is fetched live from the HF API so the script never goes
# stale when new models are published.
#
# Usage:
#   bash scripts/download_models.sh                 # all models
#   bash scripts/download_models.sh --filter tiny   # only names containing "tiny"
#   bash scripts/download_models.sh --list          # just print the remote file list
#   bash scripts/download_models.sh --keep-ggml     # keep the raw .ggml files
#
# Prerequisites: curl, python3 (for JSON parsing)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"

REPO="PABannier/sam3.cpp"
API_URL="https://huggingface.co/api/models/${REPO}"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

FILTER=""
LIST_ONLY=0
KEEP_GGML=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter) FILTER="$2"; shift 2 ;;
        --list)   LIST_ONLY=1; shift ;;
        --keep-ggml) KEEP_GGML=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl not found" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found" >&2
    exit 1
fi

echo "Fetching model list from ${API_URL} ..."
FILES="$(curl -s "${API_URL}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for s in d.get("siblings", []):
    n = s["rfilename"]
    if n.endswith(".ggml"):
        print(n)
')"

if [[ -z "${FILES}" ]]; then
    echo "error: no model files found (network or API issue?)" >&2
    exit 1
fi

if [[ "${LIST_ONLY}" -eq 1 ]]; then
    echo "${FILES}"
    exit 0
fi

mkdir -p "${MODEL_DIR}"

downloaded=0
skipped=0
for name in ${FILES}; do
    if [[ -n "${FILTER}" ]] && [[ "${name}" != *"${FILTER}"* ]]; then
        continue
    fi
    dst="${MODEL_DIR}/${name}"
    if [[ -f "${dst}" ]] && [[ -s "${dst}" ]]; then
        echo "skip  ${name} (already present: $(du -h "${dst}" | cut -f1))"
        skipped=$((skipped + 1))
        continue
    fi
    echo "get   ${name} ..."
    curl -L --fail --progress-bar -o "${dst}" "${BASE_URL}/${name}"
    downloaded=$((downloaded + 1))

    # Repack the legacy .ggml into standard .gguf (byte-identical tensor data)
    if [[ "${name}" == *.ggml ]] && [[ "${KEEP_GGML}" -eq 0 ]]; then
        gguf_path="${dst%.ggml}.gguf"
        python3 "${SCRIPT_DIR}/convert_ggml_to_gguf.py" "${dst}" "${gguf_path}"
        rm -f "${dst}"
    fi
done

echo ""
echo "done: downloaded ${downloaded}, skipped ${skipped} -> ${MODEL_DIR}"
ls -lh "${MODEL_DIR}" | grep -v "^total" | grep -v "^$" || true
