#!/usr/bin/env bash

# Wrapper to launch GPT-OSS-20B LoRA training with BlockFFN objectives.
# Usage:
#   scripts/run_gpt_oss_blockffn.sh
#   scripts/run_gpt_oss_blockffn.sh --gpus 0,1 --gpu-mem 30gb
#
# Options:
#   --gpus LIST       Comma-separated GPU IDs passed via CUDA_VISIBLE_DEVICES.
#   --gpu-mem SIZE    Optional per-GPU memory cap (requires nvidia-smi). Example: 30gb.

set -euo pipefail

GPUS=""
GPU_MEM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpu-mem)
      GPU_MEM="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ -n "$GPU_MEM" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    IFS=',' read -ra GPU_ARRAY <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}"
    for gpu in "${GPU_ARRAY[@]}"; do
      nvidia-smi -i "$gpu" -pl "$GPU_MEM" || {
        echo "Warning: failed to set power limit for GPU $gpu" >&2
      }
    done
  else
    echo "Warning: nvidia-smi not found; skipping GPU memory capping." >&2
  fi
fi

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# Set HuggingFace token if available (to avoid rate limits)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "Using HuggingFace token for authentication"
fi

# Pre-download dataset to avoid rate limits during multi-GPU training
echo "Pre-downloading dataset to avoid rate limits..."
python3 -c "
from datasets import load_dataset
import sys
try:
    print('Downloading dolmino_wiki subset (this may take a while)...')
    # Use streaming=False to actually download and cache the data
    # Take a small sample to trigger the download without loading everything
    ds = load_dataset('allenai/dolmino-mix-1124', 'wiki', split='train', streaming=False)
    print(f'Dataset downloaded successfully! Total samples: {len(ds)}')
except Exception as e:
    print(f'Warning: Could not pre-download dataset: {e}', file=sys.stderr)
    print('Training will download during execution...', file=sys.stderr)
" || echo "Pre-download failed, continuing anyway..."

exec llamafactory-cli train examples/train_lora/gpt_oss_blockffn.yaml

