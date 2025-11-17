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
  end
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

exec llamafactory-cli train examples/train_lora/gpt_oss_blockffn.yaml

