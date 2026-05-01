#!/usr/bin/env bash
# exp1: baseline linear readout (no soft prompts, ridge linear). Cheap, ~1x baseline.
# Usage: bash scripts/vpt/exp1_linear.sh [GPU_ID=0]

source "$(dirname "$0")/_common.sh"
GPU="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    "${COMMON[@]}" \
    --encoder_arch linear \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_FAST"
