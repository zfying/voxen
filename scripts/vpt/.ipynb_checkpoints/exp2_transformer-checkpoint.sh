#!/usr/bin/env bash
# exp2: baseline DETR-style transformer decoder (replicates README example).
# Usage: bash scripts/vpt/exp2_transformer.sh [GPU_ID=0]

source "$(dirname "$0")/_common.sh"
GPU="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python3 main.py \
    "${COMMON[@]}" \
    --encoder_arch transformer \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_FAST" \
    --save_model 1
