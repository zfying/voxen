#!/usr/bin/env bash
# exp3: shared VPT + DETR decoder. K shared soft tokens prepended to a frozen
# DINOv2 backbone; decoder cross-attends ONLY over the patch grid (prompts
# influence patches indirectly via backbone self-attention).
# Sweeps K = 1, 5, 10, 20, 40 by default; override via KS env var.
#
# Usage:
#   bash scripts/vpt/exp3_shared_vpt.sh [GPU_ID=0]
#   KS="20 40" bash scripts/vpt/exp3_shared_vpt.sh 0

source "$(dirname "$0")/_common.sh"
GPU="${1:-0}"
KS="${KS:-1 5 10 20 40}"

for K in $KS; do
    CUDA_VISIBLE_DEVICES="$GPU" python main.py \
        "${COMMON[@]}" \
        --encoder_arch vpt \
        --vpt_prompt_share shared \
        --vpt_readout decoder \
        --vpt_num_prompts_per_roi "$K" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_FAST"
done
