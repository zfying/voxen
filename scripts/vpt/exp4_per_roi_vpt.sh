#!/usr/bin/env bash
# exp4: per-ROI VPT + shared linear readout. One prompt per ROI, ROI folded
# into the backbone batch (~N_ROI x compute). Slow — runs alone on its GPU.
# Usage: bash scripts/vpt/exp4_per_roi_vpt.sh [GPU_ID=1]

source "$(dirname "$0")/_common.sh"
GPU="${1:-1}"

CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    "${COMMON[@]}" \
    --encoder_arch vpt \
    --vpt_prompt_share per_roi \
    --vpt_readout linear \
    --vpt_linear_share shared \
    --vpt_linear_feature prompt \
    --vpt_num_prompts_per_roi 1 \
    --vpt_roi_chunk "$ROI_CHUNK" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_VPT"
