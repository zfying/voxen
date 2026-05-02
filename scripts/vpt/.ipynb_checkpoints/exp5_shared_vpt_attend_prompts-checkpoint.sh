#!/usr/bin/env bash
# exp5: shared VPT + DETR decoder, but the decoder cross-attends over BOTH
# patches AND prompt tokens (--vpt_decoder_attend_prompts). Otherwise
# identical to exp3. Sweeps K = 1, 5, 10, 20, 40.
#
# Usage:
#   bash scripts/vpt/exp5_shared_vpt_attend_prompts.sh [GPU_ID=0]
#   KS="20 40" bash scripts/vpt/exp5_shared_vpt_attend_prompts.sh 0

source "$(dirname "$0")/_common.sh"
GPU="${1:-0}"
KS="${KS:-1 5 10 20 40}"

for K in $KS; do
    CUDA_VISIBLE_DEVICES="$GPU" python3 main.py \
        "${COMMON[@]}" \
        --encoder_arch vpt \
        --vpt_prompt_share shared \
        --vpt_readout decoder \
        --vpt_decoder_attend_prompts \
        --vpt_num_prompts_per_roi "$K" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_FAST"
done
