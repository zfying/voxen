#!/usr/bin/env bash
# exp6: 3-stage shared VPT + decoder.
#   Stage 1: readout only.  Auto-loads the readout from a prior exp2 transformer
#            checkpoint when one exists; otherwise trains stage 1 in-run.
#   Stage 2: prompts only (readout frozen).
#   Stage 3: joint (readout + prompts both trainable).
#
# Sweeps K = 1, 5, 10, 20, 40 by default; override via KS env var.
# Per-stage epoch budget via S1/S2/S3 env vars (defaults 5/5/5).
# Per-stage LR via LR1/LR2/LR3 env vars (default to $LR from _common.sh).
#
# Usage:
#   bash scripts/vpt/exp6_staged_vpt.sh [GPU_ID=0]
#   KS="20 40" S1=3 S2=8 S3=4 bash scripts/vpt/exp6_staged_vpt.sh 0

source "$(dirname "$0")/_common.sh"
GPU="${1:-0}"
KS="${KS:-1 5 10 20 40}"
S1="${S1:-5}"; S2="${S2:-5}"; S3="${S3:-5}"
LR1="${LR1:-$LR}"; LR2="${LR2:-$LR}"; LR3="${LR3:-$LR}"

for K in $KS; do
    CUDA_VISIBLE_DEVICES="$GPU" python3 main.py \
        "${COMMON[@]}" \
        --encoder_arch vpt \
        --vpt_prompt_share shared \
        --vpt_readout decoder \
        --vpt_num_prompts_per_roi "$K" \
        --vpt_staged \
        --vpt_stage1_epochs "$S1" \
        --vpt_stage2_epochs "$S2" \
        --vpt_stage3_epochs "$S3" \
        --vpt_stage1_lr "$LR1" \
        --vpt_stage2_lr "$LR2" \
        --vpt_stage3_lr "$LR3" \
        --vpt_load_readout auto \
        --batch_size "$BATCH_FAST" \
        --save_model 1
done
