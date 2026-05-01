#!/usr/bin/env bash
# Run the 4 experiment groups on subj 1 across 2x A100-80GB.
#
#   exp1  baseline_linear      : --encoder_arch linear (no soft prompts, ridge linear readout)
#   exp2  baseline_transformer : --encoder_arch transformer (replicates README example)
#   exp3  shared VPT + decoder : K shared soft tokens, single backbone forward + cross-attn
#                                 decoder (sweep K = 1, 5, 10)
#   exp4  per-ROI VPT + linear : 1 prompt per ROI, shared linear readout (~50x compute)
#
# All experiments share: subj=1, enc_output_layer=1, readout_res=rois_all,
# backbone=dinov2_q (the README default), run=1, lr=5e-4.
#
# GPU layout:
#   GPU 0 -> exp1, exp2, exp3 (K=1,5,10)  serial; each is cheap (~1x baseline cost)
#   GPU 1 -> exp4                          (~50x baseline cost, runs alone)
#
# Usage:
#   bash scripts/run_vpt_experiments.sh                 # run everything
#   bash scripts/run_vpt_experiments.sh exp3            # run a single group
#   WANDB_PROJECT=brain_vpt bash scripts/run_vpt_experiments.sh   # enable W&B

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- shared config ----
SUBJ=1
RUN=1
ENC_LAYER=1
READOUT=rois_all
BACKBONE=dinov2_q
LR=5e-4
EPOCHS_FAST=15      # baselines + shared VPT
EPOCHS_VPT_HEAVY=5  # per-ROI VPT (cost ~ N_ROI x baseline; trade epochs for tractability)
BATCH_FAST=16
BATCH_VPT=4         # per-ROI VPT effective backbone batch = BATCH_VPT * N_ROI
ROI_CHUNK=25        # chunk ROIs inside one forward to bound peak memory

WANDB_FLAGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_FLAGS=(--wandb_p "$WANDB_PROJECT")
fi

COMMON=(
    --run "$RUN"
    --subj "$SUBJ"
    --enc_output_layer "$ENC_LAYER"
    --readout_res "$READOUT"
    --backbone_arch "$BACKBONE"
    --lr "$LR"
    "${WANDB_FLAGS[@]}"
)

run_exp1_linear() {
    CUDA_VISIBLE_DEVICES="${1:-0}" python main.py \
        "${COMMON[@]}" \
        --encoder_arch linear \
        --epochs "$EPOCHS_FAST" \
        --batch_size "$BATCH_FAST"
}

run_exp2_transformer() {
    CUDA_VISIBLE_DEVICES="${1:-0}" python main.py \
        "${COMMON[@]}" \
        --encoder_arch transformer \
        --epochs "$EPOCHS_FAST" \
        --batch_size "$BATCH_FAST"
}

run_exp3_shared_vpt() {
    local gpu="${1:-0}"
    for K in 1 5 10; do
        CUDA_VISIBLE_DEVICES="$gpu" python main.py \
            "${COMMON[@]}" \
            --encoder_arch vpt \
            --vpt_prompt_share shared \
            --vpt_readout decoder \
            --vpt_num_prompts_per_roi "$K" \
            --epochs "$EPOCHS_FAST" \
            --batch_size "$BATCH_FAST"
    done
}

run_exp4_per_roi_vpt() {
    CUDA_VISIBLE_DEVICES="${1:-1}" python main.py \
        "${COMMON[@]}" \
        --encoder_arch vpt \
        --vpt_prompt_share per_roi \
        --vpt_readout linear \
        --vpt_linear_share shared \
        --vpt_linear_feature prompt \
        --vpt_num_prompts_per_roi 1 \
        --vpt_roi_chunk "$ROI_CHUNK" \
        --epochs "$EPOCHS_VPT_HEAVY" \
        --batch_size "$BATCH_VPT"
}

case "${1:-all}" in
    exp1) run_exp1_linear 0 ;;
    exp2) run_exp2_transformer 0 ;;
    exp3) run_exp3_shared_vpt 0 ;;
    exp4) run_exp4_per_roi_vpt 1 ;;
    all)
        # Long job on GPU 1 in the background; cheap pipeline on GPU 0 in foreground.
        mkdir -p logs
        run_exp4_per_roi_vpt 1 > logs/exp4_per_roi_vpt.log 2>&1 &
        EXP4_PID=$!
        echo "exp4 (per-ROI VPT) launched on GPU 1, pid=$EXP4_PID, log=logs/exp4_per_roi_vpt.log"

        run_exp1_linear      0 2>&1 | tee logs/exp1_linear.log
        run_exp2_transformer 0 2>&1 | tee logs/exp2_transformer.log
        run_exp3_shared_vpt  0 2>&1 | tee logs/exp3_shared_vpt.log

        wait "$EXP4_PID"
        echo "all experiments finished."
        ;;
    *) echo "usage: $0 [all|exp1|exp2|exp3|exp4]" ; exit 2 ;;
esac
