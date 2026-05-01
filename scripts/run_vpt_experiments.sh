#!/usr/bin/env bash
# Thin orchestrator over scripts/vpt/exp*.sh — see those files for details.
#
#   exp1  baseline_linear           -> scripts/vpt/exp1_linear.sh
#   exp2  baseline_transformer      -> scripts/vpt/exp2_transformer.sh
#   exp3  shared VPT + decoder      -> scripts/vpt/exp3_shared_vpt.sh                (K=1,5,10,20,40)
#   exp4  per-ROI VPT + linear      -> scripts/vpt/exp4_per_roi_vpt.sh               (~50x compute)
#   exp5  shared VPT + decoder,     -> scripts/vpt/exp5_shared_vpt_attend_prompts.sh (K=1,5,10,20,40)
#         decoder attends to prompts
#
# GPU layout for `all`:
#   GPU 0 -> exp1, exp2, exp3, exp5 (serial; each ~1x baseline cost)
#   GPU 1 -> exp4 (~50x baseline cost, runs alone in background)
#
# Usage:
#   bash scripts/run_vpt_experiments.sh                 # run everything
#   bash scripts/run_vpt_experiments.sh exp3            # run a single group
#   WANDB_PROJECT=brain_vpt bash scripts/run_vpt_experiments.sh   # enable W&B

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VPT="$HERE/vpt"

case "${1:-all}" in
    exp1) bash "$VPT/exp1_linear.sh" 0 ;;
    exp2) bash "$VPT/exp2_transformer.sh" 0 ;;
    exp3) bash "$VPT/exp3_shared_vpt.sh" 0 ;;
    exp4) bash "$VPT/exp4_per_roi_vpt.sh" 1 ;;
    exp5) bash "$VPT/exp5_shared_vpt_attend_prompts.sh" 0 ;;
    all)
        cd "$(cd "$HERE/.." && pwd)"
        mkdir -p logs
        bash "$VPT/exp4_per_roi_vpt.sh" 1 > logs/exp4_per_roi_vpt.log 2>&1 &
        EXP4_PID=$!
        echo "exp4 (per-ROI VPT) launched on GPU 1, pid=$EXP4_PID, log=logs/exp4_per_roi_vpt.log"

        bash "$VPT/exp1_linear.sh"                    0 2>&1 | tee logs/exp1_linear.log
        bash "$VPT/exp2_transformer.sh"               0 2>&1 | tee logs/exp2_transformer.log
        bash "$VPT/exp3_shared_vpt.sh"                0 2>&1 | tee logs/exp3_shared_vpt.log
        bash "$VPT/exp5_shared_vpt_attend_prompts.sh" 0 2>&1 | tee logs/exp5_shared_vpt_attend_prompts.log

        wait "$EXP4_PID"
        echo "all experiments finished."
        ;;
    *) echo "usage: $0 [all|exp1|exp2|exp3|exp4|exp5]" ; exit 2 ;;
esac
