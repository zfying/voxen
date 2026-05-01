# Shared config for the VPT experiment scripts under scripts/vpt/.
# Sourced (not executed); each per-exp script chdir's to the repo root and
# exports COMMON / WANDB_FLAGS / batch + epoch budgets.
#
# Override on the CLI, e.g.:
#   EPOCHS=5 bash scripts/vpt/exp3_shared_vpt.sh 0
#   WANDB_PROJECT=brain_vpt bash scripts/vpt/exp3_shared_vpt.sh 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SUBJ="${SUBJ:-1}"
RUN="${RUN:-1}"
ENC_LAYER="${ENC_LAYER:-1}"
READOUT="${READOUT:-rois_all}"
BACKBONE="${BACKBONE:-dinov2_q}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-15}"
BATCH_FAST="${BATCH_FAST:-16}"
BATCH_VPT="${BATCH_VPT:-4}"     # per-ROI VPT effective backbone batch = BATCH_VPT * N_ROI
ROI_CHUNK="${ROI_CHUNK:-25}"

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
