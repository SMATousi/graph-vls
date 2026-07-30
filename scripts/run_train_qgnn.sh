#!/usr/bin/env bash
# Step 2/3 of the jet -> QGNN pipeline (T4.5): freeze the pretrained GVLS
# checkpoint, extract (z_tilde, A_z) for every jet, and train the QGNN
# classifier's circuit parameters. Requires run_pretrain_gvls_jets_final.sh
# to have already produced checkpoints/gvls_jets_m*.pt.
#
# Usage:
#   ./scripts/run_train_qgnn.sh                 # offline W&B (default)
#   ./scripts/run_train_qgnn.sh --online        # sync to W&B live
#   ./scripts/run_train_qgnn.sh train.epochs=100 train.num_layers=2
#   ./scripts/run_train_qgnn.sh train.optimizer=adam train.lr=0.05 \
#       train.batch_size=32   # restore the pre-T4.10 configuration
#   ./scripts/run_train_qgnn.sh data.num_jets=800   # T4.10 literature-
#       # comparability subset (Lorentz-EQGNN, see specs/phase4/plan.md
#       # Design Decision 12) -- or edit run_full_qgnn_pipeline.sh's NUM_JETS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

# --online is a convenience alias for the Hydra override wandb.mode=online;
# everything else passes through unchanged (e.g. train.epochs=100).
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--online" ]]; then
        ARGS+=("wandb.mode=online")
    else
        ARGS+=("$arg")
    fi
done

python experiments/train_qgnn.py "${ARGS[@]}"
