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
