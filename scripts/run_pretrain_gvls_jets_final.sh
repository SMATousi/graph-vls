#!/usr/bin/env bash
# Step 1/3 of the jet -> QGNN pipeline: train and persist the production
# PooledGVLS checkpoint at a fixed M (T4.3 selected M=4; see
# specs/phase4/validation.md V-3). Prerequisite for run_train_qgnn.sh.
#
# Usage:
#   ./scripts/run_pretrain_gvls_jets_final.sh              # offline W&B (default)
#   ./scripts/run_pretrain_gvls_jets_final.sh --online     # sync to W&B live
#   ./scripts/run_pretrain_gvls_jets_final.sh train.m=6 train.epochs=200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

# --online is a convenience alias for the Hydra override wandb.mode=online;
# everything else passes through unchanged (e.g. train.m=6).
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--online" ]]; then
        ARGS+=("wandb.mode=online")
    else
        ARGS+=("$arg")
    fi
done

python experiments/pretrain_gvls_jets_final.py "${ARGS[@]}"
