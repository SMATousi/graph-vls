#!/usr/bin/env bash
# Step 3/3 of the jet -> QGNN pipeline (T4.6, metrics only -- no literature
# comparison, see experiments/evaluate_qgnn.py's docstring): full
# classification metrics on the held-out test split. Requires
# run_train_qgnn.sh to have already produced checkpoints/qgnn_jets_m*.pt.
#
# Usage:
#   ./scripts/run_evaluate_qgnn.sh              # offline W&B (default)
#   ./scripts/run_evaluate_qgnn.sh --online     # sync to W&B live
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

# --online is a convenience alias for the Hydra override wandb.mode=online;
# everything else passes through unchanged.
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--online" ]]; then
        ARGS+=("wandb.mode=online")
    else
        ARGS+=("$arg")
    fi
done

python experiments/evaluate_qgnn.py "${ARGS[@]}"
