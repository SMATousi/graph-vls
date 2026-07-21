#!/usr/bin/env bash
# Step 3/3 of the jet -> QGNN pipeline (T4.6, metrics only -- no literature
# comparison, see experiments/evaluate_qgnn.py's docstring): full
# classification metrics on the held-out test split. Requires
# run_train_qgnn.sh to have already produced checkpoints/qgnn_jets_m*.pt.
#
# Usage:
#   ./scripts/run_evaluate_qgnn.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

# This script doesn't use W&B, so --online (understood by the other two
# run_*.sh scripts) is silently dropped here rather than passed through to
# Hydra, letting run_full_qgnn_pipeline.sh forward the same "$@" to all three
# steps uniformly.
ARGS=()
for arg in "$@"; do
    if [[ "$arg" != "--online" ]]; then
        ARGS+=("$arg")
    fi
done

python experiments/evaluate_qgnn.py "${ARGS[@]}"
