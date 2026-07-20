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

python experiments/evaluate_qgnn.py "$@"
