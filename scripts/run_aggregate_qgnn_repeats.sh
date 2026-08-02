#!/usr/bin/env bash
# Aggregation step for a repeated QGNN evaluation sweep (T4.10 seed-repeat
# study): reads N per-seed results JSON files (as written by
# run_evaluate_qgnn.sh) and reports mean +/- std per metric, comparable to a
# literature mean +/- std figure (e.g. Lorentz-EQGNN's 74.00% +/- 0.26%).
#
# Usage:
#   ./scripts/run_aggregate_qgnn_repeats.sh "results/qgnn/qg_jets_metrics_lorentz800_seed*.json" \
#       --output results/qgnn/qg_jets_metrics_lorentz800_summary.json
#   ./scripts/run_aggregate_qgnn_repeats.sh "..." --output ... --wandb-mode online
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

python experiments/aggregate_qgnn_repeats.py "$@"
