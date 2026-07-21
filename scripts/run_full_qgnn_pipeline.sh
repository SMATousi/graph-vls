#!/usr/bin/env bash
# Convenience wrapper: runs the full jet -> QGNN pipeline in order --
# GVLS pretraining (T4.3-final) -> QGNN training (T4.5) -> evaluation (T4.6,
# metrics only). Equivalent to running the three run_*.sh scripts in this
# directory back to back with their default configs.
#
# Usage:
#   ./scripts/run_full_qgnn_pipeline.sh              # offline W&B (default)
#   ./scripts/run_full_qgnn_pipeline.sh --online     # sync all W&B runs live
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] Pretraining production GVLS checkpoint ==="
"${SCRIPT_DIR}/run_pretrain_gvls_jets_final.sh" "$@"

echo
echo "=== [2/3] Training QGNN classifier ==="
"${SCRIPT_DIR}/run_train_qgnn.sh" "$@"

echo
echo "=== [3/3] Evaluating QGNN on held-out test jets ==="
"${SCRIPT_DIR}/run_evaluate_qgnn.sh" "$@"
