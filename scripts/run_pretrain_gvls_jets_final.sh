#!/usr/bin/env bash
# Step 1/3 of the jet -> QGNN pipeline: train and persist the production
# PooledGVLS checkpoint at a fixed M (T4.3 selected M=4; see
# specs/phase4/validation.md V-3). Prerequisite for run_train_qgnn.sh.
#
# Usage:
#   ./scripts/run_pretrain_gvls_jets_final.sh
#   ./scripts/run_pretrain_gvls_jets_final.sh train.m=6 train.epochs=200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

python experiments/pretrain_gvls_jets_final.py "$@"
