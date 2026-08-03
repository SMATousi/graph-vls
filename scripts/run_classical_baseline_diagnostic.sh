#!/usr/bin/env bash
# Classical-baseline diagnostic on frozen GVLS features (T4.10 followup,
# validation.md V-11): mirrors the QGNN comparability sweep's 5-trial
# repeated-training-subset protocol exactly, fitting LogisticRegression +
# a shallow MLPClassifier on the same frozen features instead of the QGNN
# circuit, to tell whether the ~67% accuracy ceiling is a GVLS feature-
# separability limit or a QGNN-training deficiency.
#
# Requires checkpoints/gvls_jets_m4_lorentz800.pt to already exist -- run
# run_pretrain_gvls_jets_final.sh (or the full
# run_qgnn_lorentz_comparability.sh pipeline) first if it doesn't.
#
# Usage:
#   ./scripts/run_classical_baseline_diagnostic.sh
#   ./scripts/run_classical_baseline_diagnostic.sh \
#       gvls_checkpoint_path=checkpoints/gvls_jets_m6_lorentz800.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

python experiments/classical_baseline_diagnostic.py "$@"
