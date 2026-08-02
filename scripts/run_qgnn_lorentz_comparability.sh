#!/usr/bin/env bash
# T4.10 (plan.md Design Decision 12): dedicated Lorentz-EQGNN literature-
# comparability run -- runs the full jet -> QGNN pipeline (GVLS pretraining ->
# QGNN training -> evaluation) with num_jets/optimizer/lr/batch_size/epochs
# realigned to sota-table.png Table II (Quark-Gluon dataset, 4 qubits,
# matching our M=4). This is a fork of run_full_qgnn_pipeline.sh with those
# numbers pre-set, and with its own checkpoint/results paths (suffixed
# _lorentz800) so it does not overwrite the already-completed 20000-jet run's
# checkpoints/gvls_jets_m4.pt, checkpoints/qgnn_jets_m4.pt, or
# results/qgnn/qg_jets_metrics.json (commit 53e14f0: accuracy=0.695,
# auc=0.735).
#
# Usage:
#   ./scripts/run_qgnn_lorentz_comparability.sh
#
# Takes no CLI arguments by design (matches run_full_qgnn_pipeline.sh) -- to
# override one-off without editing this file, call the three run_*.sh
# scripts in scripts/ individually instead, each of which still accepts
# arbitrary key=value Hydra overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Hyperparameters -- edit these directly; running the script needs no flags.
# ============================================================================

# --- Shared: configs/data/qg_jets.yaml (used by all three stages) ---
# T4.10: num_jets=800 to reproduce the Lorentz-EQGNN "800 (subset)" figure.
# TRAIN_RATIO/VAL_RATIO's 70/15/15 split gives 560/120/120 jets; the
# literature table does not state a train/val/test breakdown for its own
# 800-jet subset, so this split is an explicit assumption (NFR-5).
NUM_JETS=800
K_GRAPH_CAP=8
TRAIN_RATIO=0.7
VAL_RATIO=0.15
DATA_SEED=42

# --- W&B ---
WANDB_MODE=online   # offline | online

# --- Stage 1: GVLS pretraining -- configs/train/jet_pretrain_final.yaml ---
GVLS_HIDDEN_DIM=32
GVLS_LATENT_DIM=8
GVLS_K=3
GVLS_M=4                       # pooled latent node count; T4.3 selected 4,
                                # and it happens to match Lorentz-EQGNN's 4 qubits
GVLS_GRAPH_METHOD=attention    # attention | fgp | nri
GVLS_PRIOR=isotropic           # isotropic | graph_mrf
GVLS_MP_ROUNDS=1
GVLS_LR=0.01
GVLS_BETA=0.001
GVLS_LAMBDA=1.0
GVLS_EPOCHS=200
GVLS_BATCH_SIZE=32
GVLS_SEED=42

# --- Stage 2: QGNN training -- configs/train/qgnn_classifier.yaml ---
QGNN_NUM_LAYERS=1
# T4.10 (plan.md Design Decision 12): realigned to the Lorentz-EQGNN
# literature baseline for direct comparability.
QGNN_OPTIMIZER=adamw
QGNN_LR=0.001
QGNN_EPOCHS=50
QGNN_BATCH_SIZE=16
QGNN_SEED=42
QGNN_GRADIENT_METHOD=spsa      # spsa (default, ~15x fewer circuit evals,
                                # T4.9) | param_shift (exact, much slower --
                                # see specs/phase4/validation.md V-9)
QGNN_SPSA_EPSILON=1e-6
QGNN_SPSA_BATCH_SIZE=1

# --- Checkpoint / results paths (suffixed _lorentz800 so this run's outputs
#     never collide with the 20000-jet run's checkpoints/results) ---
GVLS_CHECKPOINT_PATH="checkpoints/gvls_jets_m${GVLS_M}_lorentz800.pt"
QGNN_CHECKPOINT_PATH="checkpoints/qgnn_jets_m${GVLS_M}_lorentz800.pt"
RESULTS_PATH="results/qgnn/qg_jets_metrics_lorentz800.json"

# ============================================================================

# Built as one combined, always-non-empty array per stage (rather than
# referencing several separate arrays -- some possibly empty, e.g. when
# WANDB_MODE=offline -- directly in the call below): the macOS-default
# bash (3.2) treats "${empty_array[@]}" as an unbound-variable error under
# `set -u`, a bug fixed only in bash 4.4+.
DATA_ARGS=(
    "data.num_jets=${NUM_JETS}"
    "data.k_graph_cap=${K_GRAPH_CAP}"
    "data.train_ratio=${TRAIN_RATIO}"
    "data.val_ratio=${VAL_RATIO}"
    "data.seed=${DATA_SEED}"
)

STAGE1_ARGS=("${DATA_ARGS[@]}")
STAGE2_ARGS=("${DATA_ARGS[@]}")
STAGE3_ARGS=("${DATA_ARGS[@]}")
if [[ "$WANDB_MODE" == "online" ]]; then
    STAGE1_ARGS+=(--online)
    STAGE2_ARGS+=(--online)
    STAGE3_ARGS+=(--online)
fi

STAGE1_ARGS+=(
    "train.hidden_dim=${GVLS_HIDDEN_DIM}"
    "train.latent_dim=${GVLS_LATENT_DIM}"
    "train.k=${GVLS_K}"
    "train.m=${GVLS_M}"
    "train.graph_method=${GVLS_GRAPH_METHOD}"
    "train.prior=${GVLS_PRIOR}"
    "train.mp_rounds=${GVLS_MP_ROUNDS}"
    "train.lr=${GVLS_LR}"
    "train.beta=${GVLS_BETA}"
    "train.lambda_=${GVLS_LAMBDA}"
    "train.epochs=${GVLS_EPOCHS}"
    "train.batch_size=${GVLS_BATCH_SIZE}"
    "train.seed=${GVLS_SEED}"
    "checkpoint_path=${GVLS_CHECKPOINT_PATH}"
)

STAGE2_ARGS+=(
    "train.num_layers=${QGNN_NUM_LAYERS}"
    "train.optimizer=${QGNN_OPTIMIZER}"
    "train.lr=${QGNN_LR}"
    "train.epochs=${QGNN_EPOCHS}"
    "train.batch_size=${QGNN_BATCH_SIZE}"
    "train.seed=${QGNN_SEED}"
    "train.gradient_method=${QGNN_GRADIENT_METHOD}"
    "train.spsa_epsilon=${QGNN_SPSA_EPSILON}"
    "train.spsa_batch_size=${QGNN_SPSA_BATCH_SIZE}"
    "gvls_checkpoint_path=${GVLS_CHECKPOINT_PATH}"
    "qgnn_checkpoint_path=${QGNN_CHECKPOINT_PATH}"
)

STAGE3_ARGS+=(
    "gvls_checkpoint_path=${GVLS_CHECKPOINT_PATH}"
    "qgnn_checkpoint_path=${QGNN_CHECKPOINT_PATH}"
    "results_path=${RESULTS_PATH}"
)

echo "=== [1/3] Pretraining production GVLS checkpoint (M=${GVLS_M}, num_jets=${NUM_JETS}) ==="
"${SCRIPT_DIR}/run_pretrain_gvls_jets_final.sh" "${STAGE1_ARGS[@]}"

echo
echo "=== [2/3] Training QGNN classifier (optimizer=${QGNN_OPTIMIZER}, gradient_method=${QGNN_GRADIENT_METHOD}) ==="
"${SCRIPT_DIR}/run_train_qgnn.sh" "${STAGE2_ARGS[@]}"

echo
echo "=== [3/3] Evaluating QGNN on held-out test jets ==="
"${SCRIPT_DIR}/run_evaluate_qgnn.sh" "${STAGE3_ARGS[@]}"
