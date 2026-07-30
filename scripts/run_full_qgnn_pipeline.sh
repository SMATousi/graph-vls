#!/usr/bin/env bash
# Convenience wrapper: runs the full jet -> QGNN pipeline in order --
# GVLS pretraining (T4.3-final) -> QGNN training (T4.5) -> evaluation (T4.6,
# metrics only).
#
# All hyperparameters are declared as variables directly below -- edit them
# here and re-run, no CLI arguments needed. Defaults mirror configs/*.yaml.
#
# Each stage is invoked with only the override keys valid for THAT stage's
# Hydra config (a previous version of this script forwarded one shared
# argument list to all three stages, which hard-errored -- Hydra configs are
# struct-checked -- the moment a stage-specific key like train.gradient_method
# or train.num_layers reached a stage that doesn't have it; with set -euo
# pipefail that aborted the whole pipeline, sometimes after already burning
# the time to run earlier stages. Building each stage's argument list
# explicitly from these variables closes that off structurally.)
#
# Usage:
#   ./scripts/run_full_qgnn_pipeline.sh
#
# Takes no CLI arguments by design (see above) -- to override one-off without
# editing this file, call the three run_*.sh scripts in scripts/ individually
# instead, each of which still accepts arbitrary key=value Hydra overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Hyperparameters -- edit these directly; running the script needs no flags.
# ============================================================================

# --- Shared: configs/data/qg_jets.yaml (used by all three stages) ---
# T4.10 (literature comparability, plan.md Design Decision 12): to reproduce
# the Lorentz-EQGNN literature baseline (sota-table.png Table II, Quark-Gluon
# dataset) as closely as possible, set NUM_JETS=800 -- a dedicated
# comparability run, additive to (not a replacement for) this 20000 default.
# TRAIN_RATIO/VAL_RATIO's 70/15/15 split then gives 560/120/120 jets; the
# literature table itself does not state a train/val/test breakdown for its
# "800 (subset)" figure, so this split is an explicit assumption (NFR-5).
NUM_JETS=20000
K_GRAPH_CAP=8
TRAIN_RATIO=0.7
VAL_RATIO=0.15
DATA_SEED=42

# --- W&B ---
WANDB_MODE=offline   # offline | online

# --- Stage 1: GVLS pretraining -- configs/train/jet_pretrain_final.yaml ---
GVLS_HIDDEN_DIM=32
GVLS_LATENT_DIM=8
GVLS_K=3
GVLS_M=4                       # pooled latent node count; T4.3 selected 4
GVLS_GRAPH_METHOD=attention    # attention | fgp | nri
GVLS_PRIOR=isotropic           # isotropic | graph_mrf
GVLS_MP_ROUNDS=1
GVLS_LR=0.01
GVLS_BETA=0.001
GVLS_LAMBDA=1.0
GVLS_EPOCHS=100
GVLS_BATCH_SIZE=32
GVLS_SEED=42

# --- Stage 2: QGNN training -- configs/train/qgnn_classifier.yaml ---
QGNN_NUM_LAYERS=1
# T4.10: optimizer/lr/batch_size default to the Lorentz-EQGNN literature
# baseline's protocol (plan.md Design Decision 12) -- set QGNN_OPTIMIZER=adam,
# QGNN_LR=0.05, QGNN_BATCH_SIZE=32 to restore the pre-T4.10 configuration.
QGNN_OPTIMIZER=adamw           # adamw (default, T4.10) | adam (pre-T4.10)
QGNN_LR=0.001
QGNN_EPOCHS=50
QGNN_BATCH_SIZE=16
QGNN_SEED=42
QGNN_GRADIENT_METHOD=spsa      # spsa (default, ~15x fewer circuit evals,
                                # T4.9) | param_shift (exact, much slower --
                                # see specs/phase4/validation.md V-9)
QGNN_SPSA_EPSILON=1e-6
QGNN_SPSA_BATCH_SIZE=1

# --- Checkpoint / results paths (kept consistent across stages via GVLS_M
#     automatically -- change GVLS_M above and these follow without having
#     to edit them separately for each stage) ---
GVLS_CHECKPOINT_PATH="checkpoints/gvls_jets_m${GVLS_M}.pt"
QGNN_CHECKPOINT_PATH="checkpoints/qgnn_jets_m${GVLS_M}.pt"
RESULTS_PATH="results/qgnn/qg_jets_metrics.json"

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

echo "=== [1/3] Pretraining production GVLS checkpoint (M=${GVLS_M}) ==="
"${SCRIPT_DIR}/run_pretrain_gvls_jets_final.sh" "${STAGE1_ARGS[@]}"

echo
echo "=== [2/3] Training QGNN classifier (gradient_method=${QGNN_GRADIENT_METHOD}) ==="
"${SCRIPT_DIR}/run_train_qgnn.sh" "${STAGE2_ARGS[@]}"

echo
echo "=== [3/3] Evaluating QGNN on held-out test jets ==="
"${SCRIPT_DIR}/run_evaluate_qgnn.sh" "${STAGE3_ARGS[@]}"
