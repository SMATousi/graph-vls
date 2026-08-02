#!/usr/bin/env bash
# T4.10 (plan.md Design Decision 12): dedicated Lorentz-EQGNN literature-
# comparability run -- GVLS pretraining (once) -> QGNN training + evaluation
# (repeated across several seeds) -- with the dataset protocol matched to
# arXiv:2411.01641 Section IV.A.1 exactly (configs/data/qg_jets_lorentz.yaml)
# and optimizer/lr/batch_size/epochs realigned to sota-table.png Table II
# (Quark-Gluon dataset, 4 qubits -- our M=4's own architectural choice, not a
# claim of matching theirs; see the discussion in validation.md V-10).
#
# Only the QGNN classifier is retrained per repeat. GVLS pretraining is
# skipped after the first run because it is a deterministic, frozen upstream
# feature extractor (Design Decision 8: no gradient updates to GVLS after
# pretraining; extract_latent_features has no gradient) -- repeating it would
# not add any variance the QGNN-training repeats aren't already sampling
# over, only cost. This means the reported mean +/- std captures the QGNN
# classifier's own training-seed variance (weight init, minibatch order,
# SPSA gradient noise), not full end-to-end pipeline variance -- an explicit,
# documented scope narrowing (NFR-5), not an oversight. See
# specs/phase4/validation.md V-10.
#
# Checkpoints/results use paths suffixed _lorentz800 (and per-seed for the
# QGNN stage) so this run's outputs never collide with the already-completed
# 20000-jet run's checkpoints/gvls_jets_m4.pt, checkpoints/qgnn_jets_m4.pt,
# or results/qgnn/qg_jets_metrics.json (commit 53e14f0: accuracy=0.695,
# auc=0.735). Each seed's W&B run also gets a distinct name/group/tags
# (validation.md V-10) so it's identifiable in the W&B UI without opening
# checkpoint configs by hand.
#
# Usage:
#   ./scripts/run_qgnn_lorentz_comparability.sh
#
# Takes no CLI arguments by design (matches run_full_qgnn_pipeline.sh) -- to
# override one-off without editing this file, call the individual scripts/
# run_*.sh scripts (and experiments/aggregate_qgnn_repeats.py) directly
# instead, each of which still accepts arbitrary key=value Hydra overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Hyperparameters -- edit these directly; running the script needs no flags.
# ============================================================================

# --- Shared: configs/data/qg_jets_lorentz.yaml (used by all stages) ---
# T4.10: reproduces the Lorentz-EQGNN paper's data protocol exactly
# (arXiv:2411.01641 Section IV.A.1, specs/phase4/validation.md V-10) --
# a single random draw of NUM_TRAIN+NUM_VAL+NUM_TEST jets with
# >= MIN_PARTICLES particles each (no forced 50/50 class balance), sliced
# positionally into train/val/test, with TRAIN_SUBSET further shrinking only
# the training set (val/test stay fixed at NUM_VAL/NUM_TEST) -- NOT
# qg_jets.yaml's exact-50/50-draw-then-ratio-split. DATA_SEED is fixed (not
# swept) across all repeats below so every repeat trains/evaluates on the
# exact same split -- only the QGNN's own training seed varies between
# repeats.
K_GRAPH_CAP=8
NUM_TRAIN=10000
NUM_VAL=1250
NUM_TEST=1250
MIN_PARTICLES=10
TRAIN_SUBSET=800
DATA_SEED=42

# --- W&B ---
WANDB_MODE=online   # offline | online
WANDB_GROUP=qgnn-jet-classification-lorentz800
WANDB_TAGS="[lorentz-comparability,qg-jets-800]"

# --- Stage 1: GVLS pretraining (once) -- configs/train/jet_pretrain_final.yaml ---
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
GVLS_CHECKPOINT_PATH="checkpoints/gvls_jets_m${GVLS_M}_lorentz800.pt"

# --- Stage 2/3: QGNN training + evaluation, repeated per seed --
#     configs/train/qgnn_classifier.yaml ---
QGNN_NUM_LAYERS=1
# T4.10 (plan.md Design Decision 12): realigned to the Lorentz-EQGNN
# literature baseline for direct comparability.
QGNN_OPTIMIZER=adamw
QGNN_LR=0.001
QGNN_EPOCHS=50
QGNN_BATCH_SIZE=16
QGNN_GRADIENT_METHOD=spsa      # spsa (default, ~15x fewer circuit evals,
                                # T4.9) | param_shift (exact, much slower --
                                # see specs/phase4/validation.md V-9)
QGNN_SPSA_EPSILON=1e-6
QGNN_SPSA_BATCH_SIZE=1

# 5 repeats (seeds arbitrary, just mutually distinct) to report the QGNN's
# test accuracy as mean +/- std, comparable to Lorentz-EQGNN's reported
# 74.00% +/- 0.26%.
QGNN_SEEDS=(42 43 44 45 46)

RESULTS_DIR="results/qgnn"
RESULTS_PATTERN="${RESULTS_DIR}/qg_jets_metrics_lorentz800_seed*.json"
SUMMARY_PATH="${RESULTS_DIR}/qg_jets_metrics_lorentz800_summary.json"

# ============================================================================

# Built as one combined, always-non-empty array (rather than referencing
# several separate arrays -- some possibly empty, e.g. when
# WANDB_MODE=offline -- directly in the call below): the macOS-default
# bash (3.2) treats "${empty_array[@]}" as an unbound-variable error under
# `set -u`, a bug fixed only in bash 4.4+.
DATA_ARGS=(
    "data=qg_jets_lorentz"
    "data.k_graph_cap=${K_GRAPH_CAP}"
    "data.num_train=${NUM_TRAIN}"
    "data.num_val=${NUM_VAL}"
    "data.num_test=${NUM_TEST}"
    "data.min_particles=${MIN_PARTICLES}"
    "data.train_subset=${TRAIN_SUBSET}"
    "data.seed=${DATA_SEED}"
)
WANDB_ONLINE_ARGS=()
if [[ "$WANDB_MODE" == "online" ]]; then
    WANDB_ONLINE_ARGS+=(--online)
fi

echo "=== [1/3] Pretraining production GVLS checkpoint (M=${GVLS_M}, train_subset=${TRAIN_SUBSET}) -- once, reused across all ${#QGNN_SEEDS[@]} QGNN repeats ==="
STAGE1_ARGS=(
    "${DATA_ARGS[@]}"
    "${WANDB_ONLINE_ARGS[@]}"
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
    "wandb.name=gvls-jets-M${GVLS_M}-lorentz800"
    "wandb.group=${WANDB_GROUP}"
    "wandb.tags=${WANDB_TAGS}"
)
"${SCRIPT_DIR}/run_pretrain_gvls_jets_final.sh" "${STAGE1_ARGS[@]}"

for SEED in "${QGNN_SEEDS[@]}"; do
    QGNN_CHECKPOINT_PATH="checkpoints/qgnn_jets_m${GVLS_M}_lorentz800_seed${SEED}.pt"
    RESULTS_PATH="${RESULTS_DIR}/qg_jets_metrics_lorentz800_seed${SEED}.json"

    echo
    echo "=== [2/3] Training QGNN classifier (seed=${SEED}, optimizer=${QGNN_OPTIMIZER}, gradient_method=${QGNN_GRADIENT_METHOD}) ==="
    STAGE2_ARGS=(
        "${DATA_ARGS[@]}"
        "${WANDB_ONLINE_ARGS[@]}"
        "train.num_layers=${QGNN_NUM_LAYERS}"
        "train.optimizer=${QGNN_OPTIMIZER}"
        "train.lr=${QGNN_LR}"
        "train.epochs=${QGNN_EPOCHS}"
        "train.batch_size=${QGNN_BATCH_SIZE}"
        "train.seed=${SEED}"
        "train.gradient_method=${QGNN_GRADIENT_METHOD}"
        "train.spsa_epsilon=${QGNN_SPSA_EPSILON}"
        "train.spsa_batch_size=${QGNN_SPSA_BATCH_SIZE}"
        "gvls_checkpoint_path=${GVLS_CHECKPOINT_PATH}"
        "qgnn_checkpoint_path=${QGNN_CHECKPOINT_PATH}"
        "wandb.name=qgnn-M${GVLS_M}-lorentz800-seed${SEED}"
        "wandb.group=${WANDB_GROUP}"
        "wandb.tags=${WANDB_TAGS}"
    )
    "${SCRIPT_DIR}/run_train_qgnn.sh" "${STAGE2_ARGS[@]}"

    echo
    echo "=== [3/3] Evaluating QGNN on held-out test jets (seed=${SEED}) ==="
    STAGE3_ARGS=(
        "${DATA_ARGS[@]}"
        "${WANDB_ONLINE_ARGS[@]}"
        "gvls_checkpoint_path=${GVLS_CHECKPOINT_PATH}"
        "qgnn_checkpoint_path=${QGNN_CHECKPOINT_PATH}"
        "results_path=${RESULTS_PATH}"
        "wandb.name=qgnn-eval-M${GVLS_M}-lorentz800-seed${SEED}"
        "wandb.group=${WANDB_GROUP}"
        "wandb.tags=${WANDB_TAGS}"
    )
    "${SCRIPT_DIR}/run_evaluate_qgnn.sh" "${STAGE3_ARGS[@]}"
done

echo
echo "=== Aggregating ${#QGNN_SEEDS[@]} repeats into mean +/- std ==="
AGGREGATE_WANDB_MODE="offline"
if [[ "$WANDB_MODE" == "online" ]]; then
    AGGREGATE_WANDB_MODE="online"
fi
"${SCRIPT_DIR}/run_aggregate_qgnn_repeats.sh" \
    "${RESULTS_PATTERN}" \
    --output "${SUMMARY_PATH}" \
    --wandb-mode "${AGGREGATE_WANDB_MODE}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-name "qgnn-M${GVLS_M}-lorentz800-summary"
