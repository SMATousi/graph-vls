#!/usr/bin/env bash
# T4.10 (plan.md Design Decision 12): dedicated Lorentz-EQGNN literature-
# comparability run -- GVLS pretraining (once) -> QGNN training + evaluation
# (repeated across several seeds) -- with the QGNN classifier's dataset
# protocol matched to arXiv:2411.01641 Section IV.A.1 exactly
# (configs/data/qg_jets_lorentz.yaml) and optimizer/lr/batch_size/epochs
# realigned to sota-table.png Table II (Quark-Gluon dataset, 4 qubits -- our
# M=4's own architectural choice, not a claim of matching theirs; see the
# discussion in validation.md V-10).
#
# Only the QGNN classifier is retrained per repeat. GVLS pretraining is
# skipped after the first run because it is a deterministic, frozen upstream
# feature extractor (Design Decision 8: no gradient updates to GVLS after
# pretraining; extract_latent_features has no gradient) -- repeating it would
# not add any variance the repeats are meant to sample, only cost.
#
# GVLS's own pretraining set size (T4.10 followup, validation.md V-11,
# user-directed) is decoupled from the QGNN's 800-jet comparability subset:
# GVLS trains on the full NUM_TRAIN-jet pool (see GVLS_TRAIN_SUBSET below),
# since the paper's 800-jet constraint is about ITS classifier's own
# training budget -- it has no unsupervised pretraining stage to be fair to.
# A same-800-jets-everywhere run had produced only ~59% mean test accuracy
# (vs. Lorentz-EQGNN's 74.00%), and a starved upstream feature extractor was
# one of two suspected causes alongside an uncalibrated decision threshold
# (see evaluate_qgnn.py).
#
# Each of the 5 trials draws a *different* class-balanced (400/400) 800-jet
# training subset from the same fixed 10000-jet training pool (user-directed,
# validation.md V-10) -- neither (a) reusing one identical training subset
# across all trials (would vary only the QGNN's own training seed, not the
# data) nor (b) true k-fold CV (would also re-partition val/test and likely
# require retraining the frozen GVLS encoder per fold). Validation and test
# stay fixed at 1250/1250 across every trial -- only the training subset and
# the QGNN's own training seed vary. See load_split_from_config's
# train_subset_seed vs. seed distinction (gvls/data/jets.py) and
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
# positionally into train/val/test -- NOT qg_jets.yaml's exact-50/50-draw-
# then-ratio-split. DATA_SEED seeds *this* outer partition and is fixed (not
# swept) across all trials, so every trial shares the exact same val/test.
#
# TRAIN_SUBSET(_BALANCED) then draws each QGNN trial's 800-jet training
# subset from the fixed NUM_TRAIN-jet pool -- 400/400 class-balanced (user-
# directed), reseeded per trial in the loop below via train_subset_seed
# (distinct from DATA_SEED, see load_split_from_config), so each trial trains
# on a different 800 jets while val/test never change. This applies to
# STAGE2/3 (the QGNN classifier) ONLY -- see GVLS_TRAIN_SUBSET below for why
# stage 1 does not inherit it.
K_GRAPH_CAP=8
NUM_TRAIN=10000
NUM_VAL=1250
NUM_TEST=1250
MIN_PARTICLES=10
TRAIN_SUBSET=800
TRAIN_SUBSET_BALANCED=true
DATA_SEED=42

# T4.10 followup (validation.md V-11, user-directed): Lorentz-EQGNN's
# 800-jet row constrains ITS end-to-end classifier's training budget -- it
# has no unsupervised pretraining stage at all, so nothing about comparability
# requires OUR upstream GVLS feature extractor to also be capped at 800
# jets. GVLS_TRAIN_SUBSET=null means stage 1 pretrains on the full
# NUM_TRAIN-jet pool instead (better (z_tilde, A_z) for every downstream
# QGNN trial, at zero cost to the comparability claim); set back to 800 to
# restore the original matched-budget behavior if ever needed for an
# apples-to-apples ablation against this change.
GVLS_TRAIN_SUBSET=null

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
# T4.10 followup (validation.md V-11, user-directed): re-running this script
# (e.g. to pick up the QGNN_NUM_LAYERS re-uploading fix below) must NOT
# retrain GVLS from scratch if a checkpoint from a prior run already exists
# at GVLS_CHECKPOINT_PATH -- pretraining is the expensive, unrelated-to-this-
# fix stage 1, and the whole point is to reuse its already-trained output.
# Set to false to force a fresh GVLS pretraining run regardless.
SKIP_GVLS_PRETRAIN_IF_CHECKPOINT_EXISTS=true

# --- Stage 2/3: QGNN training + evaluation, repeated per seed --
#     configs/train/qgnn_classifier.yaml ---
# T4.10 followup (validation.md V-11, user-directed): QGNNClassifier.encode_input
# re-uploads z_tilde dimension `layer % d` at each layer -- at the old
# num_layers=1 default, every jet's circuit only ever saw dimension 0 of
# GVLS's 8-dim z_tilde, discarding the other 7 before the circuit ever saw
# them (the classical-baseline diagnostic used the full 32-dim z_tilde and
# beat the QGNN by ~2.6 points on the same features). Setting
# QGNN_NUM_LAYERS=GVLS_LATENT_DIM re-uploads every dimension exactly once
# across the circuit's layers, closing that gap structurally. Must track
# GVLS_LATENT_DIM, not a fixed number, or re-uploading coverage silently
# regresses if GVLS_LATENT_DIM ever changes.
#
# First real run of this fix (num_layers=8, epochs=50) REGRESSED every
# metric (test_accuracy 66.88%->65.07%, train_accuracy 69.75%->68.60%,
# training_time_s 287->849) rather than improving them. Both train and test
# accuracy dropped together -- not a generalization/overfitting signature,
# an optimization-difficulty one: trainable weights went 9->44 (theta+bias
# per layer x8, plus readout), but SPSA estimates every parameter's gradient
# from a single shared random-direction perturbation each step, and that
# estimate's quality is known to degrade as parameter count grows. QGNN_EPOCHS
# raised 50->150 (user-directed): recovered most of the regression
# (65.07%->66.08%) but never surpassed the original num_layers=1 baseline
# (66.88%), at ~8.8x the compute (287s->2539s/trial) and ~4x the run-to-run
# std (0.30%->1.17%) -- three real runs now agree this direction alone
# doesn't pay off under SPSA. num_layers is kept at GVLS_LATENT_DIM anyway
# (user-directed) and QGNN_READOUT_MODE=learned (below) is layered on top as
# a complementary, lower-risk fix targeting a different gap (the fixed
# readout's inability to weight per-qubit measurements, unlike a classical
# linear model) -- see specs/phase4/validation.md V-11.
QGNN_NUM_LAYERS=${GVLS_LATENT_DIM}
# T4.10 followup (validation.md V-11, user-directed): "learned" replaces the
# fixed, unweighted sum(Z_i) readout with a trainable classical Linear(m,1)
# head on separately-measured per-qubit Z expectations -- targets the gap
# the classical-baseline diagnostic exposed (logistic regression/MLP use a
# LEARNED weighted combination of their inputs; the QGNN's old readout
# structurally could not). This head's own gradient is exact PyTorch
# autograd, not SPSA -- it does not add to the quantum circuit's own
# trainable-weight count that SPSA's joint-perturbation estimate has to
# spread across (unlike num_layers, which measurably did). Does modestly
# increase circuit evaluations per gradient step (measured: SPSA 2->8 pubs
# at m=4, i.e. proportional to m) -- small in absolute terms, unlike
# num_layers' depth-driven wall-clock blowup.
QGNN_READOUT_MODE=learned
# T4.10 (plan.md Design Decision 12): realigned to the Lorentz-EQGNN
# literature baseline for direct comparability.
QGNN_OPTIMIZER=adamw
QGNN_LR=0.001
QGNN_EPOCHS=150
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
    "data.seed=${DATA_SEED}"
)
WANDB_ONLINE_ARGS=()
if [[ "$WANDB_MODE" == "online" ]]; then
    WANDB_ONLINE_ARGS+=(--online)
fi

if [[ "$SKIP_GVLS_PRETRAIN_IF_CHECKPOINT_EXISTS" == "true" && -f "$GVLS_CHECKPOINT_PATH" ]]; then
    echo "=== [1/3] GVLS checkpoint already exists at ${GVLS_CHECKPOINT_PATH} -- skipping pretraining, reusing it as-is ==="
else
    echo "=== [1/3] Pretraining production GVLS checkpoint (M=${GVLS_M}, train_subset=${GVLS_TRAIN_SUBSET} i.e. full ${NUM_TRAIN}-jet pool) -- once, reused across all ${#QGNN_SEEDS[@]} QGNN repeats ==="
    STAGE1_ARGS=(
        "${DATA_ARGS[@]}"
        "data.train_subset=${GVLS_TRAIN_SUBSET}"
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
fi

for SEED in "${QGNN_SEEDS[@]}"; do
    QGNN_CHECKPOINT_PATH="checkpoints/qgnn_jets_m${GVLS_M}_lorentz800_seed${SEED}.pt"
    RESULTS_PATH="${RESULTS_DIR}/qg_jets_metrics_lorentz800_seed${SEED}.json"

    echo
    echo "=== [2/3] Training QGNN classifier (seed=${SEED}, training subset reseeded per trial, optimizer=${QGNN_OPTIMIZER}, gradient_method=${QGNN_GRADIENT_METHOD}) ==="
    STAGE2_ARGS=(
        "${DATA_ARGS[@]}"
        "data.train_subset=${TRAIN_SUBSET}"
        "data.train_subset_balanced=${TRAIN_SUBSET_BALANCED}"
        "data.train_subset_seed=${SEED}"
        "${WANDB_ONLINE_ARGS[@]}"
        "train.num_layers=${QGNN_NUM_LAYERS}"
        "train.readout_mode=${QGNN_READOUT_MODE}"
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
    echo "=== [3/3] Evaluating QGNN on held-out test jets (seed=${SEED}, same fixed test set every trial) ==="
    STAGE3_ARGS=(
        "${DATA_ARGS[@]}"
        "data.train_subset=${TRAIN_SUBSET}"
        "data.train_subset_balanced=${TRAIN_SUBSET_BALANCED}"
        "data.train_subset_seed=${SEED}"
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
