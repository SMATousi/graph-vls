#!/usr/bin/env bash
# GVLS-side (k, beta, prior) sweep at fixed M=4 -- 2026-08-04 audit followup.
#
# Answers which of the three GVLS knobs the audit found inert actually move
# compression fidelity and downstream separability:
#   k     -- k=3 (production default) makes A_z the *complete* graph on every
#            jet at M=4, since LatentGraphLearner uses k = min(self.k, N-1);
#            only k=1/2 leave any latent topology to learn.
#   beta  -- 0.001 (production default) leaves the KL at ~0.25% of the
#            post-training loss, against assignment_link_loss's ~80%.
#   prior -- isotropic (production default) means kl_graph_mrf is never
#            called, so A_z never enters the prior at all.
#
# 24 configs (3 k x 4 beta x 2 prior), each a full GVLS pretraining run on the
# production Lorentz 10000-jet training pool for 30 epochs, then scored on
# held-out compression metrics + a 5-trial classical-baseline downstream probe
# (see experiments/gvls_prior_sweep.py's module docstring for why the
# classical baseline, not the QGNN, is the right downstream signal here).
#
# Runtime: ~2 hours total on a CPU machine (~280s/config, measured locally at
# ~0.8 ms per jet-step). No GPU needed -- jets are tiny and the loop is
# per-jet, so this is dominated by Python/dense-op overhead, not matmul size.
# Nothing here touches the QGNN or any quantum simulation.
#
# Writes results/compression/qg_jets_prior_sweep.csv (one row per config,
# flushed after each config, so a partial run is still readable if it dies).
#
# Usage:
#   ./scripts/run_gvls_prior_sweep.sh
#
# Recommended for a long remote run (survives disconnect, streams progress):
#   nohup ./scripts/run_gvls_prior_sweep.sh > gvls_prior_sweep.log 2>&1 &
#   tail -f gvls_prior_sweep.log
#
# Takes no CLI arguments by design (matches run_qgnn_lorentz_comparability.sh)
# -- the grid and the fixed production hyperparameters are declared as module
# constants at the top of experiments/gvls_prior_sweep.py; edit them there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

python experiments/gvls_prior_sweep.py
