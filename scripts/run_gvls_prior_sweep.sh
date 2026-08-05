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
# PERFORMANCE -- read this before changing OMP/thread settings.
# The first version of this sweep ran ~9x slower on a remote machine than
# predicted (~40 min/config vs. a locally measured ~4 min). Cause: jets are
# ~43 particles, so every tensor op is a ~43x43 matrix, and BLAS thread-pool
# dispatch costs more than the arithmetic at that size. Wall-clock therefore
# gets *worse* with more threads (measured 0.71 ms/jet-step at 1 thread vs.
# 0.82 at 8 on a 10-core box) and the penalty grows with core count, so a
# server defaulting torch to 32-128 threads pays it hardest.
#
# The fix is two multiplicative changes, both handled inside the Python:
#   1. every worker pins itself to a single thread (BLAS env vars are exported
#      below too, since several backends only read them at import time);
#   2. the 24 independent configs run as parallel processes, which then
#      scales ~linearly in core count.
# Measured after both: 24 configs in 4.4 min at 5 epochs / 8 workers, vs. the
# original serial run's ~40 min for a single config. Expect roughly 15-25 min
# for the full 30-epoch grid on an 8-16 core machine.
#
# Do NOT "optimize" this by raising thread counts -- that is the bug, not the
# cure. Add workers instead.
#
# Writes results/compression/qg_jets_prior_sweep.csv (one row per config,
# flushed as each config finishes, so a partial run is still readable).
#
# Usage:
#   ./scripts/run_gvls_prior_sweep.sh                    # all cores, 30 epochs
#   ./scripts/run_gvls_prior_sweep.sh --workers 16
#   ./scripts/run_gvls_prior_sweep.sh --epochs 5         # fast first pass
#
# For a long remote run (survives disconnect, streams progress):
#   nohup ./scripts/run_gvls_prior_sweep.sh > gvls_prior_sweep.log 2>&1 &
#   tail -f gvls_prior_sweep.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_activate_env.sh
source "${SCRIPT_DIR}/_activate_env.sh"

# Belt-and-braces: the Python sets these too, but several BLAS backends read
# them only at library load time, which can precede any Python-level setting
# depending on how torch was built. Harmless if redundant.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python experiments/gvls_prior_sweep.py "$@"
