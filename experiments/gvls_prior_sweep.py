"""GVLS-side (k, beta, prior) sweep at fixed M=4 (2026-08-04 audit followup).

Motivation: an audit of `src/gvls/models/` against `specs/mission.md`'s six
claimed GVLS components found three of them effectively inert in the
production jet configuration (`configs/train/jet_pretrain.yaml`):

  1. `prior: isotropic` means `kl_graph_mrf` is never called, so the learned
     latent graph `A_z` never enters the prior -- mission component 5
     ("graph-aware prior and ELBO") is switched off.
  2. `k=3` at `M=4` makes `A_z` the *complete* graph on every jet, since
     `LatentGraphLearner` uses `k = min(self.k, N-1)` and `N-1 == 3` here.
     There is no latent topology to learn, only 6 edge weights -- which also
     makes the QGNN's topology-equivariant ansatz (plan.md Design Decision 2)
     architecturally vacuous, every jet's circuit having identical structure.
  3. `beta=0.001` leaves the KL term at ~0.25% of the post-training loss
     (measured), against `assignment_link_loss`'s ~80% -- the model is
     trained overwhelmingly by DiffPool's auxiliary input-graph link
     objective, not by the ELBO.

This sweep varies exactly those three knobs and holds everything else at the
production config, to establish which (if any) actually move compression
fidelity and downstream separability before any GVLS architecture work
(e.g. the deferred auxiliary-supervision idea, plan.md's "Stretch / explicitly
deferred") is designed.

Downstream signal uses the **classical baseline** (`gvls.eval.classical_
baseline`), not the QGNN: validation.md V-11 Step 2 established that the QGNN
currently underperforms a plain logistic regression on identical frozen
features (66.88% vs. 69.07%), so classical accuracy is the ceiling this sweep
needs to move, and it costs seconds per config instead of the QGNN's
~849s/trial. Whichever configs win here are the ones worth spending real QGNN
runs on.

Data protocol matches the production comparability run exactly
(`configs/data/qg_jets_lorentz.yaml`): the fixed 10000/1250/1250 Lorentz
split, GVLS pretrained on the full 10000-jet training pool (validation.md
V-11's dominant fix), classical baselines trained on the same 5 balanced
800-jet subsets and scored on the same fixed 1250-jet test set.

Performance (2026-08-04: the first attempt at this sweep ran ~9x slower than
predicted on a remote machine -- ~40 min/config against a locally measured
~4 min/config -- and this module is now built around why).

  * **Thread pinning is mandatory, not an optimization.** Jets are ~43
    particles, so every tensor op is a ~43x43 matrix. BLAS thread-pool
    dispatch costs more than the arithmetic at that size, so wall-clock gets
    *worse* as thread count rises: measured 0.71 ms/jet-step at 1 thread vs.
    0.82 at 8 on a 10-core machine, and the penalty grows with core count --
    a remote server whose torch defaults to 32-128 threads pays it hardest.
    Every worker here pins itself to one thread (`_pin_threads`), and the
    BLAS env vars are set before torch is imported, since several backends
    read them only at load time.
  * **Configs are run in parallel processes.** The 24 grid points are fully
    independent -- no shared state, no ordering -- so with single-threaded
    workers this scales close to linearly in core count. Serial-at-1-thread
    would be ~1.4h locally; at `--workers 12` it is ~10 min.

Usage:
    python experiments/gvls_prior_sweep.py                 # all cores
    python experiments/gvls_prior_sweep.py --workers 8
    python experiments/gvls_prior_sweep.py --epochs 10     # faster first pass
"""

from __future__ import annotations

import argparse
import csv
import os

# BLAS backends read these at import time, so they must be set before torch is
# imported -- setting torch.set_num_threads() alone is too late for some of
# them. See the module docstring for why single-threaded is the fast path here.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gvls.compression.jet_sweep import (  # noqa: E402
    evaluate_pooled_gvls_on_jets,
    jet_adjacency,
    jet_pos_weight,
    train_pooled_gvls_on_jets,
)
from gvls.data.jets import load_split_from_config, subsample_train_balanced  # noqa: E402
from gvls.eval.classical_baseline import evaluate_classical_baselines  # noqa: E402
from gvls.losses.elbo import kl_graph_mrf, kl_isotropic  # noqa: E402
from gvls.models.pooling import assignment_entropy, assignment_link_loss  # noqa: E402
from gvls.qgnn_training import extract_latent_features  # noqa: E402

# --- fixed at the production jet config (configs/train/jet_pretrain.yaml) ---
NUM_CLUSTERS = 4
LATENT_DIM = 8
HIDDEN_DIM = 32
GRAPH_METHOD = "attention"
MP_ROUNDS = 1
LR = 0.01
LAMBDA_ = 1.0
EPOCHS = 30
BATCH_SIZE = 32
ENTROPY_WEIGHT = 0.1
AUX_LINK_WEIGHT = 5.0
F1_NEGATIVE_RATIO = 1.0
SEED = 42
NUM_FEATURES = 18

# --- the swept axes ---
# k=3 is the production default and is the point where A_z goes complete at
# M=4; k=1/2 are the only settings that leave any topology to learn.
K_GRID = [1, 2, 3]
# beta=0.001 is the production default; the rest probe whether the
# variational term does anything once it is not 0.25% of the loss.
BETA_GRID = [0.001, 0.01, 0.1, 1.0]
PRIOR_GRID = ["isotropic", "graph_mrf"]

# --- downstream (classical-baseline) protocol, matching the QGNN run ---
NUM_TRIALS = 5
TRAIN_SUBSET_PER_CLASS = 400  # 800-jet balanced subset, Lorentz-EQGNN's row

RESULT_PATH = Path("results/compression/qg_jets_prior_sweep.csv")

RESULT_FIELDS = [
    "k",
    "beta",
    "prior",
    # compression fidelity (held-out val jets)
    "avg_reconstruction_f1",
    "avg_bits_per_edge",
    "avg_latent_density",
    "avg_num_latent_edges",
    "avg_edge_compression_ratio",
    # is the latent graph actually sparse, or complete on every jet?
    "frac_jets_complete_a_z",
    "min_num_latent_edges",
    "max_num_latent_edges",
    # post-training loss decomposition (does the variational term matter?)
    "loss_share_recon",
    "loss_share_kl",
    "loss_share_entropy",
    "loss_share_link",
    "raw_kl_per_node",
    "posterior_sigma_mean",
    # downstream separability of the frozen features (5 balanced-800 trials)
    "logreg_accuracy_mean",
    "logreg_accuracy_std",
    "logreg_auc_mean",
    "mlp_accuracy_mean",
    "mlp_accuracy_std",
    "mlp_auc_mean",
    "train_time_s",
]


_THREADS_PINNED = False


def _pin_threads() -> None:
    """Force single-threaded tensor ops (see the module docstring's perf note).

    Called in every worker process, not just the parent: `torch.set_num_threads`
    is per-process, and a forked child does not reliably inherit it.

    Idempotent, and deliberately tolerant of failure on the interop setting:
    `set_num_interop_threads` raises if called more than once or after any
    parallel work has started, and it is the less important of the two knobs
    (the intra-op pool is what a 43x43 matmul actually thrashes). Failing to
    set it must not take the run down.
    """
    global _THREADS_PINNED
    if _THREADS_PINNED:
        return
    _THREADS_PINNED = True
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def data_config() -> dict[str, Any]:
    """The production Lorentz protocol, with no train_subset (GVLS sees the full pool)."""
    return {
        "protocol": "lorentz",
        "k_graph_cap": 8,
        "seed": SEED,
        "num_train": 10000,
        "num_val": 1250,
        "num_test": 1250,
        "min_particles": 10,
        "train_subset": None,
        "train_subset_seed": None,
        "train_subset_balanced": False,
    }


def base_cfg(k: int, beta: float, prior: str) -> dict[str, Any]:
    return {
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "k": k,
        "graph_method": GRAPH_METHOD,
        "prior": prior,
        "mp_rounds": MP_ROUNDS,
        "lr": LR,
        "beta": beta,
        "lambda_": LAMBDA_,
    }


@torch.no_grad()
def latent_graph_stats(model, jets, device) -> dict[str, float]:
    """How sparse is A_z really, per jet? (the 'is it always complete' check)"""
    model.eval()
    m = NUM_CLUSTERS
    off = ~torch.eye(m, dtype=torch.bool, device=device)
    max_edges = m * (m - 1) // 2
    counts = []
    for jet in jets:
        _mu, _lv, _z, a_z, _zt, _s, _rl = model(jet.x.to(device), jet.edge_index.to(device))
        counts.append(int((a_z[off] > 0).sum().item()) // 2)
    counts_arr = np.array(counts, dtype=float)
    return {
        "frac_jets_complete_a_z": float((counts_arr == max_edges).mean()),
        "min_num_latent_edges": float(counts_arr.min()),
        "max_num_latent_edges": float(counts_arr.max()),
    }


def loss_decomposition(model, jets, cfg, device) -> dict[str, float]:
    """Post-training share of each loss term, and the raw (unweighted) KL.

    Runs in train mode so the reparameterization sampling that the variational
    term actually regularizes is exercised, matching how the loss is computed
    during training rather than at deterministic eval time.
    """
    model.train()
    beta = float(cfg["beta"])
    totals = {"recon": 0.0, "kl": 0.0, "entropy": 0.0, "link": 0.0}
    raw_kl, sigma = 0.0, 0.0
    for jet in jets:
        x, edge_index = jet.x.to(device), jet.edge_index.to(device)
        adj = jet_adjacency(jet, device)
        pos_weight = jet_pos_weight(jet)
        mu, log_var, _z, a_z, _zt, s, recon_logits = model(x, edge_index)
        totals["recon"] += F.binary_cross_entropy_with_logits(
            recon_logits, adj, pos_weight=torch.tensor(pos_weight), reduction="mean"
        ).item()
        kl = (
            kl_isotropic(mu, log_var)
            if cfg["prior"] == "isotropic"
            else kl_graph_mrf(mu, log_var, a_z, float(cfg["lambda_"]))
        ).item()
        raw_kl += kl
        totals["kl"] += beta * kl
        totals["entropy"] += ENTROPY_WEIGHT * assignment_entropy(s).item()
        totals["link"] += AUX_LINK_WEIGHT * assignment_link_loss(s, adj, pos_weight).item()
        sigma += (0.5 * log_var).exp().mean().item()

    n = len(jets)
    # Shares are taken over absolute magnitudes: kl_graph_mrf is not
    # guaranteed non-negative here (its log-det term is detached), so a signed
    # denominator could make "share of total loss" meaningless or explode.
    denom = sum(abs(v) for v in totals.values()) or 1.0
    return {
        "loss_share_recon": abs(totals["recon"]) / denom,
        "loss_share_kl": abs(totals["kl"]) / denom,
        "loss_share_entropy": abs(totals["entropy"]) / denom,
        "loss_share_link": abs(totals["link"]) / denom,
        "raw_kl_per_node": raw_kl / n,
        "posterior_sigma_mean": sigma / n,
    }


def downstream_scores(model, split, device) -> dict[str, float]:
    """Classical-baseline accuracy on frozen features, over NUM_TRIALS balanced subsets."""
    test_features = extract_latent_features(model, split.test, device)
    per_model: dict[str, dict[str, list[float]]] = {
        "logreg": {"accuracy": [], "auc": []},
        "mlp": {"accuracy": [], "auc": []},
    }
    for trial in range(NUM_TRIALS):
        trial_split = subsample_train_balanced(split, TRAIN_SUBSET_PER_CLASS, seed=trial)
        train_features = extract_latent_features(model, trial_split.train, device)
        scores = evaluate_classical_baselines(train_features, test_features, seed=trial)
        for name in per_model:
            per_model[name]["accuracy"].append(scores[name]["accuracy"])
            per_model[name]["auc"].append(scores[name]["auc"])

    out: dict[str, float] = {}
    for name, metrics in per_model.items():
        out[f"{name}_accuracy_mean"] = float(np.mean(metrics["accuracy"]))
        out[f"{name}_accuracy_std"] = float(np.std(metrics["accuracy"]))
        out[f"{name}_auc_mean"] = float(np.mean(metrics["auc"]))
    return out


# Each worker loads the jet split once and caches it here. Loading is ~3s and
# the split is a few tens of MB, so a per-worker copy is cheaper and far more
# portable than relying on fork's copy-on-write (macOS spawns by default, and
# would not inherit a parent-loaded global at all).
_SPLIT_CACHE: Any = None


def _get_split():
    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        _pin_threads()
        _SPLIT_CACHE = load_split_from_config(data_config())
    return _SPLIT_CACHE


def run_config(point: tuple[int, float, str], epochs: int = EPOCHS) -> dict[str, Any]:
    """Train and score one grid point. Self-contained so it can run in any process."""
    k, beta, prior = point
    _pin_threads()
    split = _get_split()
    cfg = base_cfg(k, beta, prior)
    device = torch.device("cpu")

    start = time.time()
    model = train_pooled_gvls_on_jets(
        split.train,
        in_channels=NUM_FEATURES,
        latent_dim=LATENT_DIM,
        k=k,
        num_clusters=NUM_CLUSTERS,
        base_cfg=cfg,
        epochs=epochs,
        seed=SEED,
        device=device,
        batch_size=BATCH_SIZE,
        entropy_weight=ENTROPY_WEIGHT,
        aux_link_weight=AUX_LINK_WEIGHT,
        show_progress=False,
        eval_jets=split.val,
        eval_every=5,
        f1_negative_ratio=F1_NEGATIVE_RATIO,
    )
    train_time = time.time() - start

    row: dict[str, Any] = {"k": k, "beta": beta, "prior": prior}
    metrics = evaluate_pooled_gvls_on_jets(
        model,
        split.val,
        num_clusters=NUM_CLUSTERS,
        latent_dim=LATENT_DIM,
        k=k,
        num_features=NUM_FEATURES,
        f1_negative_ratio=F1_NEGATIVE_RATIO,
        seed=SEED,
        device=device,
    )
    for field in (
        "avg_reconstruction_f1",
        "avg_bits_per_edge",
        "avg_latent_density",
        "avg_num_latent_edges",
        "avg_edge_compression_ratio",
    ):
        row[field] = metrics[field]
    row.update(latent_graph_stats(model, split.val, device))
    row.update(loss_decomposition(model, split.val, cfg, device))
    row.update(downstream_scores(model, split, device))
    row["train_time_s"] = train_time
    return row


def _run_job(job: tuple[tuple[int, float, str], int]) -> dict[str, Any]:
    """Single-argument adapter so `run_config` can go through `imap_unordered`."""
    point, epochs = job
    return run_config(point, epochs)


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"k={row['k']} beta={row['beta']} prior={row['prior']:9s} | "
        f"f1={row['avg_reconstruction_f1']:.4f} "
        f"A_z={row['avg_num_latent_edges']:.2f}/6 "
        f"complete={row['frac_jets_complete_a_z']:.2f} "
        f"kl_share={row['loss_share_kl']:.4f} "
        f"logreg={row['logreg_accuracy_mean']:.4f}±{row['logreg_accuracy_std']:.4f} "
        f"mlp={row['mlp_accuracy_mean']:.4f} "
        f"({row['train_time_s']:.0f}s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel worker processes (0 = min(cpu_count, grid size)). "
        "Each worker is pinned to one thread, so this scales ~linearly.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="epochs per config")
    parser.add_argument("--out", type=Path, default=RESULT_PATH, help="output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _pin_threads()

    grid = [(k, b, p) for k in K_GRID for b in BETA_GRID for p in PRIOR_GRID]
    workers = args.workers or min(mp.cpu_count(), len(grid))
    workers = max(1, min(workers, len(grid)))
    print(
        f"{len(grid)} configs, {workers} parallel workers (1 thread each), "
        f"{args.epochs} epochs/config",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        def record(index: int, row: dict[str, Any]) -> None:
            writer.writerow(row)
            handle.flush()  # a killed run still leaves readable partial results
            print(f"[{index}/{len(grid)}] {_format_row(row)}", flush=True)

        if workers == 1:
            for i, point in enumerate(grid, start=1):
                record(i, run_config(point, args.epochs))
        else:
            # imap_unordered (not starmap/map) so each config's row is written
            # and printed the moment it finishes, rather than all of them after
            # the slowest worker returns -- this run is long enough that live
            # progress and a readable partial CSV both matter.
            # maxtasksperchild=1 keeps each config in a fresh process, so a
            # config that leaks or wedges cannot affect the ones after it.
            with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
                jobs = [(point, args.epochs) for point in grid]
                for i, row in enumerate(pool.imap_unordered(_run_job, jobs), start=1):
                    record(i, row)

    print(
        f"\nwrote {args.out} ({len(grid)} configs in {(time.time() - started) / 60:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
