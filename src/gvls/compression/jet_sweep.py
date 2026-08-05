"""Per-jet inductive GVLS training (T4.2) and jet-scale compression sweep (T4.3).

T4.2's core risk isn't new model code -- `PooledGVLS` (T3.6) is reused
completely unmodified -- it's whether a *per-jet* forward-loop with gradient
accumulation is correct: gradients must reach every submodule, and one jet's
computation must never leak into another's (see plan.md Design Decision 7).
Since each jet is pooled/reconstructed via its own independent `model(x,
edge_index)` call -- there is no batched tensor spanning multiple jets
anywhere in this loop -- leakage is structurally impossible, not just
empirically rare; `tests/test_jet_sweep.py` checks this directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

from gvls.data.jets import JetGraph
from gvls.eval.compression import (
    dim_compression_ratio,
    edge_compression_ratio,
    eval_pairs_with_labels,
    node_compression_ratio,
    reconstruction_f1,
)
from gvls.eval.metrics import bits_per_edge
from gvls.losses.elbo import elbo
from gvls.models.encoder import GVLSEncoder
from gvls.models.latent_graph import LatentGraphLearner
from gvls.models.pooling import (
    LatentGraphPooling,
    PooledGVLS,
    assignment_entropy,
    assignment_link_loss,
)

# Same schema convention as results/compression/{dataset}_pooling.csv (T3.6),
# adapted for jets: M/k/latent_dim/num_features are fixed per grid point (M is
# what's swept, the rest come from a jet-scale starting config, not re-tuned
# via NAS -- plan.md T4.3), everything else is a per-jet quantity averaged
# over the held-out evaluation jets (jets vary in N, so num_nodes,
# num_input_edges, and their derived ratios are jet-specific, unlike the
# citation-network sweeps where N was fixed for the whole dataset).
JET_RESULT_FIELDS = [
    "dataset",
    "num_clusters",
    "latent_dim",
    "k",
    "num_features",
    "num_eval_jets",
    "avg_num_nodes",
    "avg_num_input_edges",
    "avg_num_latent_edges",
    "dim_compression_ratio",
    "avg_edge_compression_ratio",
    "avg_node_compression_ratio",
    "avg_reconstruction_f1",
    "avg_bits_per_edge",
    "avg_latent_density",
]


def jet_adjacency(jet: JetGraph, device: torch.device) -> Tensor:
    """Dense (N, N) adjacency for one jet's k-NN graph."""
    n = int(jet.num_nodes)
    adj = torch.zeros(n, n, device=device)
    edge_index = jet.edge_index
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def jet_pos_weight(jet: JetGraph) -> float:
    """BCE pos_weight for one jet: (N^2 - E) / E, computed from its own (N, E)."""
    n = int(jet.num_nodes)
    e = jet.edge_index.size(1)
    if e == 0:
        return 1.0
    return float((n * n - e) / e)


def build_pooled_gvls(
    in_channels: int,
    latent_dim: int,
    k: int,
    num_clusters: int,
    base_cfg: dict[str, Any],
) -> PooledGVLS:
    encoder = GVLSEncoder(in_channels, int(base_cfg["hidden_dim"]), latent_dim)
    pooling = LatentGraphPooling(latent_dim, num_clusters)
    lgl = LatentGraphLearner(latent_dim, method=str(base_cfg["graph_method"]), k=k)
    return PooledGVLS(
        encoder, pooling, lgl, latent_dim=latent_dim, mp_rounds=int(base_cfg["mp_rounds"])
    )


def jet_loss(
    model: PooledGVLS,
    jet: JetGraph,
    base_cfg: dict[str, Any],
    device: torch.device,
    entropy_weight: float,
    aux_link_weight: float,
) -> Tensor:
    """Forward pass + full per-jet loss (ELBO + T3.6's two DiffPool auxiliary terms).

    A single self-contained call: every tensor it touches (x, edge_index,
    adj_true, pos_weight) is derived from this jet alone, so nothing here can
    read another jet's state.
    """
    x = jet.x.to(device)
    edge_index = jet.edge_index.to(device)
    adj_true = jet_adjacency(jet, device)
    pos_weight = jet_pos_weight(jet)

    mu, log_var, _z, a_z, _z_tilde, s, recon_logits = model(x, edge_index)
    loss = elbo(
        recon_logits,
        adj_true,
        mu,
        log_var,
        a_z,
        beta=float(base_cfg["beta"]),
        lambda_=float(base_cfg["lambda_"]),
        prior=str(base_cfg["prior"]),
        pos_weight=pos_weight,
        # T5.1: defaults to "legacy" so configs written before Phase 5 keep
        # their exact behaviour; configs/train/jet_pretrain.yaml opts into
        # "per_jet". See gvls.losses.elbo.elbo's docstring.
        normalization=str(base_cfg.get("normalization", "legacy")),
    )
    loss = loss + entropy_weight * assignment_entropy(s)
    loss = loss + aux_link_weight * assignment_link_loss(s, adj_true, pos_weight)
    return loss


# Structural/config fields in evaluate_pooled_gvls_on_jets's return dict that
# don't change epoch to epoch (already present in the run's W&B config from
# wandb.init) -- excluded from per-epoch logging so val_* only reports
# quantities that actually evolve during training.
_STATIC_EVAL_KEYS = {"num_clusters", "latent_dim", "k", "num_features", "dim_compression_ratio"}

# T5.1 (specs/phase5/): selectable checkpoint criteria for
# train_pooled_gvls_on_jets. Maps each name to whether larger is better, so
# the selection loop itself stays metric-agnostic.
#
# Why this exists: the pre-T5.1 behaviour (always "reconstruction_f1") was
# selecting on a signal measured to correlate only +0.245 with downstream
# classification accuracy across the usable beta range -- the +0.922
# correlation over the full (k, beta, prior) grid turned out to be an
# artifact of the collapsed high-beta rows, and the single best-F1
# configuration in that grid was *not* the best-downstream one. See
# specs/phase5/validation.md V-0 Part B.
SELECTION_METRICS = {
    "reconstruction_f1": True,  # pre-T5.1 behaviour, still the default
    "val_loss": False,
    "probe_accuracy": True,
}


@torch.no_grad()
def jet_probe_features(
    model: PooledGVLS, jets: list[JetGraph], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Frozen `(z_tilde, A_z)` per jet, flattened to one feature vector each.

    Deliberately a small local implementation rather than an import of
    `gvls.qgnn_training.extract_latent_features` /
    `gvls.eval.classical_baseline.jet_features_to_array`: both of those
    modules import `gvls.models.qgnn`, which imports qiskit. Pulling the
    entire quantum stack into GVLS's own classical pretraining path just to
    score a logistic regression would be a heavy and surprising dependency,
    and would make GVLS pretraining fail on a machine without qiskit
    installed. The feature layout matches `jet_features_to_array`'s
    (flattened `z_tilde`, then `A_z`'s strict upper triangle).
    """
    model.eval()
    rows: list[Tensor] = []
    labels: list[float] = []
    for jet in jets:
        _mu, _lv, _z, a_z, z_tilde, _s, _rl = model(jet.x.to(device), jet.edge_index.to(device))
        m = a_z.size(0)
        iu, ju = torch.triu_indices(m, m, offset=1, device=a_z.device)
        rows.append(torch.cat([z_tilde.reshape(-1), a_z[iu, ju]]).cpu())
        labels.append(float(jet.y.item()))
    return torch.stack(rows), torch.tensor(labels)


def probe_accuracy(
    model: PooledGVLS, jets: list[JetGraph], device: torch.device, seed: int
) -> float:
    """Downstream separability of the frozen features, as a selection signal.

    Fits a logistic regression on half of `jets` and scores it on the other
    half, both drawn from the *validation* split -- the test split is never
    touched here. This is a cheap proxy for the metric Phase 5 actually
    optimizes (`plan.md` Design Decision 2), not a reported result; the
    reported downstream numbers still come from
    `gvls.eval.classical_baseline` under the full 5-trial protocol.

    Returns 0.5 (chance) if either half ends up single-class, rather than
    letting sklearn raise mid-training.
    """
    from sklearn.linear_model import LogisticRegression  # local: keeps import cost off callers

    x, y = jet_probe_features(model, jets, device)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(jets), generator=generator)
    half = len(jets) // 2
    tr, te = perm[:half], perm[half:]
    y_tr, y_te = y[tr].numpy(), y[te].numpy()
    if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
        return 0.5
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x[tr].numpy(), y_tr)
    return float(clf.score(x[te].numpy(), y_te))


@torch.no_grad()
def validation_loss(
    model: PooledGVLS,
    jets: list[JetGraph],
    base_cfg: dict[str, Any],
    device: torch.device,
    entropy_weight: float,
    aux_link_weight: float,
) -> float:
    """Mean per-jet training loss over held-out jets (lower is better).

    Uses the same `jet_loss` the training loop optimizes, so the selection
    criterion and the objective agree. Runs in eval mode, so the pooled
    Gaussian is taken at its mean rather than resampled -- this is a
    lower-variance estimate of the same quantity, not a different one.
    """
    model.eval()
    total = 0.0
    for jet in jets:
        total += jet_loss(model, jet, base_cfg, device, entropy_weight, aux_link_weight).item()
    return total / max(len(jets), 1)


def train_pooled_gvls_on_jets(
    jets: list[JetGraph],
    in_channels: int,
    latent_dim: int,
    k: int,
    num_clusters: int,
    base_cfg: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    entropy_weight: float = 0.1,
    aux_link_weight: float = 5.0,
    show_progress: bool = True,
    progress_desc: str = "pretrain GVLS",
    eval_jets: list[JetGraph] | None = None,
    eval_every: int = 1,
    f1_negative_ratio: float = 1.0,
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
    selection_metric: str = "reconstruction_f1",
) -> PooledGVLS:
    """Train one PooledGVLS unsupervised (ELBO only) over many jets (T4.2/T4.3).

    Jets vary in particle count N, so they're iterated one at a time through
    `model(x, edge_index)` -- reusing PooledGVLS unmodified (plan.md Design
    Decision 3) -- rather than batched into one tensor. Per-jet losses within
    a minibatch are averaged and their gradients accumulated (each jet's
    `.backward()` call adds into the shared parameters' `.grad`, standard
    gradient accumulation) before a single `optimizer.step()`, mirroring
    `train_pooled_gvls_full_graph`'s (T3.6) hyperparameter conventions and
    auxiliary losses, adapted from one large graph to many small ones.

    `show_progress` reports one tqdm tick per epoch, postfixed with the
    running mean per-jet loss for that epoch (and validation F1 once
    computed) -- real pretraining runs over thousands of jets take minutes,
    and this is meant to be run unattended on a remote machine (see
    `scripts/run_pretrain_gvls_jets_final.sh`), so visible progress matters.
    Set to `False` in tests/tight inner loops.

    `eval_jets`, if given, is passed to `evaluate_pooled_gvls_on_jets` every
    `eval_every` epochs (and always on the final epoch) to compute held-out
    compression metrics (F1, bits-per-edge, latent density, etc.) as training
    progresses -- otherwise the only way to see how compression fidelity
    evolves is to train fully and evaluate once at the end. `on_epoch_end`,
    if given, is called once per epoch as `on_epoch_end(epoch, metrics)` with
    `metrics = {"epoch": ..., "train_loss": ..., **(val_* keys, if
    eval_jets was given and this was an eval epoch)}` -- the natural hook for
    live per-epoch logging (e.g. `wandb.log`) without coupling this reusable
    training function to any specific logging backend.

    `selection_metric` (T5.1, specs/phase5/) chooses which validation signal
    the best-epoch snapshot is selected on: `"reconstruction_f1"` (default,
    the pre-T5.1 behaviour), `"val_loss"` (mean held-out `jet_loss`), or
    `"probe_accuracy"` (a logistic regression on the frozen features, fit and
    scored within the validation split). The default is kept for backward
    compatibility, but it selects on a signal measured to correlate only
    +0.245 with downstream accuracy in the usable beta range -- see
    `SELECTION_METRICS` above and specs/phase5/validation.md V-0 Part B.
    Whichever criterion a run uses is reported in the returned model's
    per-epoch metrics (`val_loss` / `val_probe_accuracy` keys) so it is
    recoverable from the run's own logs.

    Whenever `eval_jets` is given, the returned model's weights are the
    best-validation snapshot (by `selection_metric`) seen across eval epochs,
    not necessarily the final epoch's -- mirrors `train_qgnn_classifier`'s
    best-val-accuracy checkpointing (`src/gvls/qgnn_training.py`), which this
    function lacked until now (specs/phase4/validation.md V-10: GVLS
    pretraining was returning the last-epoch model unconditionally, even
    when a later epoch had drifted to worse validation fidelity than an
    earlier one). If `eval_jets` is `None`, no validation signal exists to
    select a "best" epoch from, so behavior is unchanged: the last-epoch
    model is returned. The return type stays `PooledGVLS` either way (the
    best weights are loaded into `model` in place before returning) so every
    existing caller works unmodified.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {set(SELECTION_METRICS)}, got '{selection_metric}'"
        )
    higher_is_better = SELECTION_METRICS[selection_metric]

    torch.manual_seed(seed)
    model = build_pooled_gvls(in_channels, latent_dim, k, num_clusters, base_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(base_cfg["lr"]))
    shuffle_generator = torch.Generator().manual_seed(seed)

    best_score = float("-inf") if higher_is_better else float("inf")
    best_epoch = -1
    best_state_dict: dict[str, Tensor] | None = None

    epoch_iter = tqdm(range(epochs), desc=progress_desc, disable=not show_progress)
    for epoch in epoch_iter:
        model.train()
        perm = torch.randperm(len(jets), generator=shuffle_generator).tolist()
        running_loss, n_seen = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            for idx in batch_idx:
                loss = jet_loss(
                    model, jets[idx], base_cfg, device, entropy_weight, aux_link_weight
                )
                (loss / len(batch_idx)).backward()
                batch_loss += loss.item()
            optimizer.step()
            running_loss += batch_loss
            n_seen += len(batch_idx)
        train_loss = running_loss / max(n_seen, 1)

        epoch_metrics: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        postfix: dict[str, Any] = {"loss": train_loss}
        is_eval_epoch = eval_jets is not None and (epoch % eval_every == 0 or epoch == epochs - 1)
        if is_eval_epoch:
            eval_metrics = evaluate_pooled_gvls_on_jets(
                model,
                eval_jets,  # type: ignore[arg-type]
                num_clusters=num_clusters,
                latent_dim=latent_dim,
                k=k,
                num_features=in_channels,
                f1_negative_ratio=f1_negative_ratio,
                seed=seed,
                device=device,
            )
            epoch_metrics.update(
                {
                    f"val_{key}": value
                    for key, value in eval_metrics.items()
                    if key not in _STATIC_EVAL_KEYS
                }
            )
            val_f1 = eval_metrics["avg_reconstruction_f1"]
            postfix["val_f1"] = val_f1

            if selection_metric == "reconstruction_f1":
                score = val_f1
            elif selection_metric == "val_loss":
                score = validation_loss(
                    model,
                    eval_jets,  # type: ignore[arg-type]
                    base_cfg,
                    device,
                    entropy_weight,
                    aux_link_weight,
                )
                epoch_metrics["val_loss"] = score
                postfix["val_loss"] = score
            else:  # probe_accuracy
                score = probe_accuracy(
                    model,
                    eval_jets,  # type: ignore[arg-type]
                    device,
                    seed,
                )
                epoch_metrics["val_probe_accuracy"] = score
                postfix["probe"] = score

            improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                best_state_dict = {name: t.clone() for name, t in model.state_dict().items()}
        epoch_iter.set_postfix(**postfix)

        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_metrics)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if show_progress:
            tqdm.write(
                f"{progress_desc}: restoring best-validation checkpoint "
                f"(epoch={best_epoch}, {selection_metric}={best_score:.4f})"
            )
    model.eval()
    return model


def save_gvls_checkpoint(
    model: PooledGVLS,
    config: dict[str, Any],
    path: str,
) -> None:
    """Persist a trained PooledGVLS's weights plus enough config to rebuild
    its architecture (`build_pooled_gvls`) before loading them back -- unlike
    the citation-network convention (`checkpoints/best.pt`, a bare
    `state_dict()` for one fixed, always-identical architecture), jets vary
    `M`/`latent_dim`/`k` across runs, so the config must travel with the
    weights (T4.5 needs this to freeze and reuse T4.3's compression-optimal
    model, which T4.3 itself never persisted).
    """
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_gvls_checkpoint(path: str, device: torch.device) -> tuple[PooledGVLS, dict[str, Any]]:
    """Inverse of `save_gvls_checkpoint`: rebuilds the architecture from the
    saved config, loads weights, and returns `(model.eval(), config)`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_pooled_gvls(
        in_channels=int(config["in_channels"]),
        latent_dim=int(config["latent_dim"]),
        k=int(config["k"]),
        num_clusters=int(config["num_clusters"]),
        base_cfg=config["base_cfg"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def evaluate_pooled_gvls_on_jets(
    model: PooledGVLS,
    jets: list[JetGraph],
    num_clusters: int,
    latent_dim: int,
    k: int,
    num_features: int,
    f1_negative_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Average per-jet compression metrics over a held-out set of jets (T4.3).

    Mirrors `gvls.compression.pooling_sweep.evaluate_pooled_compression`
    (T3.6), but jets have no single shared N: `A_z`'s topology is itself a
    function of each jet's own pooled representation (not fixed across
    jets, unlike a citation-network model's single A_z), so
    edge_compression_ratio/node_compression_ratio/latent_density are all
    per-jet quantities here, averaged rather than reported once.

    Every pair in a jet's own dense N x N grid is used directly (jets are
    small -- at most ~139 particles in `qg_jets` -- so unlike PubMed there is
    no need for `sample_node_pairs`'s large-graph fallback); only
    reconstruction_f1's negative side is sampled, for consistency with the
    citation-network convention, clamped to the number of non-edges actually
    available. Jets with fewer than 2 particles (no possible pairs), zero
    edges (F1/edge-ratio undefined), or a *complete* graph (zero non-edges --
    happens whenever a jet's particle count is small enough that
    `k_graph_cap >= n - 1`, plan.md Design Decision 5, so every pair is
    already an edge) are skipped from the relevant running average rather
    than crashing or, for the complete-graph case, spinning forever in
    `eval_pairs_with_labels`'s negative-sampling loop, which cannot terminate
    if asked for negatives that don't exist.
    """
    f1s: list[float] = []
    bpes: list[float] = []
    densities: list[float] = []
    edge_ratios: list[float] = []
    node_ratios: list[float] = []
    n_nodes_list: list[int] = []
    n_input_edges_list: list[int] = []
    n_latent_edges_list: list[int] = []

    model.eval()
    with torch.no_grad():
        for i, jet in enumerate(jets):
            n = int(jet.num_nodes)
            if n < 2:
                continue

            x = jet.x.to(device)
            edge_index = jet.edge_index.to(device)
            adj_true = jet_adjacency(jet, device)
            num_input_edges = int(edge_index.size(1) // 2)

            _, _, _, a_z, _z_tilde, _s, recon_logits = model(x, edge_index)

            density = (a_z > 0).float().mean().item()
            num_latent_edges = int(torch.triu(a_z, diagonal=1).ne(0).sum().item())
            node_ratio = node_compression_ratio(num_clusters, n)

            n_nodes_list.append(n)
            n_input_edges_list.append(num_input_edges)
            n_latent_edges_list.append(num_latent_edges)
            densities.append(density)
            node_ratios.append(node_ratio)

            iu, ju = torch.triu_indices(n, n, offset=1, device=device)
            bpes.append(bits_per_edge(adj_true[iu, ju], recon_logits[iu, ju]))

            if num_input_edges == 0:
                continue  # no positive edges: F1/edge_compression_ratio undefined
            edge_ratios.append(edge_compression_ratio(a_z, num_input_edges))

            # A small jet with k_graph_cap >= n-1 (Design Decision 5) is a
            # *complete* graph -- zero non-edges exist, so eval_pairs_with_
            # labels' negative-sampling loop would spin forever if asked for
            # any negatives at all. Clamp to what's actually available and
            # skip F1 for this jet (undefined with zero negatives) rather
            # than hanging.
            max_possible_negatives = n * (n - 1) // 2 - num_input_edges
            if max_possible_negatives == 0:
                continue
            num_negatives = min(
                max_possible_negatives, max(1, round(num_input_edges * f1_negative_ratio))
            )
            pos_edge_index = edge_index[:, edge_index[0] < edge_index[1]].cpu()
            rows, cols, labels = eval_pairs_with_labels(
                n_nodes=n, pos_edge_index=pos_edge_index, num_negatives=num_negatives, seed=seed + i
            )
            rows, cols, labels = rows.to(device), cols.to(device), labels.to(device)
            f1s.append(reconstruction_f1(labels, recon_logits[rows, cols]))

    if not n_nodes_list:
        raise ValueError("no evaluable jets (all had fewer than 2 particles)")

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "num_clusters": num_clusters,
        "latent_dim": latent_dim,
        "k": k,
        "num_features": num_features,
        "num_eval_jets": len(n_nodes_list),
        "avg_num_nodes": _mean(n_nodes_list),
        "avg_num_input_edges": _mean(n_input_edges_list),
        "avg_num_latent_edges": _mean(n_latent_edges_list),
        "dim_compression_ratio": dim_compression_ratio(latent_dim, num_features),
        "avg_edge_compression_ratio": _mean(edge_ratios) if edge_ratios else float("nan"),
        "avg_node_compression_ratio": _mean(node_ratios),
        "avg_reconstruction_f1": _mean(f1s) if f1s else float("nan"),
        "avg_bits_per_edge": _mean(bpes),
        "avg_latent_density": _mean(densities),
    }


def select_compression_optimal_m(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    """Pick the smallest M whose avg F1 is within `tolerance` of the best M's F1.

    Unlike T3.3's `select_compression_optimal` (a fixed 0.90 fidelity floor,
    calibrated from citation-network experience), jets have no such precedent
    yet (plan.md T4.3), so the criterion here is relative: the smallest `M`
    that gives up at most `tolerance` average F1 relative to the largest `M`
    tested, rather than an absolute floor.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    best_f1 = max(r["avg_reconstruction_f1"] for r in rows)
    candidates = [r for r in rows if best_f1 - r["avg_reconstruction_f1"] <= tolerance]
    return min(candidates, key=lambda r: r["num_clusters"])


def run_jet_compression_sweep(
    train_jets: list[JetGraph],
    eval_jets: list[JetGraph],
    m_grid: list[int],
    in_channels: int,
    latent_dim: int,
    k: int,
    base_cfg: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    entropy_weight: float = 0.1,
    aux_link_weight: float = 5.0,
    f1_negative_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Train + evaluate one PooledGVLS per M in `m_grid` (T4.3's rate-distortion sweep).

    `k` (the latent-graph learner's sparsification parameter) is clamped to
    `min(k, m - 1)` at each grid point, since a graph with M nodes can't have
    a node degree of M or more -- the same clamp `tests/test_pooling.py`'s
    `make_model` helper already uses for the citation-network sweeps.
    """
    rows: list[dict[str, Any]] = []
    for m in tqdm(m_grid, desc="M grid"):
        k_m = min(k, m - 1)
        model = train_pooled_gvls_on_jets(
            train_jets,
            in_channels=in_channels,
            latent_dim=latent_dim,
            k=k_m,
            num_clusters=m,
            base_cfg=base_cfg,
            epochs=epochs,
            seed=seed,
            device=device,
            batch_size=batch_size,
            entropy_weight=entropy_weight,
            aux_link_weight=aux_link_weight,
            progress_desc=f"pretrain GVLS (M={m})",
        )
        metrics = evaluate_pooled_gvls_on_jets(
            model,
            eval_jets,
            num_clusters=m,
            latent_dim=latent_dim,
            k=k_m,
            num_features=in_channels,
            f1_negative_ratio=f1_negative_ratio,
            seed=seed,
            device=device,
        )
        rows.append({"dataset": "qg_jets", **metrics})
    return rows
