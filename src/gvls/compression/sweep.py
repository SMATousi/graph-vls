from __future__ import annotations

import csv
import os
from typing import Any

import torch
from torch import Tensor

from gvls.eval.compression import (
    dim_compression_ratio,
    edge_compression_ratio,
    eval_pairs_with_labels,
    reconstruction_f1,
    sample_node_pairs,
)
from gvls.eval.metrics import bits_per_edge
from gvls.losses.elbo import elbo
from gvls.models.decoder import LatentGraphDecoder
from gvls.models.encoder import GVLSEncoder
from gvls.models.gvls import GVLS
from gvls.models.latent_graph import LatentGraphLearner

_DECODERS = {"inner_product", "graph_conditioned"}

RESULT_FIELDS = [
    "dataset",
    "latent_dim",
    "k",
    "num_nodes",
    "num_features",
    "num_input_edges",
    "num_latent_edges",
    "dim_compression_ratio",
    "edge_compression_ratio",
    "reconstruction_f1",
    "bits_per_edge",
    "latent_density",
]


def train_gvls_full_graph(
    x: Tensor,
    train_edge_index: Tensor,
    adj_true: Tensor,
    pos_weight: float,
    in_channels: int,
    latent_dim: int,
    k: int,
    base_cfg: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
    decoder: str = "inner_product",
) -> tuple[GVLS, LatentGraphDecoder | None]:
    """Train one GVLS model on the full graph (no held-out split).

    `base_cfg` supplies every hyperparameter except latent_dim and k (which
    the rate-distortion sweep varies): hidden_dim, mp_rounds, graph_method,
    prior, beta, lambda_, lr. It is a plain dict/DictConfig, not tied to any
    one dataset's Phase 2 NAS-best config.

    `decoder` (T3.4, revived 2026-07-13 -- see specs/phase3/plan.md T3.4 and
    validation.md V-3/V-4/V-8): `"inner_product"` (default, unchanged
    behavior) computes `recon_logits = z_tilde @ z_tilde.T`, where A_z only
    reaches the reconstruction indirectly through `mp_rounds` rounds of
    `LatentMessagePassing` inside `model` (zero for configs with
    `mp_rounds=0`, e.g. CiteSeer's and PubMed's Phase 2 NAS-best). `"graph_
    conditioned"` additionally routes `z_tilde` through a `LatentGraphDecoder`
    (`src/gvls/models/decoder.py`) -- one more, unconditional round of
    message passing over A_z -- so A_z has a guaranteed path into the
    reconstruction regardless of `mp_rounds`. Returns `(model, decoder)`;
    `decoder` is `None` for `"inner_product"` so callers can tell which path
    was used without inspecting the string again.
    """
    if decoder not in _DECODERS:
        raise ValueError(f"decoder must be one of {_DECODERS}, got '{decoder}'")

    torch.manual_seed(seed)
    encoder = GVLSEncoder(in_channels, int(base_cfg["hidden_dim"]), latent_dim)
    lgl = LatentGraphLearner(latent_dim, method=str(base_cfg["graph_method"]), k=k)
    model = GVLS(encoder, lgl, latent_dim=latent_dim, mp_rounds=int(base_cfg["mp_rounds"]))
    model = model.to(device)

    use_decoder = decoder == "graph_conditioned"
    decoder_module = LatentGraphDecoder(latent_dim).to(device) if use_decoder else None
    extra_params = list(decoder_module.parameters()) if decoder_module is not None else []
    optimizer = torch.optim.Adam(list(model.parameters()) + extra_params, lr=float(base_cfg["lr"]))

    for _epoch in range(1, epochs + 1):
        model.train()
        if decoder_module is not None:
            decoder_module.train()
        optimizer.zero_grad()
        mu, log_var, _z, a_z, z_tilde = model(x, train_edge_index)
        if decoder_module is not None:
            recon_logits = decoder_module(z_tilde, a_z)
        else:
            recon_logits = z_tilde @ z_tilde.T
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
        )
        loss.backward()
        optimizer.step()

    model.eval()
    if decoder_module is not None:
        decoder_module.eval()
    return model, decoder_module


def evaluate_compression(
    model: GVLS,
    x: Tensor,
    train_edge_index: Tensor,
    adj_true: Tensor,
    pos_edge_index: Tensor,
    n_nodes: int,
    num_features: int,
    num_input_edges: int,
    latent_dim: int,
    k: int,
    f1_negative_ratio: float,
    dense_pair_limit: int,
    bpe_sample_size: int,
    seed: int,
    device: torch.device,
    decoder_module: LatentGraphDecoder | None = None,
) -> dict[str, Any]:
    """Compute compression metrics for a trained model.

    Reconstruction F1 uses every real edge plus a sampled set of negatives
    (`eval_pairs_with_labels`) so the score isn't dominated by trivially-
    correct non-edges. Bits-per-edge uses the exact full upper triangle for
    graphs small enough to materialize densely, or an unbiased random sample
    (`sample_node_pairs`) above `dense_pair_limit` pairs (e.g. PubMed).

    `decoder_module`: the `LatentGraphDecoder` returned by
    `train_gvls_full_graph` when `decoder="graph_conditioned"` (T3.4) --
    must be the same instance used in training, not re-constructed, so its
    learned weight is applied. `None` (default) reproduces the original
    plain inner-product decode.

    Returns a flat dict with every key in RESULT_FIELDS except 'dataset'
    (added by the caller, which knows the dataset name).
    """
    with torch.no_grad():
        _, _, _, a_z, z_tilde = model(x, train_edge_index)
        if decoder_module is not None:
            scores = decoder_module(z_tilde, a_z)  # (N, N)
        else:
            scores = z_tilde @ z_tilde.T  # (N, N)

        num_negatives = max(1, round(num_input_edges * f1_negative_ratio))
        rows, cols, labels = eval_pairs_with_labels(
            n_nodes=n_nodes,
            pos_edge_index=pos_edge_index,
            num_negatives=num_negatives,
            seed=seed,
        )
        rows, cols, labels = rows.to(device), cols.to(device), labels.to(device)
        f1 = reconstruction_f1(labels, scores[rows, cols])

        max_pairs = n_nodes * (n_nodes - 1) // 2
        if max_pairs <= dense_pair_limit:
            iu, ju = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
        else:
            iu, ju = sample_node_pairs(n_nodes, bpe_sample_size, seed=seed)
            iu, ju = iu.to(device), ju.to(device)
        bpe = bits_per_edge(adj_true[iu, ju], scores[iu, ju])

        density = (a_z > 0).float().mean().item()
        num_latent_edges = int(torch.triu(a_z, diagonal=1).ne(0).sum().item())

    return {
        "latent_dim": latent_dim,
        "k": k,
        "num_nodes": n_nodes,
        "num_features": num_features,
        "num_input_edges": num_input_edges,
        "num_latent_edges": num_latent_edges,
        "dim_compression_ratio": dim_compression_ratio(latent_dim, num_features),
        "edge_compression_ratio": edge_compression_ratio(a_z, num_input_edges),
        "reconstruction_f1": f1,
        "bits_per_edge": bpe,
        "latent_density": density,
    }


def write_results_csv(
    rows: list[dict[str, Any]], path: str, fieldnames: list[str] = RESULT_FIELDS
) -> None:
    """Write one row per grid point to a CSV at `path`, creating parent dirs.

    `fieldnames` defaults to T3.3's RESULT_FIELDS but is overridable so other
    sweeps (e.g. T3.6's node-count pooling sweep, gvls.compression.pooling_sweep)
    can reuse this writer with their own schema instead of duplicating it.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_compression_optimal(
    rows: list[dict[str, Any]], fidelity_floor: float
) -> tuple[dict[str, Any], bool]:
    """Pick the smallest (d, k) meeting the fidelity floor.

    Score = dim_compression_ratio + edge_compression_ratio (smaller is more
    compressed); ties broken by smallest latent_dim. If no grid point meets
    the floor, falls back to the highest-F1 point and reports floor_met=False
    so the caller can surface a clear warning instead of silently pretending
    the floor was met.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    candidates = [r for r in rows if r["reconstruction_f1"] >= fidelity_floor]
    if candidates:
        best = min(
            candidates,
            key=lambda r: (
                r["dim_compression_ratio"] + r["edge_compression_ratio"],
                r["latent_dim"],
            ),
        )
        return best, True

    best = max(rows, key=lambda r: r["reconstruction_f1"])
    return best, False
