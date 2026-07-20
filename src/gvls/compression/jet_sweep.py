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

from typing import Any

import torch
from torch import Tensor

from gvls.data.jets import JetGraph
from gvls.losses.elbo import elbo
from gvls.models.encoder import GVLSEncoder
from gvls.models.latent_graph import LatentGraphLearner
from gvls.models.pooling import (
    LatentGraphPooling,
    PooledGVLS,
    assignment_entropy,
    assignment_link_loss,
)


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
    )
    loss = loss + entropy_weight * assignment_entropy(s)
    loss = loss + aux_link_weight * assignment_link_loss(s, adj_true, pos_weight)
    return loss


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
    """
    torch.manual_seed(seed)
    model = build_pooled_gvls(in_channels, latent_dim, k, num_clusters, base_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(base_cfg["lr"]))
    shuffle_generator = torch.Generator().manual_seed(seed)

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(jets), generator=shuffle_generator).tolist()
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            for idx in batch_idx:
                loss = jet_loss(
                    model, jets[idx], base_cfg, device, entropy_weight, aux_link_weight
                )
                (loss / len(batch_idx)).backward()
            optimizer.step()

    model.eval()
    return model
