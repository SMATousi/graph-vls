from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gvls.eval.compression import (
    assignment_storage_bits,
    dim_compression_ratio,
    edge_compression_ratio,
    eval_pairs_with_labels,
    node_compression_ratio,
    reconstruction_f1,
    sample_node_pairs,
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

RESULT_FIELDS = [
    "dataset",
    "pool_ratio",
    "latent_dim",
    "k",
    "num_nodes",
    "num_clusters",
    "num_features",
    "num_input_edges",
    "num_latent_edges",
    "dim_compression_ratio",
    "edge_compression_ratio",
    "node_compression_ratio",
    "assignment_storage_bits",
    "reconstruction_f1",
    "bits_per_edge",
    "latent_density",
]


def train_pooled_gvls_full_graph(
    x: Tensor,
    train_edge_index: Tensor,
    adj_true: Tensor,
    pos_weight: float,
    in_channels: int,
    latent_dim: int,
    k: int,
    num_clusters: int,
    base_cfg: dict[str, Any],
    epochs: int,
    seed: int,
    device: torch.device,
    entropy_weight: float = 0.1,
    aux_link_weight: float = 5.0,
) -> PooledGVLS:
    """Train one PooledGVLS model on the full graph (no held-out split).

    `base_cfg` supplies every hyperparameter except `num_clusters` (the axis
    the node-count pooling sweep, T3.6, varies): hidden_dim, mp_rounds,
    graph_method, prior, beta, lambda_, lr, plus latent_dim and k -- all held
    fixed at the dataset's T3.3 compression-optimal config
    (configs/compression/{dataset}.yaml), not re-tuned here.

    Two auxiliary loss terms (both standard DiffPool components, Ying et al.
    2018) are needed to avoid a cold-start collapse diagnosed in the first
    T3.6 sweep, where every grid point converged to a trivial
    always-predict-edge classifier (F1 stuck at exactly 2/3 regardless of
    pool_ratio) -- see specs/phase3/validation.md V-7:
      - `entropy_weight * assignment_entropy(S)`: encourages each node's
        assignment to specialize rather than stay diffuse.
      - `aux_link_weight * assignment_link_loss(S, adj_true, pos_weight)`:
        gives S a *direct* gradient signal from the real input graph, since
        the reconstruction/KL losses' gradient to S has to travel through
        the entire pooled-graph pipeline and vanishes when S starts
        near-uniform (entropy regularization alone was not sufficient to
        escape this -- the aux loss is what breaks the deadlock).
    """
    torch.manual_seed(seed)
    encoder = GVLSEncoder(in_channels, int(base_cfg["hidden_dim"]), latent_dim)
    pooling = LatentGraphPooling(latent_dim, num_clusters)
    lgl = LatentGraphLearner(latent_dim, method=str(base_cfg["graph_method"]), k=k)
    model = PooledGVLS(
        encoder, pooling, lgl, latent_dim=latent_dim, mp_rounds=int(base_cfg["mp_rounds"])
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(base_cfg["lr"]))

    for _epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        mu, log_var, _z, a_z, _z_tilde, s, recon_logits = model(x, train_edge_index)
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
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate_pooled_compression(
    model: PooledGVLS,
    x: Tensor,
    train_edge_index: Tensor,
    adj_true: Tensor,
    pos_edge_index: Tensor,
    n_nodes: int,
    num_clusters: int,
    num_features: int,
    num_input_edges: int,
    latent_dim: int,
    k: int,
    pool_ratio: float,
    f1_negative_ratio: float,
    dense_pair_limit: int,
    bpe_sample_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compute compression metrics for a trained PooledGVLS model.

    Mirrors gvls.compression.sweep.evaluate_compression (T3.3), with two
    differences: reconstruction_f1/bits_per_edge are computed on the
    *unpooled* N x N `recon_logits` returned by PooledGVLS (not a plain
    z_tilde @ z_tilde.T), and two new metrics are added --
    node_compression_ratio and assignment_storage_bits -- to characterise the
    node-count axis T3.6 introduces.

    Returns a flat dict with every key in RESULT_FIELDS except 'dataset'
    (added by the caller, which knows the dataset name).
    """
    with torch.no_grad():
        _, _, _, a_z, _z_tilde, _s, recon_logits = model(x, train_edge_index)

        num_negatives = max(1, round(num_input_edges * f1_negative_ratio))
        rows, cols, labels = eval_pairs_with_labels(
            n_nodes=n_nodes,
            pos_edge_index=pos_edge_index,
            num_negatives=num_negatives,
            seed=seed,
        )
        rows, cols, labels = rows.to(device), cols.to(device), labels.to(device)
        f1 = reconstruction_f1(labels, recon_logits[rows, cols])

        max_pairs = n_nodes * (n_nodes - 1) // 2
        if max_pairs <= dense_pair_limit:
            iu, ju = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
        else:
            iu, ju = sample_node_pairs(n_nodes, bpe_sample_size, seed=seed)
            iu, ju = iu.to(device), ju.to(device)
        bpe = bits_per_edge(adj_true[iu, ju], recon_logits[iu, ju])

        density = (a_z > 0).float().mean().item()
        num_latent_edges = int(torch.triu(a_z, diagonal=1).ne(0).sum().item())

    return {
        "pool_ratio": pool_ratio,
        "latent_dim": latent_dim,
        "k": k,
        "num_nodes": n_nodes,
        "num_clusters": num_clusters,
        "num_features": num_features,
        "num_input_edges": num_input_edges,
        "num_latent_edges": num_latent_edges,
        "dim_compression_ratio": dim_compression_ratio(latent_dim, num_features),
        "edge_compression_ratio": edge_compression_ratio(a_z, num_input_edges),
        "node_compression_ratio": node_compression_ratio(num_clusters, n_nodes),
        "assignment_storage_bits": assignment_storage_bits(n_nodes, num_clusters),
        "reconstruction_f1": f1,
        "bits_per_edge": bpe,
        "latent_density": density,
    }
