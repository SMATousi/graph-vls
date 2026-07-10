from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor

ArrayLike = np.ndarray | Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reconstruction_f1(adj_true: ArrayLike, adj_logits: ArrayLike, threshold: float = 0.5) -> float:
    """Binary F1 between thresholded reconstruction probabilities and ground truth.

    Args:
        adj_true:   Binary ground-truth labels (0 or 1) for a set of node pairs, shape (N,).
        adj_logits: Raw (pre-sigmoid) logits for the same pairs, shape (N,).
        threshold:  Probability strictly above which a pair is predicted as an edge.

    Returns:
        F1 score as a float in [0, 1].
    """
    yt = _to_numpy(adj_true).ravel().astype(np.int64)
    yl = _to_numpy(adj_logits).ravel().astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-yl))
    y_pred = (probs > threshold).astype(np.int64)
    return float(f1_score(yt, y_pred, zero_division=0))


def dim_compression_ratio(latent_dim: int, num_features: int) -> float:
    """Ratio of latent dimension to input feature dimension (d / F).

    Smaller is more compressed (fewer numbers per node than the raw features).
    """
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")
    return float(latent_dim) / float(num_features)


def edge_compression_ratio(a_z: ArrayLike, num_input_edges: int) -> float:
    """Ratio of latent graph edge count to input graph edge count (|A_z| / |E|).

    Counts non-zero entries in the upper triangle (i < j) of A_z -- the number
    of undirected edges in the (already top-k sparsified, symmetrized) latent
    graph. Smaller is more compressed; values > 1 mean the latent graph is
    actually denser than the input graph (e.g. a link-prediction-optimized k
    rather than a compression-optimal one -- see specs/phase3/plan.md).
    """
    if num_input_edges <= 0:
        raise ValueError(f"num_input_edges must be positive, got {num_input_edges}")
    a = _to_numpy(a_z)
    n = a.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    num_latent_edges = int(np.count_nonzero(a[iu, ju]))
    return num_latent_edges / float(num_input_edges)


def node_compression_ratio(m_clusters: int, n_nodes: int) -> float:
    """Ratio of latent (pooled) node count to input node count (M / N).

    Smaller is more compressed (T3.6): the latent graph of distributions has
    genuinely fewer nodes than the input graph, not just a smaller per-node
    dimensionality. Reported as a separate ratio alongside dim_compression_ratio
    and edge_compression_ratio, per specs/phase3/plan.md Design Decision 1.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    return float(m_clusters) / float(n_nodes)


def assignment_storage_bits(n_nodes: int, m_clusters: int) -> float:
    """Storage cost of the hardened node-to-cluster assignment (T3.6).

    Reconstructing the full N-node graph from a pooled M-node latent graph
    requires the assignment matrix S (learned during pooling), which must be
    counted as part of the compressed representation's size -- otherwise the
    compression claim is misleading. Post-training, S is hardened via
    row-wise argmax to a single cluster index per node (the soft weights are
    only needed for differentiable training), so the cost is one
    ceil(log2(M))-bit index per input node.
    """
    if m_clusters <= 0:
        raise ValueError(f"m_clusters must be positive, got {m_clusters}")
    if m_clusters == 1:
        return 0.0
    return float(n_nodes) * math.ceil(math.log2(m_clusters))


def sample_node_pairs(n_nodes: int, num_samples: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Uniformly sample distinct node pairs (i < j) from the full pair space.

    Used to build an unbiased subset of all N(N-1)/2 pairs when N is too large
    to materialize a dense N x N adjacency (e.g. PubMed, N=19717 -> ~389M
    pairs). Feeding the result into `bits_per_edge` gives an estimate of the
    true average coding cost across the whole adjacency -- correctly
    dominated by non-edges for a sparse graph, since the sample matches the
    natural class distribution rather than a balanced pos/neg split.

    Args:
        n_nodes:     Number of nodes in the graph.
        num_samples: Number of distinct pairs to sample (clamped to the total
                      number of pairs available if larger).
        seed:        RNG seed for reproducibility.

    Returns:
        (row_idx, col_idx) LongTensors, each of length `min(num_samples, N*(N-1)/2)`.
    """
    max_pairs = n_nodes * (n_nodes - 1) // 2
    n_target = min(num_samples, max_pairs)
    rng = torch.Generator()
    rng.manual_seed(seed)
    seen: set[tuple[int, int]] = set()
    rows: list[int] = []
    cols: list[int] = []
    while len(rows) < n_target:
        batch_size = min((n_target - len(rows)) * 2, 200_000)
        u = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        v = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        for ui, vi in zip(u.tolist(), v.tolist()):
            if ui == vi:
                continue
            lo, hi = (ui, vi) if ui < vi else (vi, ui)
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            rows.append(lo)
            cols.append(hi)
            if len(rows) == n_target:
                break
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def eval_pairs_with_labels(
    n_nodes: int, pos_edge_index: Tensor, num_negatives: int, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Build (row, col, label) pairs for reconstruction-fidelity evaluation.

    Every real edge is included (label=1) -- compression fidelity means
    reconstructing the exact graph that was encoded, not a sampled subset of
    it (unlike `sample_node_pairs`, which is for the unbiased bits-per-edge
    estimate). Only the negative side is sampled, for tractability at
    PubMed's scale. Intended for `reconstruction_f1`.

    Args:
        n_nodes:        Number of nodes in the graph.
        pos_edge_index: Real edges, shape (2, E) (any orientation; self-loops
                         and duplicate/reverse pairs are collapsed internally).
        num_negatives:  Number of non-existing pairs to sample.
        seed:           RNG seed for the negative sample.

    Returns:
        (row_idx, col_idx, label) -- label is 1.0 for real edges, 0.0 for
        sampled negatives. Length is `num_positives + num_negatives`.
    """
    src = pos_edge_index[0].tolist()
    dst = pos_edge_index[1].tolist()
    pos_set: set[tuple[int, int]] = set()
    for u, v in zip(src, dst):
        lo, hi = (u, v) if u < v else (v, u)
        if lo != hi:
            pos_set.add((lo, hi))

    rng = torch.Generator()
    rng.manual_seed(seed)
    excluded = set(pos_set)
    neg: list[tuple[int, int]] = []
    while len(neg) < num_negatives:
        batch_size = min((num_negatives - len(neg)) * 4, 200_000)
        u = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        v = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        for ui, vi in zip(u.tolist(), v.tolist()):
            if ui == vi:
                continue
            lo, hi = (ui, vi) if ui < vi else (vi, ui)
            if (lo, hi) in excluded:
                continue
            excluded.add((lo, hi))
            neg.append((lo, hi))
            if len(neg) == num_negatives:
                break

    pos_list = sorted(pos_set)
    rows = [p[0] for p in pos_list] + [p[0] for p in neg]
    cols = [p[1] for p in pos_list] + [p[1] for p in neg]
    labels = [1.0] * len(pos_list) + [0.0] * len(neg)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )
