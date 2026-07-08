from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


@dataclass
class EdgeSplit:
    """Holds train/val/test edge sets for link prediction evaluation.

    train_edge_index: both directions (i→j and j→i) for message passing.
    val_pos / test_pos: canonical undirected edges (i < j) held out from training.
    val_neg / test_neg: non-existing edges sampled at 1:1 ratio with positives.
    """

    train_edge_index: Tensor  # (2, 2 * E_train)
    val_pos: Tensor           # (2, E_val)
    val_neg: Tensor           # (2, E_val)
    test_pos: Tensor          # (2, E_test)
    test_neg: Tensor          # (2, E_test)
    n_nodes: int


def split_edges(data: Data, train_ratio: float, seed: int = 42) -> EdgeSplit:
    """Deterministically split graph edges into train / val / test sets.

    Args:
        data: PyG Data object with edge_index and num_nodes.
        train_ratio: fraction of edges kept for training (0 < ratio < 1).
        seed: RNG seed for reproducibility.

    Returns:
        EdgeSplit with train edges (both directions) and pos/neg val & test pairs.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    n_nodes: int = int(data.num_nodes)
    edge_index, _ = remove_self_loops(data.edge_index)

    # Canonical undirected edges: keep only (i, j) where i < j.
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    canon = edge_index[:, mask]  # (2, E)

    n_edges = canon.size(1)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_edges, generator=rng)
    canon = canon[:, perm]

    n_train = int(n_edges * train_ratio)
    n_remaining = n_edges - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    train_edges = canon[:, :n_train]
    val_pos = canon[:, n_train : n_train + n_val]
    test_pos = canon[:, n_train + n_val :]

    assert test_pos.size(1) == n_test

    # Both directions for message passing.
    train_edge_index = torch.cat([train_edges, train_edges.flip(0)], dim=1)

    pos_set = {(int(u), int(v)) for u, v in zip(canon[0].tolist(), canon[1].tolist())}
    val_neg = _sample_negatives(n_val, n_nodes, pos_set, seed=seed + 1)
    test_neg = _sample_negatives(n_test, n_nodes, pos_set, seed=seed + 2)

    return EdgeSplit(
        train_edge_index=train_edge_index,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        n_nodes=n_nodes,
    )


def full_graph_split(data: Data, seed: int = 42) -> EdgeSplit:
    """Full-graph split for compression evaluation: every real edge is kept
    for training (and is the reconstruction target); none are held out.

    Unlike `split_edges` (used for link-prediction generalization), this
    split serves a memorization/compression objective -- fidelity is judged
    by how well the model reconstructs the exact graph it was trained on,
    not by generalization to unseen edges. Negative sampling for evaluation
    happens separately, at eval time (see `gvls.eval.compression`), not here.

    Args:
        data: PyG Data object with edge_index and num_nodes.
        seed: RNG seed for the (functionally irrelevant, but reproducible)
              edge ordering, for consistency with `split_edges`'s signature.

    Returns:
        EdgeSplit with every real edge (both directions) in train_edge_index
        and empty val/test tensors.
    """
    n_nodes: int = int(data.num_nodes)
    edge_index, _ = remove_self_loops(data.edge_index)

    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    canon = edge_index[:, mask]  # (2, E)

    n_edges = canon.size(1)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n_edges, generator=rng)
    canon = canon[:, perm]

    train_edge_index = torch.cat([canon, canon.flip(0)], dim=1)
    empty = torch.empty((2, 0), dtype=torch.long)

    return EdgeSplit(
        train_edge_index=train_edge_index,
        val_pos=empty,
        val_neg=empty,
        test_pos=empty,
        test_neg=empty,
        n_nodes=n_nodes,
    )


def _sample_negatives(n: int, n_nodes: int, pos_set: set[tuple[int, int]], seed: int) -> Tensor:
    """Sample n non-existing undirected edges (i < j) not in pos_set."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    sampled: list[tuple[int, int]] = []
    while len(sampled) < n:
        batch_size = min((n - len(sampled)) * 4, 100_000)
        u = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        v = torch.randint(0, n_nodes, (batch_size,), generator=rng)
        for ui, vi in zip(u.tolist(), v.tolist()):
            ui, vi = int(ui), int(vi)
            if ui == vi:
                continue
            lo, hi = (ui, vi) if ui < vi else (vi, ui)
            if (lo, hi) not in pos_set:
                sampled.append((lo, hi))
                pos_set.add((lo, hi))  # avoid duplicates within this sample
                if len(sampled) == n:
                    break
    edges = torch.tensor(sampled, dtype=torch.long).t()  # (2, n)
    return edges
