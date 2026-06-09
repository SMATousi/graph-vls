import torch
import pytest
from torch_geometric.data import Data

from gvls.data.splits import EdgeSplit, split_edges


def _make_graph(n_nodes: int, edges: list[tuple[int, int]]) -> Data:
    """Build a small undirected PyG graph (both directions stored)."""
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=n_nodes)


# 50-node sparse graph: path + skip-2 edges → 97 edges, 1225 possible → 1128 non-existing.
# Sparse enough that negative sampling never exhausts available edges at any train ratio.
_N = 50
_EDGES = [(i, i + 1) for i in range(_N - 1)] + [(i, i + 2) for i in range(_N - 2)]
_GRAPH = _make_graph(_N, _EDGES)


@pytest.mark.parametrize("ratio", [0.2, 0.4, 0.8])
def test_split_sizes(ratio: float) -> None:
    split = split_edges(_GRAPH, train_ratio=ratio, seed=42)
    n_edges = len(_EDGES)
    n_train = int(n_edges * ratio)
    n_remaining = n_edges - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    assert split.train_edge_index.size(1) == 2 * n_train
    assert split.val_pos.size(1) == n_val
    assert split.test_pos.size(1) == n_test
    assert split.val_neg.size(1) == n_val
    assert split.test_neg.size(1) == n_test


def test_no_train_test_leakage() -> None:
    split = split_edges(_GRAPH, train_ratio=0.6, seed=42)

    def edge_set(t: torch.Tensor) -> set[tuple[int, int]]:
        return {(int(t[0, i]), int(t[1, i])) for i in range(t.size(1))}

    # Canonicalise train edges (keep i < j).
    train = split.train_edge_index
    mask = train[0] < train[1]
    train_canon = edge_set(train[:, mask])

    val_pos = edge_set(split.val_pos)
    test_pos = edge_set(split.test_pos)

    assert train_canon.isdisjoint(val_pos), "train/val overlap"
    assert train_canon.isdisjoint(test_pos), "train/test overlap"
    assert val_pos.isdisjoint(test_pos), "val/test overlap"


def test_negative_sampling_ratio() -> None:
    split = split_edges(_GRAPH, train_ratio=0.6, seed=42)
    assert split.val_neg.size(1) == split.val_pos.size(1)
    assert split.test_neg.size(1) == split.test_pos.size(1)


def test_negatives_are_not_positive_edges() -> None:
    split = split_edges(_GRAPH, train_ratio=0.6, seed=42)
    all_edges = set(_EDGES)

    def check(neg: torch.Tensor, label: str) -> None:
        for i in range(neg.size(1)):
            u, v = int(neg[0, i]), int(neg[1, i])
            lo, hi = (u, v) if u < v else (v, u)
            assert (lo, hi) not in all_edges, f"{label} negative edge {lo},{hi} is a real edge"

    check(split.val_neg, "val")
    check(split.test_neg, "test")


def test_no_self_loops_in_negatives() -> None:
    split = split_edges(_GRAPH, train_ratio=0.6, seed=42)
    for neg in (split.val_neg, split.test_neg):
        assert (neg[0] != neg[1]).all(), "self-loop found in negative edges"


def test_determinism() -> None:
    s1 = split_edges(_GRAPH, train_ratio=0.6, seed=99)
    s2 = split_edges(_GRAPH, train_ratio=0.6, seed=99)
    assert torch.equal(s1.train_edge_index, s2.train_edge_index)
    assert torch.equal(s1.val_pos, s2.val_pos)
    assert torch.equal(s1.test_pos, s2.test_pos)
    assert torch.equal(s1.val_neg, s2.val_neg)
    assert torch.equal(s1.test_neg, s2.test_neg)


def test_different_seeds_differ() -> None:
    s1 = split_edges(_GRAPH, train_ratio=0.6, seed=1)
    s2 = split_edges(_GRAPH, train_ratio=0.6, seed=2)
    assert not torch.equal(s1.val_pos, s2.val_pos)


def test_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        split_edges(_GRAPH, train_ratio=0.0)
    with pytest.raises(ValueError):
        split_edges(_GRAPH, train_ratio=1.0)


def test_n_nodes() -> None:
    split = split_edges(_GRAPH, train_ratio=0.6, seed=42)
    assert split.n_nodes == _N
