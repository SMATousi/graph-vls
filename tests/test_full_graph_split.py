import torch
from torch_geometric.data import Data

from gvls.data.splits import full_graph_split


def _make_graph(n_nodes: int, edges: list[tuple[int, int]]) -> Data:
    """Build a small undirected PyG graph (both directions stored)."""
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=n_nodes)


_N = 50
_EDGES = [(i, i + 1) for i in range(_N - 1)] + [(i, i + 2) for i in range(_N - 2)]
_GRAPH = _make_graph(_N, _EDGES)


def _edge_set(t: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(t[0, i]), int(t[1, i])) for i in range(t.size(1))}


def test_train_edge_index_contains_every_real_edge_both_directions() -> None:
    split = full_graph_split(_GRAPH, seed=42)
    expected = {(u, v) for u, v in _EDGES} | {(v, u) for u, v in _EDGES}
    assert _edge_set(split.train_edge_index) == expected
    assert split.train_edge_index.size(1) == 2 * len(_EDGES)


def test_no_val_test_edges() -> None:
    split = full_graph_split(_GRAPH, seed=42)
    assert split.val_pos.size(1) == 0
    assert split.val_neg.size(1) == 0
    assert split.test_pos.size(1) == 0
    assert split.test_neg.size(1) == 0


def test_determinism() -> None:
    s1 = full_graph_split(_GRAPH, seed=7)
    s2 = full_graph_split(_GRAPH, seed=7)
    assert torch.equal(s1.train_edge_index, s2.train_edge_index)


def test_n_nodes() -> None:
    split = full_graph_split(_GRAPH, seed=42)
    assert split.n_nodes == _N


def test_no_self_loops() -> None:
    split = full_graph_split(_GRAPH, seed=42)
    train = split.train_edge_index
    assert (train[0] != train[1]).all()
