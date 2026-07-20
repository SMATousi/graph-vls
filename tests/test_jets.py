import numpy as np
import pytest
import torch

from gvls.data.jets import (
    NUM_FEATURES,
    PDGIDS,
    build_jet_graph,
    split_jets,
)


def _synthetic_jet(n: int, seed: int = 0) -> np.ndarray:
    """(n, 4) array of (pT, y, phi, pdgid), roughly collimated like a real jet."""
    rng = np.random.default_rng(seed)
    pt = rng.uniform(0.5, 50.0, size=n)
    y = rng.normal(0.0, 0.3, size=n)
    phi = rng.normal(4.0, 0.3, size=n)
    pdgid = rng.choice(PDGIDS, size=n)
    return np.stack([pt, y, phi, pdgid], axis=1)


# ── k-NN graph construction ─────────────────────────────────────────────────

def test_knn_graph_no_self_loops_and_symmetric() -> None:
    particles = _synthetic_jet(30, seed=1)
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    edge_index = graph.edge_index

    assert (edge_index[0] != edge_index[1]).all(), "self-loop found"

    edge_set = {(int(u), int(v)) for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist())}
    for u, v in edge_set:
        assert (v, u) in edge_set, f"edge ({u},{v}) has no reverse edge"


def test_knn_graph_degree_bounded_by_k_graph() -> None:
    # Union-symmetrized k-NN: out-degree can exceed k_graph_cap slightly (a
    # popular neighbor gets picked by more than k_graph_cap others), but stays
    # bounded -- it should not blow up towards a near-complete graph.
    n, k_graph_cap = 40, 8
    particles = _synthetic_jet(n, seed=2)
    graph = build_jet_graph(particles, label=0, k_graph_cap=k_graph_cap)

    degree = torch.bincount(graph.edge_index[0], minlength=n)
    assert degree.max().item() <= 4 * k_graph_cap
    assert degree.min().item() >= 1


def test_knn_graph_handles_single_particle_jet() -> None:
    particles = _synthetic_jet(1, seed=3)
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    assert graph.edge_index.shape == (2, 0)
    assert graph.num_nodes == 1


def test_knn_graph_small_jet_uses_all_other_particles() -> None:
    # n - 1 < k_graph_cap: every particle should end up connected to every other.
    n = 4
    particles = _synthetic_jet(n, seed=4)
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    degree = torch.bincount(graph.edge_index[0], minlength=n)
    assert (degree == n - 1).all()


def test_knn_graph_respects_phi_periodicity() -> None:
    # Two particles near phi=0 and phi=2*pi are angularly adjacent despite a
    # large raw phi difference; a non-periodic implementation would treat them
    # as far apart and might not connect them.
    particles = np.array(
        [
            [10.0, 0.0, 0.01, 22],
            [10.0, 0.0, 2 * np.pi - 0.01, 22],
            [10.0, 5.0, 5.0, 22],  # far away in y, should not be their nearest neighbor
        ]
    )
    graph = build_jet_graph(particles, label=0, k_graph_cap=1)
    edge_set = {(int(u), int(v)) for u, v in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())}
    assert (0, 1) in edge_set and (1, 0) in edge_set


# ── Feature engineering ──────────────────────────────────────────────────────

def test_feature_shape_matches_num_features() -> None:
    particles = _synthetic_jet(25, seed=5)
    graph = build_jet_graph(particles, label=1, k_graph_cap=8)
    assert graph.x.shape == (25, NUM_FEATURES)


def test_feature_log_pt_matches_input() -> None:
    particles = _synthetic_jet(10, seed=6)
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    assert torch.allclose(graph.x[:, 0], torch.from_numpy(np.log(particles[:, 0])).float(), atol=1e-5)


def test_pdgid_one_hot_rows_sum_to_one() -> None:
    particles = _synthetic_jet(15, seed=7)
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    onehot = graph.x[:, 3:]
    assert torch.allclose(onehot.sum(dim=1), torch.ones(15))


def test_unknown_pdgid_falls_into_unknown_bucket() -> None:
    particles = _synthetic_jet(5, seed=8)
    particles[0, 3] = 999999  # not in PDGIDS
    graph = build_jet_graph(particles, label=0, k_graph_cap=8)
    onehot_row0 = graph.x[0, 3:]
    assert onehot_row0[len(PDGIDS)].item() == 1.0
    assert onehot_row0.sum().item() == 1.0


def test_label_stored_correctly() -> None:
    particles = _synthetic_jet(10, seed=9)
    graph = build_jet_graph(particles, label=1, k_graph_cap=8)
    assert graph.y.item() == 1


# ── Split determinism ────────────────────────────────────────────────────────

def _dummy_graphs(n: int) -> list:
    return [build_jet_graph(_synthetic_jet(10, seed=i), label=i % 2, k_graph_cap=8) for i in range(n)]


def test_split_sizes() -> None:
    graphs = _dummy_graphs(20)
    split = split_jets(graphs, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert len(split.train) == 14
    assert len(split.val) == 3
    assert len(split.test) == 3


def test_split_determinism() -> None:
    graphs = _dummy_graphs(20)
    s1 = split_jets(graphs, seed=42)
    s2 = split_jets(graphs, seed=42)
    assert [g.y.item() for g in s1.train] == [g.y.item() for g in s2.train]
    assert [g.y.item() for g in s1.val] == [g.y.item() for g in s2.val]
    assert [g.y.item() for g in s1.test] == [g.y.item() for g in s2.test]


def test_split_different_seeds_differ() -> None:
    graphs = _dummy_graphs(20)
    s1 = split_jets(graphs, seed=1)
    s2 = split_jets(graphs, seed=2)
    assert [g.y.item() for g in s1.train] != [g.y.item() for g in s2.train]


def test_split_is_a_partition() -> None:
    graphs = _dummy_graphs(20)
    split = split_jets(graphs, seed=42)
    all_ids = {id(g) for g in split.train} | {id(g) for g in split.val} | {id(g) for g in split.test}
    assert len(all_ids) == 20


def test_split_invalid_ratios() -> None:
    graphs = _dummy_graphs(10)
    with pytest.raises(ValueError):
        split_jets(graphs, train_ratio=0.9, val_ratio=0.2)
    with pytest.raises(ValueError):
        split_jets(graphs, train_ratio=1.0)
