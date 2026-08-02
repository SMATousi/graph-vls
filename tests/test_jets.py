from unittest.mock import patch

import numpy as np
import pytest
import torch

from gvls.data.jets import (
    NUM_FEATURES,
    PDGIDS,
    JetSplit,
    build_jet_graph,
    load_qg_jets_fixed_pool,
    load_qg_jets_lorentz_protocol,
    load_split_from_config,
    split_jets,
    subsample_train,
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


# ── Lorentz-EQGNN protocol (T4.10) ──────────────────────────────────────────


def _fake_raw_pool(
    n_jets: int, min_n: int = 5, max_n: int = 15, class0_frac: float = 0.5, seed: int = 0
) -> tuple[list, np.ndarray]:
    """(raw_x, raw_y) shaped like energyflow.qg_jets.load(pad=False)'s return:
    raw_x a list of (n_particles_i, 4) (pT, y, phi, pdgid) arrays with varying
    n_particles_i, raw_y a (n_jets,) 0/1 label array.
    """
    rng = np.random.default_rng(seed)
    n_particles = rng.integers(min_n, max_n + 1, size=n_jets)
    raw_x = [_synthetic_jet(int(n), seed=seed * 10_000 + i) for i, n in enumerate(n_particles)]
    n_class0 = int(round(n_jets * class0_frac))
    raw_y = np.array([0] * n_class0 + [1] * (n_jets - n_class0), dtype=np.int64)
    rng.shuffle(raw_y)
    return raw_x, raw_y


def test_fixed_pool_respects_min_particles() -> None:
    raw_x, raw_y = _fake_raw_pool(200, min_n=5, max_n=15, seed=1)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        graphs = load_qg_jets_fixed_pool(num_jets=20, min_particles=10, seed=0)
    assert len(graphs) == 20
    assert all(g.num_nodes >= 10 for g in graphs)


def test_fixed_pool_does_not_force_class_balance() -> None:
    # Only 5 of 200 jets are class 0 -- a balanced draw is impossible, so a
    # correct (non-balance-forcing) implementation must return far fewer
    # than half class-0 jets in the sample, unlike load_qg_jets's exact 50/50.
    raw_x, raw_y = _fake_raw_pool(200, min_n=10, max_n=10, class0_frac=0.025, seed=2)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        graphs = load_qg_jets_fixed_pool(num_jets=100, min_particles=1, seed=0)
    n_class0 = sum(1 for g in graphs if g.y.item() == 0)
    assert n_class0 <= 5
    assert n_class0 != 50


def test_fixed_pool_insufficient_pool_raises() -> None:
    raw_x, raw_y = _fake_raw_pool(10, min_n=10, max_n=10, seed=3)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        with pytest.raises(ValueError):
            load_qg_jets_fixed_pool(num_jets=20, min_particles=1, seed=0)


def test_fixed_pool_deterministic() -> None:
    raw_x, raw_y = _fake_raw_pool(200, min_n=10, max_n=10, seed=4)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        g1 = load_qg_jets_fixed_pool(num_jets=20, min_particles=1, seed=42)
        g2 = load_qg_jets_fixed_pool(num_jets=20, min_particles=1, seed=42)
    assert [g.y.item() for g in g1] == [g.y.item() for g in g2]


def test_lorentz_protocol_split_sizes_and_partition() -> None:
    raw_x, raw_y = _fake_raw_pool(50, min_n=10, max_n=10, seed=5)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        split = load_qg_jets_lorentz_protocol(
            num_train=20, num_val=10, num_test=10, min_particles=1, seed=0
        )
    assert len(split.train) == 20
    assert len(split.val) == 10
    assert len(split.test) == 10
    all_ids = (
        {id(g) for g in split.train} | {id(g) for g in split.val} | {id(g) for g in split.test}
    )
    assert len(all_ids) == 40


# ── Training-subset resampling (T4.10) ──────────────────────────────────────


def _dummy_split(n_train: int, n_val: int, n_test: int) -> JetSplit:
    return JetSplit(
        train=_dummy_graphs(n_train), val=_dummy_graphs(n_val), test=_dummy_graphs(n_test)
    )


def test_subsample_train_shrinks_only_train() -> None:
    split = _dummy_split(100, 10, 10)
    sub = subsample_train(split, n=20, seed=0)
    assert len(sub.train) == 20
    assert sub.val is split.val
    assert sub.test is split.test


def test_subsample_train_rejects_n_too_large() -> None:
    split = _dummy_split(10, 5, 5)
    with pytest.raises(ValueError):
        subsample_train(split, n=20, seed=0)


# ── load_split_from_config dispatch (T4.10) ─────────────────────────────────


def test_load_split_from_config_balanced_protocol() -> None:
    raw_x, raw_y = _fake_raw_pool(200, min_n=10, max_n=10, class0_frac=0.5, seed=6)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        split = load_split_from_config(
            {
                "protocol": "balanced",
                "num_jets": 20,
                "k_graph_cap": 8,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "seed": 0,
            }
        )
    assert len(split.train) + len(split.val) + len(split.test) == 20


def test_load_split_from_config_lorentz_protocol_with_train_subset() -> None:
    raw_x, raw_y = _fake_raw_pool(100, min_n=10, max_n=10, seed=7)
    with patch("energyflow.qg_jets.load", return_value=(raw_x, raw_y)):
        split = load_split_from_config(
            {
                "protocol": "lorentz",
                "k_graph_cap": 8,
                "seed": 0,
                "num_train": 20,
                "num_val": 5,
                "num_test": 5,
                "min_particles": 1,
                "train_subset": 8,
            }
        )
    assert len(split.train) == 8
    assert len(split.val) == 5
    assert len(split.test) == 5


def test_load_split_from_config_unknown_protocol_raises() -> None:
    with pytest.raises(ValueError):
        load_split_from_config({"protocol": "bogus"})
