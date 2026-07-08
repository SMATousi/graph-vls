import numpy as np
import pytest
import torch

from gvls.eval.compression import (
    dim_compression_ratio,
    edge_compression_ratio,
    eval_pairs_with_labels,
    reconstruction_f1,
    sample_node_pairs,
)
from gvls.eval.metrics import bits_per_edge

# ── reconstruction_f1 ────────────────────────────────────────────────────────

def test_reconstruction_f1_perfect() -> None:
    y_true = np.array([1, 1, 0, 0])
    logits = np.array([100.0, 100.0, -100.0, -100.0])
    assert reconstruction_f1(y_true, logits) == pytest.approx(1.0)


def test_reconstruction_f1_all_zero_predictor() -> None:
    y_true = np.array([1, 1, 0, 0])
    logits = np.array([-100.0, -100.0, -100.0, -100.0])
    assert reconstruction_f1(y_true, logits) == pytest.approx(0.0)


def test_reconstruction_f1_threshold_is_strict() -> None:
    # sigmoid(0) == 0.5 == threshold -> predicted negative (strict >, not >=)
    y_true = np.array([1, 0])
    logits = np.array([0.0, 0.0])
    assert reconstruction_f1(y_true, logits, threshold=0.5) == pytest.approx(0.0)


def test_reconstruction_f1_accepts_tensors() -> None:
    y_true = torch.tensor([1, 0, 1, 0])
    logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
    assert reconstruction_f1(y_true, logits) == pytest.approx(1.0)


def test_reconstruction_f1_returns_float() -> None:
    assert isinstance(reconstruction_f1(np.array([1]), np.array([1.0])), float)


# ── dim_compression_ratio ────────────────────────────────────────────────────

def test_dim_compression_ratio_value() -> None:
    assert dim_compression_ratio(32, 1433) == pytest.approx(32 / 1433, rel=1e-6)


def test_dim_compression_ratio_rejects_nonpositive_features() -> None:
    with pytest.raises(ValueError):
        dim_compression_ratio(32, 0)


# ── edge_compression_ratio ───────────────────────────────────────────────────

def test_edge_compression_ratio_toy_graph() -> None:
    # 5-node A_z with edges (0,1), (0,2), (3,4) -> 3 undirected edges
    a_z = np.zeros((5, 5))
    for i, j in [(0, 1), (0, 2), (3, 4)]:
        a_z[i, j] = 1.0
        a_z[j, i] = 1.0
    assert edge_compression_ratio(a_z, num_input_edges=6) == pytest.approx(3 / 6)


def test_edge_compression_ratio_denser_than_input() -> None:
    a_z = np.ones((4, 4)) - np.eye(4)  # complete graph -> 6 undirected edges
    assert edge_compression_ratio(a_z, num_input_edges=3) == pytest.approx(2.0)


def test_edge_compression_ratio_accepts_tensors() -> None:
    a_z = torch.zeros((3, 3))
    a_z[0, 1] = a_z[1, 0] = 1.0
    assert edge_compression_ratio(a_z, num_input_edges=1) == pytest.approx(1.0)


def test_edge_compression_ratio_rejects_nonpositive_input_edges() -> None:
    with pytest.raises(ValueError):
        edge_compression_ratio(np.zeros((3, 3)), num_input_edges=0)


# ── sample_node_pairs ────────────────────────────────────────────────────────

def test_sample_node_pairs_shape_and_bounds() -> None:
    rows, cols = sample_node_pairs(n_nodes=50, num_samples=100, seed=0)
    assert rows.shape == (100,)
    assert cols.shape == (100,)
    assert torch.all(rows < cols)
    assert torch.all(cols < 50)


def test_sample_node_pairs_determinism() -> None:
    r1, c1 = sample_node_pairs(n_nodes=50, num_samples=100, seed=7)
    r2, c2 = sample_node_pairs(n_nodes=50, num_samples=100, seed=7)
    assert torch.equal(r1, r2)
    assert torch.equal(c1, c2)


def test_sample_node_pairs_no_duplicates() -> None:
    rows, cols = sample_node_pairs(n_nodes=30, num_samples=50, seed=1)
    pairs = set(zip(rows.tolist(), cols.tolist()))
    assert len(pairs) == 50


def test_sample_node_pairs_clamps_to_max_pairs() -> None:
    rows, _cols = sample_node_pairs(n_nodes=4, num_samples=1000, seed=0)  # max 6 pairs
    assert rows.shape[0] == 6


# ── eval_pairs_with_labels ───────────────────────────────────────────────────

def test_eval_pairs_includes_all_positives() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    rows, cols, labels = eval_pairs_with_labels(
        n_nodes=10, pos_edge_index=edge_index, num_negatives=5, seed=0
    )
    pos_pairs = {(0, 1), (1, 2), (2, 3)}
    predicted_pos = {
        (r, c) for r, c, label in zip(rows.tolist(), cols.tolist(), labels.tolist()) if label == 1.0
    }
    assert predicted_pos == pos_pairs
    assert (labels == 0.0).sum().item() == 5
    assert rows.shape[0] == cols.shape[0] == labels.shape[0] == 8


def test_eval_pairs_negatives_exclude_positives() -> None:
    edge_index = torch.tensor([[0], [1]])
    rows, cols, labels = eval_pairs_with_labels(
        n_nodes=5, pos_edge_index=edge_index, num_negatives=8, seed=0
    )
    neg_pairs = {
        (r, c) for r, c, label in zip(rows.tolist(), cols.tolist(), labels.tolist()) if label == 0.0
    }
    assert (0, 1) not in neg_pairs


def test_eval_pairs_determinism() -> None:
    edge_index = torch.tensor([[0, 2], [1, 3]])
    r1, c1, l1 = eval_pairs_with_labels(
        n_nodes=10, pos_edge_index=edge_index, num_negatives=4, seed=3
    )
    r2, c2, l2 = eval_pairs_with_labels(
        n_nodes=10, pos_edge_index=edge_index, num_negatives=4, seed=3
    )
    assert torch.equal(r1, r2)
    assert torch.equal(c1, c2)
    assert torch.equal(l1, l2)


# ── bits_per_edge sampled estimate vs. exact (V-1) ──────────────────────────

def test_bits_per_edge_sampled_matches_exact_on_small_graph() -> None:
    n = 40
    rng = np.random.default_rng(0)
    a = (rng.random((n, n)) < 0.1).astype(float)
    a = np.triu(a, k=1)
    a = a + a.T
    logits = rng.standard_normal((n, n)) * 2.0
    logits = np.triu(logits, k=1)
    logits = logits + logits.T

    iu, ju = np.triu_indices(n, k=1)
    exact = bits_per_edge(a[iu, ju], logits[iu, ju])

    # num_samples == all available pairs -> sample_node_pairs covers every
    # pair (just reordered), so the mean must match the exact computation.
    rows, cols = sample_node_pairs(n_nodes=n, num_samples=iu.shape[0], seed=1)
    sampled = bits_per_edge(a[rows.numpy(), cols.numpy()], logits[rows.numpy(), cols.numpy()])
    assert sampled == pytest.approx(exact, abs=1e-6)
