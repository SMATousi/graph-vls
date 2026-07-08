import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gvls.compression.sweep import (
    RESULT_FIELDS,
    evaluate_compression,
    select_compression_optimal,
    train_gvls_full_graph,
    write_results_csv,
)
from gvls.data.splits import full_graph_split

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_graph():
    """Small synthetic graph (N=40) for fast full-graph training/eval."""
    n = 40
    torch.manual_seed(7)
    x = torch.randn(n, 8)
    row, col = [], []
    for i in range(n - 1):
        row += [i, i + 1]
        col += [i + 1, i]
    for i in range(0, n - 5, 4):
        row += [i, i + 4]
        col += [i + 4, i]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=n)
    split = full_graph_split(data, seed=42)
    return data, split


def _base_cfg() -> dict:
    return {
        "hidden_dim": 16,
        "mp_rounds": 1,
        "graph_method": "attention",
        "prior": "isotropic",
        "beta": 0.001,
        "lambda_": 1.0,
        "lr": 0.01,
    }


def _run_grid_point(data, split, latent_dim: int, k: int, epochs: int = 5) -> dict:
    device = torch.device("cpu")
    x = data.x.to(device)
    train_ei = split.train_edge_index.to(device)
    n_nodes = split.n_nodes
    in_channels = int(x.size(1))

    adj_true = torch.zeros(n_nodes, n_nodes, device=device)
    adj_true[train_ei[0], train_ei[1]] = 1.0
    num_input_edges = int(train_ei.size(1) // 2)
    pos_weight = float((n_nodes * n_nodes - train_ei.size(1)) / train_ei.size(1))

    src, dst = train_ei[0], train_ei[1]
    pos_edge_index = train_ei[:, src < dst].cpu()

    model = train_gvls_full_graph(
        x, train_ei, adj_true, pos_weight, in_channels,
        latent_dim, k, _base_cfg(), epochs, seed=42, device=device,
    )
    metrics = evaluate_compression(
        model, x, train_ei, adj_true, pos_edge_index, n_nodes,
        in_channels, num_input_edges, latent_dim, k,
        f1_negative_ratio=1.0, dense_pair_limit=10_000_000, bpe_sample_size=1000,
        seed=42, device=device,
    )
    return {"dataset": "tiny", **metrics}


# ── train_gvls_full_graph / evaluate_compression ────────────────────────────

def test_single_grid_point_completes(tiny_graph) -> None:
    data, split = tiny_graph
    row = _run_grid_point(data, split, latent_dim=8, k=2)
    assert set(RESULT_FIELDS) - {"dataset"} <= set(row.keys())
    assert 0.0 <= row["reconstruction_f1"] <= 1.0
    assert row["bits_per_edge"] >= 0.0


def test_dim_compression_ratio_matches_latent_dim_over_features(tiny_graph) -> None:
    data, split = tiny_graph
    row = _run_grid_point(data, split, latent_dim=8, k=2)
    assert row["dim_compression_ratio"] == pytest.approx(8 / 8)  # tiny_graph has F=8


def test_num_latent_edges_consistent_with_edge_ratio(tiny_graph) -> None:
    data, split = tiny_graph
    row = _run_grid_point(data, split, latent_dim=8, k=2)
    expected_ratio = row["num_latent_edges"] / row["num_input_edges"]
    assert row["edge_compression_ratio"] == pytest.approx(expected_ratio)


# ── smoke test: 2x2 grid, writes 4 rows to CSV (plan.md T3.3) ──────────────

def test_2x2_grid_smoke_writes_four_rows(tiny_graph) -> None:
    data, split = tiny_graph
    rows = [
        _run_grid_point(data, split, latent_dim=ld, k=k, epochs=5)
        for ld in (4, 8)
        for k in (2, 5)
    ]
    assert len(rows) == 4

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "tiny.csv")
        write_results_csv(rows, csv_path)
        assert Path(csv_path).exists()
        content = Path(csv_path).read_text().strip().splitlines()
        assert len(content) == 5  # header + 4 rows


# ── select_compression_optimal ──────────────────────────────────────────────

def test_select_compression_optimal_prefers_smaller_within_floor() -> None:
    rows = [
        {"latent_dim": 4, "k": 1, "dim_compression_ratio": 0.1, "edge_compression_ratio": 0.1, "reconstruction_f1": 0.92},
        {"latent_dim": 128, "k": 20, "dim_compression_ratio": 1.0, "edge_compression_ratio": 2.0, "reconstruction_f1": 0.99},
    ]
    best, floor_met = select_compression_optimal(rows, fidelity_floor=0.90)
    assert floor_met is True
    assert best["latent_dim"] == 4


def test_select_compression_optimal_falls_back_when_floor_not_met() -> None:
    rows = [
        {"latent_dim": 4, "k": 1, "dim_compression_ratio": 0.1, "edge_compression_ratio": 0.1, "reconstruction_f1": 0.5},
        {"latent_dim": 128, "k": 20, "dim_compression_ratio": 1.0, "edge_compression_ratio": 2.0, "reconstruction_f1": 0.7},
    ]
    best, floor_met = select_compression_optimal(rows, fidelity_floor=0.90)
    assert floor_met is False
    assert best["latent_dim"] == 128  # highest F1


def test_select_compression_optimal_rejects_empty() -> None:
    with pytest.raises(ValueError):
        select_compression_optimal([], fidelity_floor=0.90)
