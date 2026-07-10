import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gvls.compression.pooling_sweep import (
    RESULT_FIELDS,
    evaluate_pooled_compression,
    train_pooled_gvls_full_graph,
)
from gvls.compression.sweep import write_results_csv
from gvls.data.splits import full_graph_split
from gvls.models.encoder import GVLSEncoder
from gvls.models.latent_graph import LatentGraphLearner
from gvls.models.pooling import LatentGraphPooling, PooledGVLS

N = 10
IN_CHANNELS = 16
HIDDEN = 32
LATENT_DIM = 4
M = 3
K = 2


@pytest.fixture()
def small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(N, IN_CHANNELS)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.long,
    )
    return x, edge_index


def make_model(mp_rounds: int = 1, method: str = "attention", num_clusters: int = M) -> PooledGVLS:
    encoder = GVLSEncoder(IN_CHANNELS, HIDDEN, LATENT_DIM)
    pooling = LatentGraphPooling(LATENT_DIM, num_clusters)
    lgl = LatentGraphLearner(LATENT_DIM, method=method, k=min(K, num_clusters - 1))
    return PooledGVLS(encoder, pooling, lgl, latent_dim=LATENT_DIM, mp_rounds=mp_rounds)


# ── LatentGraphPooling ───────────────────────────────────────────────────────

def test_assignment_rows_sum_to_one() -> None:
    pooling = LatentGraphPooling(LATENT_DIM, M)
    z = torch.randn(N, LATENT_DIM)
    mu = torch.randn(N, LATENT_DIM)
    log_var = torch.randn(N, LATENT_DIM)
    s, _, _ = pooling(z, mu, log_var)
    assert s.shape == (N, M)
    assert torch.allclose(s.sum(dim=1), torch.ones(N), atol=1e-5)


def test_pooled_gaussian_shapes() -> None:
    pooling = LatentGraphPooling(LATENT_DIM, M)
    z = torch.randn(N, LATENT_DIM)
    mu = torch.randn(N, LATENT_DIM)
    log_var = torch.randn(N, LATENT_DIM)
    _, mu_p, log_var_p = pooling(z, mu, log_var)
    assert mu_p.shape == (M, LATENT_DIM)
    assert log_var_p.shape == (M, LATENT_DIM)


def test_pooled_variance_is_positive() -> None:
    pooling = LatentGraphPooling(LATENT_DIM, M)
    z = torch.randn(N, LATENT_DIM)
    mu = torch.randn(N, LATENT_DIM) * 5
    log_var = torch.randn(N, LATENT_DIM)
    _, _, log_var_p = pooling(z, mu, log_var)
    assert torch.all(log_var_p.exp() > 0)
    assert not torch.isnan(log_var_p).any()


def test_gradient_flows_to_assignment_and_inputs() -> None:
    pooling = LatentGraphPooling(LATENT_DIM, M)
    z = torch.randn(N, LATENT_DIM, requires_grad=True)
    mu = torch.randn(N, LATENT_DIM, requires_grad=True)
    log_var = torch.randn(N, LATENT_DIM, requires_grad=True)
    _s, mu_p, log_var_p = pooling(z, mu, log_var)
    (mu_p.sum() + log_var_p.sum()).backward()
    assert pooling.assign.weight.grad is not None
    assert pooling.assign.weight.grad.abs().sum() > 0
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert z.grad is not None and z.grad.abs().sum() > 0


# ── PooledGVLS ───────────────────────────────────────────────────────────────

def test_pooled_gvls_output_shapes(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model()
    mu, log_var, z, a_z, z_tilde, s, recon_logits = model(x, edge_index)
    assert mu.shape == (M, LATENT_DIM)
    assert log_var.shape == (M, LATENT_DIM)
    assert z.shape == (M, LATENT_DIM)
    assert a_z.shape == (M, M)
    assert z_tilde.shape == (M, LATENT_DIM)
    assert s.shape == (N, M)
    assert recon_logits.shape == (N, N)


@pytest.mark.parametrize("num_clusters", [2, 5, 8])
def test_unpool_shape_independent_of_m(small_graph: tuple, num_clusters: int) -> None:
    x, edge_index = small_graph
    model = make_model(num_clusters=num_clusters)
    *_, recon_logits = model(x, edge_index)
    assert recon_logits.shape == (N, N)


def test_gradient_flow_to_encoder(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model()
    *_, recon_logits = model(x, edge_index)
    recon_logits.sum().backward()
    grad = model.encoder.conv1.lin.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_mp_rounds_zero_leaves_z_tilde_unchanged(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=0)
    _mu, _log_var, z, _a_z, z_tilde, _s, recon_logits = model(x, edge_index)
    assert torch.allclose(z_tilde, z)
    assert not torch.isnan(recon_logits).any()


# ── T3.6 pooling sweep: training + eval + smoke ─────────────────────────────

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


def _run_pooling_grid_point(data, split, pool_ratio: float, epochs: int = 5) -> dict:
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

    num_clusters = max(2, round(pool_ratio * n_nodes))
    model = train_pooled_gvls_full_graph(
        x, train_ei, adj_true, pos_weight, in_channels,
        latent_dim=8, k=2, num_clusters=num_clusters,
        base_cfg=_base_cfg(), epochs=epochs, seed=42, device=device,
    )
    metrics = evaluate_pooled_compression(
        model, x, train_ei, adj_true, pos_edge_index, n_nodes, num_clusters,
        in_channels, num_input_edges, latent_dim=8, k=2, pool_ratio=pool_ratio,
        f1_negative_ratio=1.0, dense_pair_limit=10_000_000, bpe_sample_size=1000,
        seed=42, device=device,
    )
    return {"dataset": "tiny", **metrics}


def test_single_pooling_grid_point_completes(tiny_graph) -> None:
    data, split = tiny_graph
    row = _run_pooling_grid_point(data, split, pool_ratio=0.5)
    assert set(RESULT_FIELDS) - {"dataset"} <= set(row.keys())
    assert 0.0 <= row["reconstruction_f1"] <= 1.0
    assert row["bits_per_edge"] >= 0.0


def test_node_compression_ratio_matches_m_over_n(tiny_graph) -> None:
    data, split = tiny_graph
    row = _run_pooling_grid_point(data, split, pool_ratio=0.5)
    expected_m = max(2, round(0.5 * split.n_nodes))
    assert row["num_clusters"] == expected_m
    assert row["node_compression_ratio"] == pytest.approx(expected_m / split.n_nodes)


def test_pooling_grid_smoke_writes_rows(tiny_graph) -> None:
    data, split = tiny_graph
    rows = [_run_pooling_grid_point(data, split, pr, epochs=5) for pr in (0.5, 0.25)]
    assert len(rows) == 2

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "tiny_pooling.csv")
        write_results_csv(rows, csv_path, fieldnames=RESULT_FIELDS)
        assert Path(csv_path).exists()
        content = Path(csv_path).read_text().strip().splitlines()
        assert len(content) == 3  # header + 2 rows
