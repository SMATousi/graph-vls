import torch
import pytest
from gvls.models.encoder import GVLSEncoder


@pytest.fixture()
def small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
        dtype=torch.long,
    )
    return x, edge_index


@pytest.fixture()
def encoder() -> GVLSEncoder:
    return GVLSEncoder(in_channels=16, hidden_channels=32, latent_dim=8)


def test_output_shapes(encoder: GVLSEncoder, small_graph: tuple) -> None:
    x, edge_index = small_graph
    mu, log_var, z = encoder(x, edge_index)
    assert mu.shape == (10, 8)
    assert log_var.shape == (10, 8)
    assert z.shape == (10, 8)


def test_gradient_flow(encoder: GVLSEncoder, small_graph: tuple) -> None:
    x, edge_index = small_graph
    mu, log_var, z = encoder(x, edge_index)
    loss = z.sum()
    loss.backward()
    assert encoder.conv1.lin.weight.grad is not None
    assert encoder.conv1.lin.weight.grad.abs().sum() > 0


def test_eval_mode_no_sampling(encoder: GVLSEncoder, small_graph: tuple) -> None:
    x, edge_index = small_graph
    encoder.eval()
    with torch.no_grad():
        mu, log_var, z = encoder(x, edge_index)
    assert torch.allclose(z, mu)


def test_log_var_clamped(encoder: GVLSEncoder, small_graph: tuple) -> None:
    x, edge_index = small_graph
    # Manually override weights to produce extreme values; clamping must cap them.
    with torch.no_grad():
        encoder.log_var_head.lin.weight.fill_(100.0)
        encoder.log_var_head.bias.fill_(100.0)  # type: ignore[union-attr]
    _, log_var, _ = encoder(x, edge_index)
    assert log_var.max().item() <= 10.0 + 1e-5
    assert log_var.min().item() >= -10.0 - 1e-5
