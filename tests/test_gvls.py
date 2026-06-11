import torch
import pytest
from gvls.models.encoder import GVLSEncoder
from gvls.models.gvls import GVLS
from gvls.models.latent_graph import LatentGraphLearner

N = 10
IN_CHANNELS = 16
HIDDEN = 32
LATENT_DIM = 8
K = 4


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


def make_model(mp_rounds: int = 1, method: str = "attention") -> GVLS:
    encoder = GVLSEncoder(IN_CHANNELS, HIDDEN, LATENT_DIM)
    lgl = LatentGraphLearner(LATENT_DIM, method=method, k=K)
    return GVLS(encoder, lgl, latent_dim=LATENT_DIM, mp_rounds=mp_rounds)


def test_output_shapes(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=1)
    mu, log_var, z, A_z, z_tilde = model(x, edge_index)
    assert mu.shape == (N, LATENT_DIM)
    assert log_var.shape == (N, LATENT_DIM)
    assert z.shape == (N, LATENT_DIM)
    assert A_z.shape == (N, N)
    assert z_tilde.shape == (N, LATENT_DIM)


def test_mp_rounds_zero(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=0)
    _, _, z, _, z_tilde = model(x, edge_index)
    assert torch.allclose(z_tilde, z)


@pytest.mark.parametrize("mp_rounds", [1, 2])
def test_mp_rounds_changes_z_tilde(small_graph: tuple, mp_rounds: int) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=mp_rounds)
    _, _, z, _, z_tilde = model(x, edge_index)
    # After at least one round of message passing, z_tilde should differ from z.
    assert not torch.allclose(z_tilde, z)


def test_gradient_flow_to_encoder(small_graph: tuple) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=1)
    _, _, _, _, z_tilde = model(x, edge_index)
    z_tilde.sum().backward()
    grad = model.encoder.conv1.lin.weight.grad
    assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["attention", "fgp", "nri"])
def test_all_latent_graph_methods(small_graph: tuple, method: str) -> None:
    x, edge_index = small_graph
    model = make_model(mp_rounds=1, method=method)
    mu, log_var, z, A_z, z_tilde = model(x, edge_index)
    assert z_tilde.shape == (N, LATENT_DIM)
    assert not torch.isnan(z_tilde).any()
