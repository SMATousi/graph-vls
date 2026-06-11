import pytest
import torch
from gvls.models.latent_graph import LatentGraphLearner

N = 12
D = 8
K = 4


@pytest.fixture(params=["attention", "fgp", "nri"])
def learner(request: pytest.FixtureRequest) -> LatentGraphLearner:
    return LatentGraphLearner(latent_dim=D, method=request.param, k=K)


@pytest.fixture()
def z() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(N, D)


def test_output_shape(learner: LatentGraphLearner, z: torch.Tensor) -> None:
    A = learner(z)
    assert A.shape == (N, N)


def test_values_in_range(learner: LatentGraphLearner, z: torch.Tensor) -> None:
    A = learner(z)
    assert A.min().item() >= 0.0
    assert A.max().item() <= 1.0


def test_zero_diagonal(learner: LatentGraphLearner, z: torch.Tensor) -> None:
    A = learner(z)
    assert A.diagonal().abs().max().item() == pytest.approx(0.0)


def test_symmetry(learner: LatentGraphLearner, z: torch.Tensor) -> None:
    A = learner(z)
    assert torch.allclose(A, A.T, atol=1e-6)


def test_sparsification(learner: LatentGraphLearner, z: torch.Tensor) -> None:
    A = learner(z)
    # Union symmetrisation: before symmetrising each row has exactly k non-zeros;
    # after, the mean is ≈ 2k. The graph must not be fully connected.
    nonzero_per_row = (A > 0).float()
    assert nonzero_per_row.mean().item() <= 2 * K
    assert nonzero_per_row.max().item() < N  # not fully connected


def test_gradient_flow_attention(z: torch.Tensor) -> None:
    # Attention has no module parameters; verify gradient flows back through z.
    z = z.requires_grad_(True)
    learner = LatentGraphLearner(latent_dim=D, method="attention", k=K)
    A = learner(z)
    A.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_gradient_flow_fgp(z: torch.Tensor) -> None:
    learner = LatentGraphLearner(latent_dim=D, method="fgp", k=K)
    A = learner(z)
    A.sum().backward()
    assert learner.log_tau.grad is not None
    assert learner.log_tau.grad.abs().sum() > 0


def test_gradient_flow_nri(z: torch.Tensor) -> None:
    learner = LatentGraphLearner(latent_dim=D, method="nri", k=K)
    A = learner(z)
    A.sum().backward()
    w = learner.nri_mlp[0].weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        LatentGraphLearner(latent_dim=D, method="bad")


def test_k_larger_than_n_minus_1() -> None:
    # k capped at N-1 — should not raise and diagonal stays zero.
    learner = LatentGraphLearner(latent_dim=D, method="attention", k=1000)
    z = torch.randn(5, D)
    A = learner(z)
    assert A.shape == (5, 5)
    assert A.diagonal().abs().max().item() == pytest.approx(0.0)
