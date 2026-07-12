import pytest
import torch
from gvls.losses.elbo import elbo, kl_graph_mrf, kl_isotropic

N, D = 8, 4


def sym_adj(n: int) -> torch.Tensor:
    """Random symmetric soft adjacency with zero diagonal."""
    A = torch.rand(n, n)
    A = (A + A.T) / 2
    A.fill_diagonal_(0.0)
    return A * 0.5


# ── kl_isotropic ─────────────────────────────────────────────────────────────


def test_kl_isotropic_zero_at_prior() -> None:
    mu = torch.zeros(N, D)
    log_var = torch.zeros(N, D)
    assert kl_isotropic(mu, log_var).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_isotropic_positive_for_nonzero_mu() -> None:
    mu = torch.ones(N, D)
    log_var = torch.zeros(N, D)
    assert kl_isotropic(mu, log_var).item() > 0.0


def test_kl_isotropic_positive_for_nonzero_log_var() -> None:
    mu = torch.zeros(N, D)
    log_var = torch.ones(N, D)
    assert kl_isotropic(mu, log_var).item() > 0.0


def test_kl_isotropic_nonneg() -> None:
    torch.manual_seed(0)
    mu = torch.randn(N, D)
    log_var = torch.randn(N, D)
    assert kl_isotropic(mu, log_var).item() >= 0.0


def test_kl_isotropic_invariant_to_node_count() -> None:
    """Normalization bug regression test (specs/phase3/validation.md V-8):
    per-node KL magnitude must not scale with how many nodes it's computed
    over -- tiling the same per-node distribution across more nodes should
    leave the returned value unchanged, not grow linearly with node count.
    """
    torch.manual_seed(3)
    mu_one = torch.randn(1, D)
    log_var_one = torch.randn(1, D)
    kl_one = kl_isotropic(mu_one, log_var_one).item()

    reps = 50
    mu_many = mu_one.repeat(reps, 1)
    log_var_many = log_var_one.repeat(reps, 1)
    kl_many = kl_isotropic(mu_many, log_var_many).item()

    assert kl_many == pytest.approx(kl_one, rel=1e-5)


# ── kl_graph_mrf ─────────────────────────────────────────────────────────────


def test_kl_graph_mrf_finite() -> None:
    torch.manual_seed(0)
    mu = torch.randn(N, D)
    log_var = torch.zeros(N, D)
    A_z = sym_adj(N)
    val = kl_graph_mrf(mu, log_var, A_z)
    assert torch.isfinite(val)


def test_kl_graph_mrf_nonneg() -> None:
    torch.manual_seed(1)
    mu = torch.randn(N, D)
    log_var = torch.zeros(N, D)
    A_z = sym_adj(N)
    assert kl_graph_mrf(mu, log_var, A_z).item() >= 0.0


def test_kl_graph_mrf_gradient_flows() -> None:
    torch.manual_seed(2)
    mu = torch.randn(N, D, requires_grad=True)
    log_var = torch.zeros(N, D)
    A_z = sym_adj(N)
    kl_graph_mrf(mu, log_var, A_z).backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0


def test_kl_graph_mrf_invariant_to_node_count() -> None:
    """Normalization bug regression test (specs/phase3/validation.md V-8):
    replicating the same disconnected component `reps` times (block-diagonal
    A_z, so each replica is an independent MRF) must leave the per-node KL
    magnitude unchanged, not scale linearly with total node count.
    """
    torch.manual_seed(4)
    n_one = 3
    mu_one = torch.randn(n_one, D)
    log_var_one = torch.zeros(n_one, D)
    A_one = sym_adj(n_one)
    kl_one = kl_graph_mrf(mu_one, log_var_one, A_one).item()

    reps = 10
    mu_many = mu_one.repeat(reps, 1)
    log_var_many = log_var_one.repeat(reps, 1)
    A_many = torch.block_diag(*[A_one for _ in range(reps)])
    kl_many = kl_graph_mrf(mu_many, log_var_many, A_many).item()

    assert kl_many == pytest.approx(kl_one, rel=1e-4)


# ── elbo ─────────────────────────────────────────────────────────────────────


def test_elbo_isotropic_finite() -> None:
    torch.manual_seed(0)
    N2 = 10
    recon_logits = torch.randn(N2, N2)
    adj_true = (torch.rand(N2, N2) > 0.8).float()
    mu = torch.randn(N2, D)
    log_var = torch.zeros(N2, D)
    A_z = sym_adj(N2)
    val = elbo(recon_logits, adj_true, mu, log_var, A_z, prior="isotropic")
    assert torch.isfinite(val)


def test_elbo_graph_mrf_finite() -> None:
    torch.manual_seed(0)
    N2 = 10
    recon_logits = torch.randn(N2, N2)
    adj_true = (torch.rand(N2, N2) > 0.8).float()
    mu = torch.randn(N2, D)
    log_var = torch.zeros(N2, D)
    A_z = sym_adj(N2)
    val = elbo(recon_logits, adj_true, mu, log_var, A_z, prior="graph_mrf")
    assert torch.isfinite(val)


def test_nan_guard_fires() -> None:
    N2 = 6
    mu = torch.full((N2, D), float("nan"))
    log_var = torch.zeros(N2, D)
    A_z = sym_adj(N2)
    recon_logits = torch.zeros(N2, N2)
    adj_true = torch.zeros(N2, N2)
    with pytest.raises(RuntimeError, match="NaN loss detected"):
        elbo(recon_logits, adj_true, mu, log_var, A_z, prior="isotropic")


def test_invalid_prior_raises() -> None:
    N2 = 4
    with pytest.raises(ValueError, match="prior must be one of"):
        elbo(
            torch.zeros(N2, N2),
            torch.zeros(N2, N2),
            torch.zeros(N2, D),
            torch.zeros(N2, D),
            torch.zeros(N2, N2),
            prior="bad",
        )
