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


# ── T5.1: ELBO normalization (specs/phase5/) ─────────────────────────────────


def _elbo_parts(
    n_nodes: int, m_clusters: int, beta: float, normalization: str, seed: int = 0
) -> tuple[float, float]:
    """(recon_only, kl_contribution) for one synthetic graph.

    The reconstruction side is held deliberately constant across `n_nodes`
    (all-zero logits against an all-zero adjacency gives BCE = log 2 per
    entry, whatever N is), so any N-dependence in the returned pair comes
    from the normalization under test and nothing else.

    Everything runs in float64. The KL contribution is recovered by
    subtracting two nearly-equal totals, and under `per_jet` at N=139 that
    difference is ~1e-5 against a ~0.69 reconstruction term -- in float32 the
    cancellation destroys almost every significant digit and the test fails
    on its own arithmetic rather than on the code under test.
    """
    torch.manual_seed(seed)
    recon_logits = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    adj_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    mu = torch.randn(m_clusters, D, dtype=torch.float64)
    log_var = torch.zeros(m_clusters, D, dtype=torch.float64)
    A_z = sym_adj(m_clusters).to(torch.float64)

    recon_only = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=0.0, normalization=normalization
    ).item()
    total = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=beta, normalization=normalization
    ).item()
    return recon_only, total - recon_only


def _beta_eff(n_nodes: int, m_clusters: int, beta: float, normalization: str) -> float:
    """Effective beta: the KL:recon ratio *relative to a true per-graph ELBO*.

    This is the quantity T5.1 is about, and it is deliberately not the raw
    KL:recon ratio. A true per-graph ELBO is `recon_sum + beta*KL_sum`, whose
    KL:recon ratio is itself N-dependent (more observed pairs means the
    likelihood outweighs the prior -- correct Bayesian behaviour, not a
    defect). What a normalization can get wrong is the *multiplier* on top of
    that, which should be exactly `beta` for every graph.
    """
    recon_mean, kl_term = _elbo_parts(n_nodes, m_clusters, beta, normalization)
    # true-ELBO ratio, scaled the same way: (beta*KL_sum/N^2) / (recon_sum/N^2)
    torch.manual_seed(0)
    mu = torch.randn(m_clusters, D, dtype=torch.float64)
    log_var = torch.zeros(m_clusters, D, dtype=torch.float64)
    kl_sum = kl_isotropic(mu, log_var).item() * m_clusters
    true_ratio = (kl_sum / (n_nodes * n_nodes)) / recon_mean
    return (kl_term / recon_mean) / true_ratio


def test_per_jet_effective_beta_is_invariant_to_graph_size() -> None:
    """`per_jet`'s effective beta must equal the requested beta for every N.

    Note what this does *not* claim: the raw KL:recon ratio still varies with
    N under `per_jet`, because a true ELBO's does. See `_beta_eff`.
    """
    beta, m = 0.5, 4
    for n_nodes in (10, 40, 139):  # the real jet range
        assert _beta_eff(n_nodes, m, beta, "per_jet") == pytest.approx(beta, rel=1e-6)


def test_legacy_effective_beta_scales_with_n_squared_over_m() -> None:
    """Characterizes what `legacy` does, so the two modes are contrasted
    directly rather than the new one asserted in isolation.

    `legacy` divides KL by M and averages reconstruction over N^2, so its
    effective beta is `beta*N^2/M` -- it holds the *raw* KL:recon ratio
    roughly constant across graph sizes instead, which is a coherent
    convention but is not a per-graph ELBO and gives `beta` no standard
    meaning.
    """
    beta, m = 0.5, 4
    for n_nodes in (10, 40, 139):
        expected = beta * n_nodes * n_nodes / m
        assert _beta_eff(n_nodes, m, beta, "legacy") == pytest.approx(expected, rel=1e-6)


def test_per_jet_matches_true_per_graph_elbo_ratio() -> None:
    """`per_jet` must equal the true ELBO's KL:recon ratio, not merely be
    N-invariant -- a constant-but-wrong ratio would pass the test above."""
    beta, m, n_nodes = 0.5, 4, 40
    torch.manual_seed(7)
    recon_logits = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    adj_true = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    mu = torch.randn(m, D, dtype=torch.float64)
    log_var = torch.zeros(m, D, dtype=torch.float64)
    A_z = sym_adj(m).to(torch.float64)

    recon_mean = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=0.0, normalization="per_jet"
    ).item()
    _, kl_term = _elbo_parts(n_nodes, m, beta, "per_jet", seed=7)

    # True per-graph ELBO, scaled by 1/N^2: recon_sum/N^2 + beta*KL_sum/N^2.
    kl_sum = kl_isotropic(mu, log_var).item() * m
    expected_kl_term = beta * kl_sum / (n_nodes * n_nodes)
    assert kl_term == pytest.approx(expected_kl_term, rel=1e-5)
    assert recon_mean == pytest.approx(torch.tensor(2.0).log().item(), rel=1e-5)


def test_legacy_is_the_default_and_unchanged() -> None:
    """Backward compatibility (NFR-4): callers that predate T5.1 and pass no
    `normalization` must get byte-identical values to the explicit legacy
    path, so Phases 0-4 stay reproducible."""
    torch.manual_seed(11)
    n_nodes, m = 12, 4
    recon_logits = torch.randn(n_nodes, n_nodes)
    adj_true = (torch.rand(n_nodes, n_nodes) > 0.8).float()
    mu = torch.randn(m, D)
    log_var = torch.zeros(m, D)
    A_z = sym_adj(m)

    default = elbo(recon_logits, adj_true, mu, log_var, A_z, beta=0.3)
    explicit = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=0.3, normalization="legacy"
    )
    assert default.item() == explicit.item()


def test_normalizations_agree_when_kl_weight_is_zero() -> None:
    """The two modes differ only in the KL term; at beta=0 they must be
    identical, which pins the change to where it is claimed to be."""
    torch.manual_seed(13)
    n_nodes, m = 9, 4
    recon_logits = torch.randn(n_nodes, n_nodes)
    adj_true = (torch.rand(n_nodes, n_nodes) > 0.7).float()
    mu = torch.randn(m, D)
    log_var = torch.zeros(m, D)
    A_z = sym_adj(m)

    legacy = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=0.0, normalization="legacy"
    ).item()
    per_jet = elbo(
        recon_logits, adj_true, mu, log_var, A_z, beta=0.0, normalization="per_jet"
    ).item()
    assert legacy == pytest.approx(per_jet, rel=1e-7)


def test_per_jet_works_with_graph_mrf_prior() -> None:
    torch.manual_seed(17)
    n_nodes, m = 20, 4
    val = elbo(
        torch.randn(n_nodes, n_nodes),
        (torch.rand(n_nodes, n_nodes) > 0.8).float(),
        torch.randn(m, D),
        torch.zeros(m, D),
        sym_adj(m),
        beta=0.1,
        prior="graph_mrf",
        normalization="per_jet",
    )
    assert torch.isfinite(val)


def test_invalid_normalization_raises() -> None:
    n_nodes = 4
    with pytest.raises(ValueError, match="normalization must be one of"):
        elbo(
            torch.zeros(n_nodes, n_nodes),
            torch.zeros(n_nodes, n_nodes),
            torch.zeros(n_nodes, D),
            torch.zeros(n_nodes, D),
            torch.zeros(n_nodes, n_nodes),
            normalization="bad",
        )
