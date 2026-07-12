from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_PRIORS = {"isotropic", "graph_mrf"}


def kl_isotropic(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL(q || N(0,I)) for a diagonal Gaussian q = N(mu, diag(exp(log_var))).

    Returns a non-negative scalar equal to 0 when q equals the prior.

    Normalized by node count (mean over the leading dimension of mu/log_var,
    summed only over the latent dims) so the returned magnitude reflects a
    per-node KL cost, independent of how many nodes it's computed over. This
    matters because `elbo()`'s reconstruction term is mean-reduced over all
    node-pair logits (so its magnitude is independent of N/M), while this KL
    term used to be a raw, un-normalized sum -- meaning beta*KL's absolute
    size scaled linearly with the number of nodes/clusters it was computed
    over. That mismatch was fine for a fixed N, but T3.6's node-count pooling
    sweep varies M over almost two orders of magnitude (169-9858), which
    exposed it: PubMed's largest pool sizes drove beta*KL to 12-24x the scale
    of recon_loss, pulling mu toward the prior and collapsing the pooled
    reconstruction to a near-constant matrix, on top of the already-known
    assignment-collapse failure mode. See specs/phase3/validation.md V-8.
    """
    n = mu.size(0)
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum() / n


def kl_graph_mrf(mu: Tensor, log_var: Tensor, A_z: Tensor, lambda_: float = 1.0) -> Tensor:
    """KL(q || Gaussian MRF) with precision Ω = I + λ·L_z.

    L_z is the graph Laplacian of A_z.  The d latent dimensions are treated as
    independent draws from the same MRF prior, so the total KL is d times the
    single-dimension KL.

    The log-det term is computed with A_z detached from the computational graph
    to avoid second-order gradients through the Laplacian determinant.

    Normalized by node count (divides the final scalar by N) for the same
    reason as `kl_isotropic` -- see its docstring and
    specs/phase3/validation.md V-8. Without this, the un-normalized joint-KL
    formula (natural for treating the whole graph as one sample, but O(N) in
    magnitude) scaled linearly with N/M while `elbo()`'s reconstruction term
    is mean-reduced and O(1), so beta's effective strength implicitly grew
    with graph/cluster size -- most visible for PubMed, whose NAS-best config
    uses graph_mrf with the largest beta of the three benchmark datasets.
    """
    N, d = mu.shape
    device = mu.device

    deg = A_z.sum(dim=1)                            # (N,)
    L_z = torch.diag(deg) - A_z                     # (N, N)
    Omega = torch.eye(N, device=device) + lambda_ * L_z  # (N, N)

    # tr(Ω Σ_q) = diag(Ω) · σ²  (diagonal covariance, summed over all i and d)
    trace_term = (Omega.diagonal().unsqueeze(1) * log_var.exp()).sum()

    # μᵀ Ω μ summed over d dimensions
    mu_quad = (Omega @ mu).mul(mu).sum()

    # log det(Ω): detach A_z to stay first-order
    deg_d = A_z.detach().sum(dim=1)
    Omega_d = torch.eye(N, device=device) + lambda_ * (torch.diag(deg_d) - A_z.detach())
    _, logdet = torch.linalg.slogdet(Omega_d)
    logdet_omega = d * logdet                        # same prior for every dimension

    return 0.5 * (trace_term + mu_quad - N * d + logdet_omega - log_var.sum()) / N


def elbo(
    recon_logits: Tensor,
    adj_true: Tensor,
    mu: Tensor,
    log_var: Tensor,
    A_z: Tensor,
    beta: float = 1.0,
    lambda_: float = 1.0,
    prior: str = "isotropic",
    pos_weight: float | None = None,
) -> Tensor:
    """VAE training loss: recon_loss + β·KL  (minimise this).

    recon_logits : (N, N) inner-product scores  z̃_i · z̃_j  for all node pairs
    adj_true     : (N, N) dense binary adjacency used for training
    prior        : 'isotropic' or 'graph_mrf'
    pos_weight   : optional scalar upweight for positive edges in BCE; pass
                   (N*N - n_edges) / n_edges to balance class imbalance in
                   sparse graphs (VGAE convention)

    Raises RuntimeError if the computed loss is NaN.
    """
    if prior not in _PRIORS:
        raise ValueError(f"prior must be one of {_PRIORS}, got '{prior}'")

    pw = (
        torch.tensor(pos_weight, dtype=recon_logits.dtype, device=recon_logits.device)
        if pos_weight is not None
        else None
    )
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits, adj_true, pos_weight=pw, reduction="mean"
    )

    kl = (
        kl_isotropic(mu, log_var)
        if prior == "isotropic"
        else kl_graph_mrf(mu, log_var, A_z, lambda_)
    )

    loss = recon_loss + beta * kl

    if torch.isnan(loss):
        raise RuntimeError("NaN loss detected — check hyperparameters")

    return loss
