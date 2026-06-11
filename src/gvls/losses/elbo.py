from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_PRIORS = {"isotropic", "graph_mrf"}


def kl_isotropic(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL(q || N(0,I)) for a diagonal Gaussian q = N(mu, diag(exp(log_var))).

    Returns a non-negative scalar equal to 0 when q equals the prior.
    """
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum()


def kl_graph_mrf(mu: Tensor, log_var: Tensor, A_z: Tensor, lambda_: float = 1.0) -> Tensor:
    """KL(q || Gaussian MRF) with precision Ω = I + λ·L_z.

    L_z is the graph Laplacian of A_z.  The d latent dimensions are treated as
    independent draws from the same MRF prior, so the total KL is d times the
    single-dimension KL.

    The log-det term is computed with A_z detached from the computational graph
    to avoid second-order gradients through the Laplacian determinant.
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

    return 0.5 * (trace_term + mu_quad - N * d + logdet_omega - log_var.sum())


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
