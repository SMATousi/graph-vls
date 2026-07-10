from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gvls.models.encoder import GVLSEncoder
from gvls.models.gvls import LatentMessagePassing
from gvls.models.latent_graph import LatentGraphLearner


class LatentGraphPooling(nn.Module):
    """Learned node-count pooling (T3.6): DiffPool-style soft assignment.

    Maps N per-node Gaussians (mu, log_var) onto M << N cluster-level
    Gaussians via a learned row-softmax assignment S in [0,1]^(N x M). The
    pooled Gaussian per cluster is set by moment matching (mixture of
    Gaussians -> single Gaussian), not a naive average of means: the pooled
    variance folds in both the within-cluster variance and the variance of
    the means across the nodes assigned to that cluster (law of total
    variance). This requires column-normalised weights (each cluster's
    contributing nodes summing to 1), since S's rows -- not columns -- sum to
    1 by construction (row-softmax).
    """

    def __init__(self, latent_dim: int, num_clusters: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.assign = nn.Linear(latent_dim, num_clusters)

    def forward(self, z: Tensor, mu: Tensor, log_var: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (S, mu_pooled, log_var_pooled).

        S: (N, M) row-softmax assignment (rows sum to 1; used unnormalised
        for unpooling at decode time, where the row-sum-to-1 property is what
        keeps S @ (...) @ S.T a convex combination).
        mu_pooled, log_var_pooled: (M, d) pooled Gaussian parameters, computed
        with column-normalised weights so each cluster's parameters are a
        proper weighted average over its assigned nodes (not a raw sum,
        which would scale with cluster size).
        """
        s = torch.softmax(self.assign(z), dim=1)  # (N, M)

        col_sum = s.sum(dim=0).clamp(min=1e-8)  # (M,)
        w = s / col_sum.unsqueeze(0)              # (N, M), columns sum to 1

        var = log_var.exp()
        mu_pooled = w.T @ mu                                        # (M, d)
        second_moment = w.T @ (var + mu.pow(2))                      # (M, d)
        var_pooled = (second_moment - mu_pooled.pow(2)).clamp(min=1e-8)
        log_var_pooled = var_pooled.log()

        return s, mu_pooled, log_var_pooled


def assignment_entropy(s: Tensor) -> Tensor:
    """Mean per-node entropy of the assignment distribution S, in nats.

    Low entropy means each node's assignment is confidently concentrated on
    one or a few clusters (peaked); high entropy means it is spread thin
    across many clusters (diffuse, approaching uniform as entropy -> log M).
    Minimizing this as an auxiliary training loss is the standard DiffPool
    remedy for soft assignment collapsing to a near-uniform blur: with a
    diffuse S, unpooling (S @ ... @ S.T) averages away almost all structure,
    producing a near-constant reconstruction regardless of the underlying
    M x M latent graph -- observed empirically in the first T3.6 sweep before
    this regularizer was added (see specs/phase3/validation.md V-7).
    """
    return -(s * s.clamp(min=1e-12).log()).sum(dim=1).mean()


def assignment_link_loss(
    s: Tensor, adj_true: Tensor, pos_weight: float | None = None
) -> Tensor:
    """Auxiliary link-prediction loss on S (DiffPool, Ying et al. 2018).

    Compares S @ S.T -- an implied "same-cluster" probability for every node
    pair, already in [0,1] by Cauchy-Schwarz since each row of S is a
    probability distribution -- against the true input adjacency. This gives
    S a *direct* gradient signal from the real input graph.

    Without this, S's only gradient signal comes from the reconstruction/KL
    losses, which must travel through the entire pooled-graph pipeline
    (pool -> latent graph -> message passing -> unpool) and vanishes when S
    starts near-uniform: a near-uniform S makes the pooled M x M similarity
    matrix collapse to an almost-constant value, which makes the unpooled
    N x N reconstruction *exactly* constant (S's rows sum to 1), which in
    turn gives S no gradient to escape the collapse -- a self-reinforcing
    dead end observed empirically in the first T3.6 sweep (every grid point
    converged to a trivial always-predict-edge classifier, F1 stuck at
    exactly 2/3; see specs/phase3/validation.md V-7). This auxiliary loss
    breaks that deadlock by giving S a training signal that doesn't depend
    on anything downstream of it.

    The diagonal is excluded: S's self-similarity (s_i . s_i) is naturally
    close to 1 for a confident (peaked) assignment, but adj_true's diagonal
    is always 0 (no self-loops, matching this codebase's convention
    elsewhere, e.g. edge_compression_ratio). Including it would directly
    fight assignment_entropy, which wants high self-similarity.
    """
    n = s.size(0)
    same_cluster_prob = (s @ s.T).clamp(1e-6, 1 - 1e-6)
    off_diag = ~torch.eye(n, device=s.device, dtype=torch.bool)
    pred = same_cluster_prob[off_diag]
    true = adj_true[off_diag]
    if pos_weight is None:
        return F.binary_cross_entropy(pred, true)
    weight = 1.0 + (pos_weight - 1.0) * true
    return F.binary_cross_entropy(pred, true, weight=weight)


class PooledGVLS(nn.Module):
    """GVLS variant with learned node-count pooling (T3.6).

    Forward pass:
      1. GNN encoder over the N input nodes  →  (mu, log_var, z)
      2. LatentGraphPooling  →  assignment S (N x M), pooled (mu_p, log_var_p)
      3. Reparameterize the pooled Gaussian  →  z_p (M x d)
      4. Latent graph learner (Phase 1, reused unchanged) over the M pooled
         nodes  →  A_z (M x M)
      5. Latent message passing (Phase 1, reused unchanged) over A_z  →  z_tilde_p
      6. Unpool at decode time via the same S:
         recon_logits (N x N) = S @ (z_tilde_p @ z_tilde_p.T) @ S.T

    Unlike GVLS, this always returns unpooled N x N reconstruction logits
    (pre-sigmoid), since fidelity is judged against the original N x N
    adjacency regardless of M -- consistent with GVLS's convention elsewhere
    in this codebase of working with logits, not probabilities, everywhere
    (see gvls.losses.elbo.elbo, gvls.eval.compression.reconstruction_f1).
    """

    def __init__(
        self,
        encoder: GVLSEncoder,
        pooling: LatentGraphPooling,
        latent_graph_learner: LatentGraphLearner,
        latent_dim: int,
        mp_rounds: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.latent_graph_learner = latent_graph_learner
        self.mp = LatentMessagePassing(latent_dim, mp_rounds)

    def _reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_var, z = self.encoder(x, edge_index)
        s, mu_p, log_var_p = self.pooling(z, mu, log_var)
        z_p = self._reparameterize(mu_p, log_var_p)
        a_z = self.latent_graph_learner(z_p)
        z_tilde_p = self.mp(z_p, a_z)
        recon_logits = s @ (z_tilde_p @ z_tilde_p.T) @ s.T  # (N, N)
        return mu_p, log_var_p, z_p, a_z, z_tilde_p, s, recon_logits
