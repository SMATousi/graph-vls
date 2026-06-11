from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_METHODS = {"attention", "fgp", "nri"}


class LatentGraphLearner(nn.Module):
    """Differentiable latent graph inference module.

    Accepts latent embeddings z ∈ ℝ^(N×d) and returns a sparse, symmetric
    soft adjacency A_z ∈ [0,1]^(N×N) with a zeroed diagonal.

    Pipeline for all methods:
      1. Compute raw scores S ∈ ℝ^(N×N)
      2. Exclude diagonal (set to -inf so it can never be in top-k)
      3. Top-k mask: keep the k highest-scoring neighbours per row
      4. sigmoid(S) * mask  →  A ∈ [0,1]^(N×N), at most k non-zeros per row
      5. Symmetrize: (A + A^T) / 2  →  A_z (up to 2k non-zeros per row)

    Gradient note: step 3 is a discrete selection; gradients flow through
    step 4 (straight-through for the top-k positions).

    Methods
    -------
    attention : scaled dot-product  S[i,j] = z_i·z_j / √d
    fgp       : cosine similarity / τ  (τ > 0 is a learned temperature)
    nri       : MLP on node-pair concatenations  (z_i ‖ z_j) → scalar
                Warning: allocates an O(N²·2d) tensor — avoid for N > 3000.
    """

    def __init__(self, latent_dim: int, method: str = "attention", k: int = 10) -> None:
        super().__init__()
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
        self.latent_dim = latent_dim
        self.method = method
        self.k = k

        if method == "fgp":
            # log_tau initialised to 0  →  tau = exp(0) = 1
            self.log_tau = nn.Parameter(torch.zeros(1))
        elif method == "nri":
            self.nri_mlp = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )

    def forward(self, z: Tensor) -> Tensor:
        N = z.size(0)

        if self.method == "attention":
            scores = z @ z.T / (self.latent_dim**0.5)
        elif self.method == "fgp":
            z_norm = F.normalize(z, p=2, dim=-1)
            scores = z_norm @ z_norm.T / self.log_tau.exp()
        else:  # nri
            zi = z.unsqueeze(1).expand(N, N, -1)
            zj = z.unsqueeze(0).expand(N, N, -1)
            scores = self.nri_mlp(torch.cat([zi, zj], dim=-1)).squeeze(-1)

        return self._sparsify_and_symmetrize(scores)

    def _sparsify_and_symmetrize(self, scores: Tensor) -> Tensor:
        N = scores.size(0)

        # Symmetrize scores so sigmoid values are consistent in both directions
        scores_sym = (scores + scores.T) / 2

        # Exclude diagonal from neighbour selection
        eye = torch.eye(N, device=scores.device, dtype=torch.bool)
        scores_nd = scores_sym.masked_fill(eye, float("-inf"))

        # Top-k per row (non-differentiable selection; gradients flow through values)
        k = min(self.k, N - 1)
        topk_idx = scores_nd.topk(k, dim=1).indices  # (N, k)
        mask = torch.zeros(N, N, device=scores.device, dtype=scores.dtype)
        mask.scatter_(1, topk_idx, 1.0)

        # Union symmetrisation: edge exists if either node selected the other.
        # Each row has at most k non-zeros before symmetrisation; after, the mean
        # is ≈ 2k (both directions) though individual rows can exceed k.
        A = torch.sigmoid(scores_nd) * mask
        return (A + A.T) / 2
