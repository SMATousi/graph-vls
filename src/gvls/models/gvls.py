from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gvls.models.encoder import GVLSEncoder
from gvls.models.latent_graph import LatentGraphLearner


class GVLS(nn.Module):
    """Graph Variational Latent Space model.

    Forward pass:
      1. GNN encoder  →  (mu, log_var, z)
      2. Latent graph learner  →  A_z
      3. L rounds of degree-normalised message passing over A_z  →  z_tilde
         z_tilde = ReLU(D^{-1} A_z z_tilde W),  z_tilde_0 = z
      4. Return (mu, log_var, z, A_z, z_tilde)

    mp_rounds=0 is valid: z_tilde == z (message passing skipped).
    """

    def __init__(
        self,
        encoder: GVLSEncoder,
        latent_graph_learner: LatentGraphLearner,
        latent_dim: int,
        mp_rounds: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_graph_learner = latent_graph_learner
        self.mp_rounds = mp_rounds
        if mp_rounds > 0:
            # Shared weight across rounds, initialised as identity
            self.mp_weight = nn.Parameter(torch.eye(latent_dim))

    def forward(
        self, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_var, z = self.encoder(x, edge_index)
        A_z = self.latent_graph_learner(z)

        z_tilde = z
        for _ in range(self.mp_rounds):
            deg = A_z.sum(dim=1).clamp(min=1e-8)   # (N,)
            D_inv = (1.0 / deg).unsqueeze(-1)        # (N, 1)
            # Residual connection: z_tilde = z_tilde + aggregate(A_z, z_tilde) @ W
            # Direct gradient path to the encoder regardless of A_z quality.
            # No activation so z_tilde stays signed for the inner-product decoder.
            agg = (D_inv * (A_z @ z_tilde)) @ self.mp_weight
            z_tilde = z_tilde + agg

        return mu, log_var, z, A_z, z_tilde
