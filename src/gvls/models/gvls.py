from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gvls.models.encoder import GVLSEncoder
from gvls.models.latent_graph import LatentGraphLearner


class LatentMessagePassing(nn.Module):
    """L rounds of degree-normalised residual message passing over A_z.

        z_tilde = z_tilde + D^{-1} A_z z_tilde W,  z_tilde_0 = z

    Shared weight across rounds, initialised as identity. No activation (the
    inner-product decoder needs z_tilde to stay signed). rounds=0 is valid:
    forward returns z unchanged.

    Extracted out of GVLS so PooledGVLS (T3.6) can reuse the exact same
    message-passing form over a smaller M-node A_z, without duplicating it.
    """

    def __init__(self, latent_dim: int, rounds: int = 1) -> None:
        super().__init__()
        self.rounds = rounds
        if rounds > 0:
            self.weight = nn.Parameter(torch.eye(latent_dim))

    def forward(self, z: Tensor, a_z: Tensor) -> Tensor:
        z_tilde = z
        for _ in range(self.rounds):
            deg = a_z.sum(dim=1).clamp(min=1e-8)   # (N,)
            D_inv = (1.0 / deg).unsqueeze(-1)        # (N, 1)
            # Residual connection: direct gradient path to the encoder
            # regardless of A_z quality.
            agg = (D_inv * (a_z @ z_tilde)) @ self.weight
            z_tilde = z_tilde + agg
        return z_tilde


class GVLS(nn.Module):
    """Graph Variational Latent Space model.

    Forward pass:
      1. GNN encoder  →  (mu, log_var, z)
      2. Latent graph learner  →  A_z
      3. L rounds of latent message passing over A_z  →  z_tilde
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
        self.mp = LatentMessagePassing(latent_dim, mp_rounds)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_var, z = self.encoder(x, edge_index)
        A_z = self.latent_graph_learner(z)
        z_tilde = self.mp(z, A_z)
        return mu, log_var, z, A_z, z_tilde
