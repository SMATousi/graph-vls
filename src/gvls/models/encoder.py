from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GVLSEncoder(nn.Module):
    """Two-layer GCN variational encoder.

    Returns per-node (mu, log_var, z) where z is reparameterized at train time
    and equals mu at eval time.
    """

    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.mu_head = GCNConv(hidden_channels, latent_dim)
        self.log_var_head = GCNConv(hidden_channels, latent_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = torch.relu(self.conv1(x, edge_index))
        mu = self.mu_head(h, edge_index)
        log_var = self.log_var_head(h, edge_index).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
