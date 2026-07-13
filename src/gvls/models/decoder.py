from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from gvls.models.gvls import LatentMessagePassing


class LatentGraphDecoder(nn.Module):
    """A_z-conditioned decoder (T3.4, originally superseded 2026-07-09,
    revived 2026-07-13 after the ELBO KL-normalization fix left all three
    datasets' compression grids as flat plateaus below the 0.90 fidelity
    floor -- see specs/phase3/plan.md T3.4 and validation.md V-3/V-4).

    The plain inner-product decoder (z_tilde @ z_tilde.T) only sees A_z
    indirectly, through however many rounds of LatentMessagePassing the
    encoder's `mp_rounds` config already applied to produce z_tilde -- zero
    for 2 of the 3 benchmark datasets (CiteSeer, PubMed), whose Phase 2
    NAS-best configs use `mp_rounds=0`. This decoder gives A_z a *guaranteed*
    path into the reconstruction, independent of `mp_rounds`, by applying one
    unconditional extra round of the same residual, degree-normalized
    message-passing form used elsewhere in this codebase:

        z_decode = z_tilde + D^{-1} A_z z_tilde W
        recon_logits = z_decode @ z_decode.T

    This is exactly the "second message-passing round in the decoder"
    described in specs/phase3/requirements.md FR-4's original (superseded,
    now revived) plan -- reusing `LatentMessagePassing` with `rounds=1`
    rather than duplicating its formula.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.mp = LatentMessagePassing(latent_dim, rounds=1)

    def forward(self, z_tilde: Tensor, a_z: Tensor) -> Tensor:
        z_decode = self.mp(z_tilde, a_z)
        return z_decode @ z_decode.T
