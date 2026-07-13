import torch

from gvls.models.decoder import LatentGraphDecoder

N = 10
LATENT_DIM = 8


def sym_adj(n: int) -> torch.Tensor:
    """Random symmetric soft adjacency with zero diagonal."""
    A = torch.rand(n, n)
    A = (A + A.T) / 2
    A.fill_diagonal_(0.0)
    return A * 0.5


def test_output_shape() -> None:
    torch.manual_seed(0)
    z_tilde = torch.randn(N, LATENT_DIM)
    a_z = sym_adj(N)
    decoder = LatentGraphDecoder(LATENT_DIM)
    logits = decoder(z_tilde, a_z)
    assert logits.shape == (N, N)


def test_a_z_affects_output() -> None:
    """The whole point of this decoder: unlike the plain inner-product
    decoder, its output must depend on A_z even when z_tilde is fixed --
    otherwise it provides no benefit over mp_rounds=0 configs where A_z
    never reaches the reconstruction logits (specs/phase3/plan.md T3.4).
    """
    torch.manual_seed(1)
    z_tilde = torch.randn(N, LATENT_DIM)
    decoder = LatentGraphDecoder(LATENT_DIM)
    decoder.eval()

    a_z_1 = sym_adj(N)
    a_z_2 = sym_adj(N)
    with torch.no_grad():
        logits_1 = decoder(z_tilde, a_z_1)
        logits_2 = decoder(z_tilde, a_z_2)
    assert not torch.allclose(logits_1, logits_2)


def test_gradient_flows_to_z_tilde_and_decoder_weight() -> None:
    torch.manual_seed(2)
    z_tilde = torch.randn(N, LATENT_DIM, requires_grad=True)
    a_z = sym_adj(N)
    decoder = LatentGraphDecoder(LATENT_DIM)
    logits = decoder(z_tilde, a_z)
    logits.sum().backward()
    assert z_tilde.grad is not None and z_tilde.grad.abs().sum() > 0
    assert decoder.mp.weight.grad is not None and decoder.mp.weight.grad.abs().sum() > 0


def test_zero_a_z_reduces_to_identity_message_passing() -> None:
    """With A_z all-zero, degree is clamped to 1e-8 (LatentMessagePassing's
    convention), so the aggregation term vanishes and z_decode == z_tilde --
    consistent with the residual formula's behavior at mp_rounds=0.
    """
    torch.manual_seed(3)
    z_tilde = torch.randn(N, LATENT_DIM)
    a_z = torch.zeros(N, N)
    decoder = LatentGraphDecoder(LATENT_DIM)
    with torch.no_grad():
        logits = decoder(z_tilde, a_z)
    expected = z_tilde @ z_tilde.T
    assert torch.allclose(logits, expected, atol=1e-5)


def test_no_nans() -> None:
    torch.manual_seed(4)
    z_tilde = torch.randn(N, LATENT_DIM)
    a_z = sym_adj(N)
    decoder = LatentGraphDecoder(LATENT_DIM)
    logits = decoder(z_tilde, a_z)
    assert not torch.isnan(logits).any()
