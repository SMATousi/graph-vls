import numpy as np
import torch

from gvls.compression.jet_sweep import (
    build_pooled_gvls,
    jet_loss,
    jet_pos_weight,
    train_pooled_gvls_on_jets,
)
from gvls.data.jets import NUM_FEATURES, PDGIDS, build_jet_graph

IN_CHANNELS = NUM_FEATURES
LATENT_DIM = 4
HIDDEN = 8
M = 4
K = 2


def _synthetic_jet(n: int, seed: int, center: float = 4.0):
    rng = np.random.default_rng(seed)
    pt = rng.uniform(0.5, 50.0, size=n)
    y = rng.normal(0.0, 0.3, size=n)
    phi = rng.normal(center, 0.3, size=n)
    pdgid = rng.choice(PDGIDS, size=n)
    particles = np.stack([pt, y, phi, pdgid], axis=1)
    return build_jet_graph(particles, label=seed % 2, k_graph_cap=8)


def _base_cfg() -> dict:
    return {
        "hidden_dim": HIDDEN,
        "mp_rounds": 1,
        "graph_method": "attention",
        "prior": "isotropic",
        "beta": 0.001,
        "lambda_": 1.0,
        "lr": 0.01,
    }


def _make_model():
    torch.manual_seed(0)
    return build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())


# ── Gradient flow (T4.2 core risk) ──────────────────────────────────────────

def test_gradient_flows_to_all_submodules_from_one_jet() -> None:
    model = _make_model()
    jet = _synthetic_jet(12, seed=1)
    device = torch.device("cpu")

    loss = jet_loss(model, jet, _base_cfg(), device, entropy_weight=0.1, aux_link_weight=5.0)
    loss.backward()

    assert model.encoder.conv1.lin.weight.grad is not None
    assert model.encoder.conv1.lin.weight.grad.abs().sum() > 0
    assert model.pooling.assign.weight.grad is not None
    assert model.pooling.assign.weight.grad.abs().sum() > 0

    # The default "attention" latent-graph-learner method has no learnable
    # parameters of its own (confirmed against "fgp"'s log_tau / "nri"'s MLP),
    # so its contribution is checked via the FGP method instead, which does.
    fgp_model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, {**_base_cfg(), "graph_method": "fgp"})
    fgp_loss = jet_loss(fgp_model, jet, _base_cfg(), device, entropy_weight=0.1, aux_link_weight=5.0)
    fgp_loss.backward()
    log_tau_grad = fgp_model.latent_graph_learner.log_tau.grad
    assert log_tau_grad is not None and log_tau_grad.abs().sum() > 0


def test_gradient_accumulation_matches_batched_mean() -> None:
    """Summing per-jet (loss/B).backward() must equal one backward() on the
    batch mean loss -- the actual correctness claim behind T4.2's per-jet
    gradient-accumulation loop (plan.md Design Decision 7)."""
    device = torch.device("cpu")
    jets = [_synthetic_jet(10 + i, seed=i) for i in range(3)]
    cfg = _base_cfg()

    model_a = _make_model()
    model_a.train()
    for jet in jets:
        loss = jet_loss(model_a, jet, cfg, device, entropy_weight=0.1, aux_link_weight=5.0)
        (loss / len(jets)).backward()
    grad_a = model_a.pooling.assign.weight.grad.clone()

    model_b = _make_model()
    model_b.train()
    total = sum(
        jet_loss(model_b, jet, cfg, device, entropy_weight=0.1, aux_link_weight=5.0)
        for jet in jets
    )
    (total / len(jets)).backward()
    grad_b = model_b.pooling.assign.weight.grad.clone()

    assert torch.allclose(grad_a, grad_b, atol=1e-6)


# ── No cross-jet leakage ─────────────────────────────────────────────────────

def test_same_jet_gives_identical_output_regardless_of_other_jets_processed() -> None:
    """A jet's forward output must depend only on its own (x, edge_index) --
    never on whatever other jet the model happened to process before it."""
    model = _make_model()
    model.eval()
    jet_a = _synthetic_jet(9, seed=10, center=1.0)
    jet_b = _synthetic_jet(15, seed=11, center=5.0)  # very different feature range

    with torch.no_grad():
        *_ , recon_a_first = model(jet_a.x, jet_a.edge_index)
        *_ , _recon_b = model(jet_b.x, jet_b.edge_index)
        *_ , recon_a_second = model(jet_a.x, jet_a.edge_index)

    assert torch.equal(recon_a_first, recon_a_second)


def test_disjoint_feature_ranges_do_not_mix() -> None:
    """Two jets with wildly different feature ranges, processed as a
    'minibatch' (sequential calls, no shared tensor), must not influence
    each other's assignment/reconstruction."""
    model = _make_model()
    model.eval()
    jet_low = _synthetic_jet(10, seed=20, center=0.5)
    jet_high = _synthetic_jet(10, seed=21, center=6.0)

    with torch.no_grad():
        *_, recon_low_alone = model(jet_low.x, jet_low.edge_index)
        *_, recon_high_alone = model(jet_high.x, jet_high.edge_index)
        # process interleaved, as a per-jet loop over a "batch" would
        *_, recon_low_batched = model(jet_low.x, jet_low.edge_index)
        *_, recon_high_batched = model(jet_high.x, jet_high.edge_index)

    assert torch.equal(recon_low_alone, recon_low_batched)
    assert torch.equal(recon_high_alone, recon_high_batched)


# ── Smoke test: tiny M grid, few jets, few epochs ───────────────────────────

def test_train_on_tiny_jet_set_completes_without_nan() -> None:
    jets = [_synthetic_jet(8 + i, seed=i) for i in range(6)]
    model = train_pooled_gvls_on_jets(
        jets,
        in_channels=IN_CHANNELS,
        latent_dim=LATENT_DIM,
        k=K,
        num_clusters=M,
        base_cfg=_base_cfg(),
        epochs=2,
        seed=42,
        device=torch.device("cpu"),
        batch_size=3,
    )
    model.eval()
    with torch.no_grad():
        for jet in jets:
            *_, recon_logits = model(jet.x, jet.edge_index)
            assert not torch.isnan(recon_logits).any()
            assert not torch.isinf(recon_logits).any()


def test_jet_pos_weight_matches_formula() -> None:
    jet = _synthetic_jet(10, seed=5)
    n = jet.num_nodes
    e = jet.edge_index.size(1)
    expected = (n * n - e) / e
    assert jet_pos_weight(jet) == expected


def test_jet_pos_weight_handles_isolated_single_node() -> None:
    jet = _synthetic_jet(1, seed=6)
    assert jet_pos_weight(jet) == 1.0
