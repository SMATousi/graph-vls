import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gvls.compression.jet_sweep import (
    JET_RESULT_FIELDS,
    build_pooled_gvls,
    evaluate_pooled_gvls_on_jets,
    jet_loss,
    jet_pos_weight,
    run_jet_compression_sweep,
    select_compression_optimal_m,
    train_pooled_gvls_on_jets,
)
from gvls.compression.sweep import write_results_csv
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


# ── Per-epoch callback + val-metric logging ─────────────────────────────────

def test_on_epoch_end_called_once_per_epoch() -> None:
    jets = [_synthetic_jet(20 + i, seed=i) for i in range(6)]
    calls: list[tuple[int, dict]] = []
    train_pooled_gvls_on_jets(
        jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
        base_cfg=_base_cfg(), epochs=3, seed=42, device=torch.device("cpu"), batch_size=3,
        on_epoch_end=lambda epoch, metrics: calls.append((epoch, metrics)),
    )
    assert [c[0] for c in calls] == [0, 1, 2]
    for _epoch, metrics in calls:
        assert "epoch" in metrics
        assert "train_loss" in metrics
        assert not torch.isnan(torch.tensor(metrics["train_loss"]))


def test_on_epoch_end_includes_val_metrics_when_eval_jets_given() -> None:
    train_jets = [_synthetic_jet(20 + i, seed=i) for i in range(6)]
    eval_jets = [_synthetic_jet(20 + i, seed=100 + i) for i in range(4)]
    calls: list[dict] = []
    train_pooled_gvls_on_jets(
        train_jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
        base_cfg=_base_cfg(), epochs=2, seed=42, device=torch.device("cpu"), batch_size=3,
        eval_jets=eval_jets, eval_every=1,
        on_epoch_end=lambda epoch, metrics: calls.append(metrics),
    )
    for metrics in calls:
        assert "val_avg_reconstruction_f1" in metrics
        assert 0.0 <= metrics["val_avg_reconstruction_f1"] <= 1.0
        assert "val_avg_bits_per_edge" in metrics
        # static/config fields shouldn't be re-logged every epoch
        assert "val_num_clusters" not in metrics
        assert "val_latent_dim" not in metrics


def test_eval_every_skips_non_eval_epochs_but_always_evals_last() -> None:
    train_jets = [_synthetic_jet(20 + i, seed=i) for i in range(6)]
    eval_jets = [_synthetic_jet(20 + i, seed=100 + i) for i in range(4)]
    calls: list[dict] = []
    train_pooled_gvls_on_jets(
        train_jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
        base_cfg=_base_cfg(), epochs=5, seed=42, device=torch.device("cpu"), batch_size=3,
        eval_jets=eval_jets, eval_every=3,
        on_epoch_end=lambda epoch, metrics: calls.append(metrics),
    )
    has_val = [i for i, m in enumerate(calls) if "val_avg_reconstruction_f1" in m]
    # epoch 0 (0 % 3 == 0) and epoch 4 (last epoch) must have eval'd; epoch 3
    # (3 % 3 == 0) also qualifies. Epochs 1, 2 must not.
    assert 0 in has_val
    assert 4 in has_val  # last epoch always evaluated regardless of eval_every
    assert 1 not in has_val
    assert 2 not in has_val


def test_no_eval_jets_means_no_val_metrics_and_no_crash() -> None:
    jets = [_synthetic_jet(20 + i, seed=i) for i in range(4)]
    calls: list[dict] = []
    train_pooled_gvls_on_jets(
        jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
        base_cfg=_base_cfg(), epochs=2, seed=42, device=torch.device("cpu"), batch_size=2,
        on_epoch_end=lambda epoch, metrics: calls.append(metrics),
    )
    for metrics in calls:
        assert not any(key.startswith("val_") for key in metrics)


def test_best_val_f1_checkpoint_is_restored_not_last_epoch() -> None:
    # F1 peaks at epoch 1, not the final epoch (2) -- the returned model must
    # carry epoch 1's weights, not whatever epoch 2 ended up at.
    train_jets = _jet_set(6, seed_offset=0)
    eval_jets = _jet_set(4, seed_offset=500)
    f1_sequence = [0.5, 0.9, 0.6]
    snapshots: list[dict] = []

    real_evaluate = evaluate_pooled_gvls_on_jets

    def fake_evaluate(model, *args, **kwargs):
        snapshots.append({name: t.clone() for name, t in model.state_dict().items()})
        metrics = real_evaluate(model, *args, **kwargs)
        metrics["avg_reconstruction_f1"] = f1_sequence[len(snapshots) - 1]
        return metrics

    with patch(
        "gvls.compression.jet_sweep.evaluate_pooled_gvls_on_jets", side_effect=fake_evaluate
    ):
        model = train_pooled_gvls_on_jets(
            train_jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
            base_cfg=_base_cfg(), epochs=3, seed=42, device=torch.device("cpu"), batch_size=3,
            eval_jets=eval_jets, eval_every=1,
        )

    assert len(snapshots) == 3
    best_snapshot, last_snapshot = snapshots[1], snapshots[2]
    returned = model.state_dict()
    for name, tensor in best_snapshot.items():
        assert torch.allclose(returned[name], tensor)
    assert any(
        not torch.allclose(returned[name], last_snapshot[name]) for name in last_snapshot
    )


def test_no_eval_jets_still_returns_last_epoch_model() -> None:
    # No validation signal exists to select a "best" epoch from -- behavior
    # must stay exactly what it was before best-val-F1 tracking existed.
    jets = [_synthetic_jet(20 + i, seed=i) for i in range(6)]
    torch.manual_seed(123)
    model = train_pooled_gvls_on_jets(
        jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K, num_clusters=M,
        base_cfg=_base_cfg(), epochs=2, seed=42, device=torch.device("cpu"), batch_size=3,
    )
    model.eval()
    with torch.no_grad():
        for jet in jets:
            *_, recon_logits = model(jet.x, jet.edge_index)
            assert not torch.isnan(recon_logits).any()


def test_jet_pos_weight_matches_formula() -> None:
    jet = _synthetic_jet(10, seed=5)
    n = jet.num_nodes
    e = jet.edge_index.size(1)
    expected = (n * n - e) / e
    assert jet_pos_weight(jet) == expected


def test_jet_pos_weight_handles_isolated_single_node() -> None:
    jet = _synthetic_jet(1, seed=6)
    assert jet_pos_weight(jet) == 1.0


# ── T4.3: jet-level compression sweep over M ────────────────────────────────

def _jet_set(n_jets: int, seed_offset: int = 0):
    # n well above k_graph_cap=8 so the k-NN graph isn't complete (a complete
    # graph has zero non-edges, which evaluate_pooled_gvls_on_jets correctly
    # skips for F1 -- using only complete graphs here would make every F1
    # skipped and the test wouldn't exercise that path at all).
    return [_synthetic_jet(20 + (i % 10), seed=seed_offset + i) for i in range(n_jets)]


def test_evaluate_pooled_gvls_on_jets_returns_all_fields() -> None:
    train_jets = _jet_set(6, seed_offset=0)
    eval_jets = _jet_set(4, seed_offset=100)
    model = train_pooled_gvls_on_jets(
        train_jets, in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K,
        num_clusters=M, base_cfg=_base_cfg(), epochs=2, seed=42, device=torch.device("cpu"),
        batch_size=3,
    )
    metrics = evaluate_pooled_gvls_on_jets(
        model, eval_jets, num_clusters=M, latent_dim=LATENT_DIM, k=K,
        num_features=IN_CHANNELS, f1_negative_ratio=1.0, seed=42, device=torch.device("cpu"),
    )
    assert set(JET_RESULT_FIELDS) - {"dataset"} <= set(metrics.keys())
    assert metrics["num_eval_jets"] == 4
    assert 0.0 <= metrics["avg_reconstruction_f1"] <= 1.0
    assert metrics["avg_bits_per_edge"] >= 0.0


def test_evaluate_pooled_gvls_on_jets_skips_single_particle_jets() -> None:
    model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())
    jets = [_synthetic_jet(1, seed=1), _synthetic_jet(9, seed=2)]
    metrics = evaluate_pooled_gvls_on_jets(
        model, jets, num_clusters=M, latent_dim=LATENT_DIM, k=K,
        num_features=IN_CHANNELS, f1_negative_ratio=1.0, seed=42, device=torch.device("cpu"),
    )
    assert metrics["num_eval_jets"] == 1


def test_select_compression_optimal_m_picks_smallest_within_tolerance() -> None:
    rows = [
        {"num_clusters": 4, "avg_reconstruction_f1": 0.80},
        {"num_clusters": 6, "avg_reconstruction_f1": 0.85},
        {"num_clusters": 8, "avg_reconstruction_f1": 0.86},
    ]
    best = select_compression_optimal_m(rows, tolerance=0.02)
    assert best["num_clusters"] == 6  # within 0.02 of best (0.86), M=4 (0.80) is not


def test_select_compression_optimal_m_falls_back_to_largest_if_none_close() -> None:
    rows = [
        {"num_clusters": 4, "avg_reconstruction_f1": 0.50},
        {"num_clusters": 8, "avg_reconstruction_f1": 0.90},
    ]
    best = select_compression_optimal_m(rows, tolerance=0.01)
    assert best["num_clusters"] == 8


def test_jet_compression_sweep_smoke_writes_rows() -> None:
    train_jets = _jet_set(6, seed_offset=0)
    eval_jets = _jet_set(4, seed_offset=200)

    rows = run_jet_compression_sweep(
        train_jets, eval_jets, m_grid=[4, 6],
        in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=K,
        base_cfg=_base_cfg(), epochs=2, seed=42, device=torch.device("cpu"), batch_size=3,
    )
    assert len(rows) == 2
    assert [r["num_clusters"] for r in rows] == [4, 6]
    for row in rows:
        assert set(JET_RESULT_FIELDS) <= set(row.keys())
        assert row["dataset"] == "qg_jets"

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "qg_jets_pooling.csv")
        write_results_csv(rows, csv_path, fieldnames=JET_RESULT_FIELDS)
        assert Path(csv_path).exists()
        content = Path(csv_path).read_text().strip().splitlines()
        assert len(content) == 3  # header + 2 rows


def test_jet_compression_sweep_clamps_k_to_m_minus_one() -> None:
    # k=5 would exceed M-1 at M=4; run_jet_compression_sweep must clamp it
    # rather than erroring inside LatentGraphLearner.
    train_jets = _jet_set(4, seed_offset=0)
    eval_jets = _jet_set(3, seed_offset=300)
    rows = run_jet_compression_sweep(
        train_jets, eval_jets, m_grid=[4],
        in_channels=IN_CHANNELS, latent_dim=LATENT_DIM, k=5,
        base_cfg=_base_cfg(), epochs=1, seed=42, device=torch.device("cpu"), batch_size=2,
    )
    assert rows[0]["k"] == 3


# ── T5.1: checkpoint-selection criterion (specs/phase5/) ─────────────────────


def _selection_jets(n_jets: int = 8):
    """Synthetic jets with both labels present, so probe_accuracy can fit."""
    return [_synthetic_jet(12 + (i % 5), seed=i) for i in range(n_jets)]


def test_selection_metric_defaults_to_reconstruction_f1() -> None:
    """Backward compatibility (NFR-4): the pre-T5.1 criterion stays the
    default, so existing callers keep selecting the same checkpoint."""
    import inspect

    sig = inspect.signature(train_pooled_gvls_on_jets)
    assert sig.parameters["selection_metric"].default == "reconstruction_f1"


def test_invalid_selection_metric_raises() -> None:
    with pytest.raises(ValueError, match="selection_metric must be one of"):
        train_pooled_gvls_on_jets(
            _selection_jets(4),
            in_channels=IN_CHANNELS,
            latent_dim=LATENT_DIM,
            k=K,
            num_clusters=M,
            base_cfg=_base_cfg(),
            epochs=1,
            seed=0,
            device=torch.device("cpu"),
            batch_size=2,
            show_progress=False,
            selection_metric="nonsense",
        )


@pytest.mark.parametrize("metric", ["reconstruction_f1", "val_loss", "probe_accuracy"])
def test_every_selection_metric_trains_and_logs_its_own_key(metric: str) -> None:
    """Each criterion must run end-to-end and surface the value it selected
    on in the per-epoch metrics, so a run's own logs record which signal
    chose its checkpoint (FR-1)."""
    jets = _selection_jets(8)
    seen: list[dict] = []
    model = train_pooled_gvls_on_jets(
        jets,
        in_channels=IN_CHANNELS,
        latent_dim=LATENT_DIM,
        k=K,
        num_clusters=M,
        base_cfg=_base_cfg(),
        epochs=2,
        seed=0,
        device=torch.device("cpu"),
        batch_size=4,
        show_progress=False,
        eval_jets=jets,
        on_epoch_end=lambda epoch, metrics: seen.append(metrics),
        selection_metric=metric,
    )
    assert model is not None
    assert len(seen) == 2
    expected_key = {
        "reconstruction_f1": "val_avg_reconstruction_f1",
        "val_loss": "val_loss",
        "probe_accuracy": "val_probe_accuracy",
    }[metric]
    assert expected_key in seen[-1]


def test_val_loss_selection_picks_the_lowest_not_the_highest() -> None:
    """`val_loss` is the one criterion where smaller is better; a
    direction-agnostic selection loop would silently keep the worst epoch."""
    from gvls.compression import jet_sweep

    jets = _selection_jets(6)
    losses = iter([5.0, 1.0, 9.0])  # best is epoch 1
    captured: dict = {}

    def fake_val_loss(model, *args, **kwargs):
        value = next(losses)
        captured.setdefault("order", []).append(value)
        # tag the model so we can tell which epoch's weights come back
        captured[value] = {k: v.clone() for k, v in model.state_dict().items()}
        return value

    with patch.object(jet_sweep, "validation_loss", side_effect=fake_val_loss):
        model = jet_sweep.train_pooled_gvls_on_jets(
            jets,
            in_channels=IN_CHANNELS,
            latent_dim=LATENT_DIM,
            k=K,
            num_clusters=M,
            base_cfg=_base_cfg(),
            epochs=3,
            seed=0,
            device=torch.device("cpu"),
            batch_size=3,
            show_progress=False,
            eval_jets=jets,
            selection_metric="val_loss",
        )

    assert captured["order"] == [5.0, 1.0, 9.0]
    best = captured[1.0]
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, best[name]), f"{name} is not the epoch-1 snapshot"


def test_probe_accuracy_is_bounded_and_finite() -> None:
    from gvls.compression.jet_sweep import probe_accuracy

    jets = _selection_jets(10)
    model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())
    score = probe_accuracy(model, jets, torch.device("cpu"), seed=0)
    assert 0.0 <= score <= 1.0


def test_probe_features_have_expected_width() -> None:
    """Feature layout must match jet_features_to_array's (flattened z_tilde,
    then A_z's strict upper triangle) -- this is a local reimplementation
    kept deliberately in sync (see its docstring)."""
    from gvls.compression.jet_sweep import jet_probe_features

    jets = _selection_jets(4)
    model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())
    x, y = jet_probe_features(model, jets, torch.device("cpu"))
    assert x.shape == (len(jets), M * LATENT_DIM + M * (M - 1) // 2)
    assert y.shape == (len(jets),)


def test_jet_loss_respects_normalization_from_base_cfg() -> None:
    """jet_loss must thread `normalization` through to elbo(), and default to
    legacy when a pre-Phase-5 config omits it."""
    jet = _synthetic_jet(14, seed=1)
    device = torch.device("cpu")
    model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())

    cfg_legacy = _base_cfg()
    cfg_default = _base_cfg()
    cfg_per_jet = {**_base_cfg(), "normalization": "per_jet"}
    cfg_legacy["normalization"] = "legacy"

    torch.manual_seed(0)
    legacy = jet_loss(model, jet, cfg_legacy, device, 0.1, 5.0).item()
    torch.manual_seed(0)
    default = jet_loss(model, jet, cfg_default, device, 0.1, 5.0).item()
    torch.manual_seed(0)
    per_jet = jet_loss(model, jet, cfg_per_jet, device, 0.1, 5.0).item()

    assert default == pytest.approx(legacy, rel=1e-7)
    assert per_jet != pytest.approx(legacy, rel=1e-9)
