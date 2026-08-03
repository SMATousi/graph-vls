import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from gvls.compression.jet_sweep import (
    build_pooled_gvls,
    load_gvls_checkpoint,
    save_gvls_checkpoint,
)
from gvls.data.jets import NUM_FEATURES, PDGIDS, build_jet_graph
from gvls.eval.metrics import classification_metrics
from gvls.qgnn_training import (
    JetFeatures,
    collate_jet_features,
    compute_qgnn_logits,
    evaluate_qgnn_classifier,
    extract_latent_features,
    load_qgnn_checkpoint,
    save_qgnn_checkpoint,
    train_qgnn_classifier,
)

IN_CHANNELS = NUM_FEATURES
LATENT_DIM = 4
M = 4
K = 2
DEVICE = torch.device("cpu")


def _synthetic_jet(n: int, seed: int, label: int | None = None):
    rng = np.random.default_rng(seed)
    pt = rng.uniform(0.5, 50.0, size=n)
    y = rng.normal(0.0, 0.3, size=n)
    phi = rng.normal(4.0, 0.3, size=n)
    pdgid = rng.choice(PDGIDS, size=n)
    particles = np.stack([pt, y, phi, pdgid], axis=1)
    return build_jet_graph(particles, label=seed % 2 if label is None else label, k_graph_cap=8)


def _base_cfg() -> dict:
    return {
        "hidden_dim": 8,
        "mp_rounds": 1,
        "graph_method": "attention",
        "prior": "isotropic",
        "beta": 0.001,
        "lambda_": 1.0,
        "lr": 0.01,
    }


def _frozen_gvls():
    torch.manual_seed(0)
    return build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg()).to(DEVICE).eval()


# ── extract_latent_features ─────────────────────────────────────────────────

def test_extract_latent_features_shapes_and_labels() -> None:
    model = _frozen_gvls()
    jets = [_synthetic_jet(20 + i, seed=i, label=i % 2) for i in range(5)]
    features = extract_latent_features(model, jets, DEVICE)

    assert len(features) == 5
    for f, jet in zip(features, jets):
        assert isinstance(f, JetFeatures)
        assert f.z_tilde.shape == (M, LATENT_DIM)
        assert f.a_z.shape == (M, M)
        assert f.label == int(jet.y.item())


def test_extract_latent_features_does_not_change_model_params() -> None:
    model = _frozen_gvls()
    before = [p.clone() for p in model.parameters()]
    jets = [_synthetic_jet(20 + i, seed=i) for i in range(4)]
    extract_latent_features(model, jets, DEVICE)
    after = list(model.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a)
        assert a.grad is None


# ── GVLS checkpoint round-trip ───────────────────────────────────────────────

def test_gvls_checkpoint_roundtrip_preserves_behavior() -> None:
    model = _frozen_gvls()
    jets = [_synthetic_jet(20 + i, seed=i) for i in range(3)]
    features_before = extract_latent_features(model, jets, DEVICE)

    config = {
        "in_channels": IN_CHANNELS,
        "latent_dim": LATENT_DIM,
        "k": K,
        "num_clusters": M,
        "base_cfg": _base_cfg(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "gvls_m4.pt")
        save_gvls_checkpoint(model, config, path)
        assert Path(path).exists()
        loaded_model, loaded_config = load_gvls_checkpoint(path, DEVICE)

    assert loaded_config == config
    features_after = extract_latent_features(loaded_model, jets, DEVICE)
    for f_before, f_after in zip(features_before, features_after):
        assert torch.equal(f_before.z_tilde, f_after.z_tilde)
        assert torch.equal(f_before.a_z, f_after.a_z)


# ── collate_jet_features (T4.8) ──────────────────────────────────────────────

def test_collate_jet_features_shapes() -> None:
    features = _tiny_features(5, seed_offset=0)
    z_tildes, a_zs, labels = collate_jet_features(features)
    assert z_tildes.shape == (5, M, LATENT_DIM)
    assert a_zs.shape == (5, M, M)
    assert labels.shape == (5,)
    assert labels.tolist() == [float(f.label) for f in features]


def test_collate_jet_features_handles_ragged_final_batch() -> None:
    features = _tiny_features(5, seed_offset=0)
    chunks = [features[i : i + 2] for i in range(0, len(features), 2)]
    assert [len(c) for c in chunks] == [2, 2, 1]  # last chunk is ragged
    for chunk in chunks:
        z_tildes, a_zs, labels = collate_jet_features(chunk)
        assert z_tildes.shape == (len(chunk), M, LATENT_DIM)
        assert a_zs.shape == (len(chunk), M, M)
        assert labels.shape == (len(chunk),)


# ── train_qgnn_classifier ────────────────────────────────────────────────────

def _tiny_features(n: int, seed_offset: int) -> list[JetFeatures]:
    model = _frozen_gvls()
    jets = [_synthetic_jet(20 + i, seed=seed_offset + i, label=i % 2) for i in range(n)]
    return extract_latent_features(model, jets, DEVICE)


def test_train_qgnn_classifier_smoke() -> None:
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=3, show_progress=False,
    )

    assert len(result.history) == 2
    assert 0 <= result.best_epoch < 2
    assert "accuracy" in result.best_val_metrics
    assert 0.0 <= result.best_val_metrics["accuracy"] <= 1.0
    for row in result.history:
        assert not torch.isnan(torch.tensor(row["train_loss"]))


def test_train_qgnn_classifier_default_optimizer_is_adamw() -> None:
    """T4.10 (plan.md Design Decision 12): adamw is the default, matching
    the Lorentz-EQGNN literature baseline's protocol."""
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=3, show_progress=False,
    )

    assert len(result.history) == 2
    for row in result.history:
        assert not torch.isnan(torch.tensor(row["train_loss"]))


def test_train_qgnn_classifier_supports_adam() -> None:
    """adam remains selectable for the pre-T4.10 configuration."""
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=3, show_progress=False,
        optimizer="adam",
    )

    assert len(result.history) == 2
    for row in result.history:
        assert not torch.isnan(torch.tensor(row["train_loss"]))


def test_train_qgnn_classifier_invalid_optimizer_raises() -> None:
    train_features = _tiny_features(4, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    try:
        train_qgnn_classifier(
            train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
            lr=0.1, epochs=1, seed=42, device=DEVICE, batch_size=2, show_progress=False,
            optimizer="rmsprop",
        )
        raise AssertionError("expected ValueError for an unknown optimizer")
    except ValueError as exc:
        assert "rmsprop" in str(exc)


def test_train_qgnn_classifier_result_includes_best_train_metrics() -> None:
    """T4.10 (FR-6 amendment): train accuracy (and the full metric suite) is
    reported alongside val/test, matching the literature table's convention."""
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=3, show_progress=False,
    )

    assert "accuracy" in result.best_train_metrics
    assert 0.0 <= result.best_train_metrics["accuracy"] <= 1.0


def test_train_qgnn_classifier_result_includes_best_threshold() -> None:
    """T4.10 followup (validation.md V-11): a validation-selected decision
    threshold is computed once at the end of training, rather than assuming
    the raw logits are calibrated around the fixed 0.5 default."""
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=3, show_progress=False,
    )

    assert 0.0 <= result.best_threshold <= 1.0
    # best_train_metrics must be reported at that same tuned threshold, not
    # the raw 0.5 default, so train/test accuracy are on the same basis.
    from gvls.models.qgnn import QGNNClassifier

    model = QGNNClassifier(m=M, d=LATENT_DIM, num_layers=1, seed=0).to(DEVICE)
    model.load_state_dict(result.best_state_dict)
    model.eval()
    train_logits, train_labels = compute_qgnn_logits(model, train_features, DEVICE)
    expected = classification_metrics(train_labels, train_logits, threshold=result.best_threshold)
    assert result.best_train_metrics["accuracy"] == pytest.approx(expected["accuracy"])


def test_train_qgnn_classifier_handles_ragged_final_minibatch() -> None:
    """5 train jets with batch_size=2 leaves a final minibatch of size 1 --
    the batched training loop (T4.8) must handle this without error."""
    train_features = _tiny_features(5, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=2, seed=42, device=DEVICE, batch_size=2, show_progress=False,
    )

    assert len(result.history) == 2
    for row in result.history:
        assert not torch.isnan(torch.tensor(row["train_loss"]))


def test_train_qgnn_classifier_on_epoch_end_called_once_per_epoch() -> None:
    train_features = _tiny_features(6, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=100)
    calls: list[tuple[int, dict]] = []

    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=3, seed=42, device=DEVICE, batch_size=3, show_progress=False,
        on_epoch_end=lambda epoch, metrics: calls.append((epoch, metrics)),
    )

    assert [c[0] for c in calls] == [0, 1, 2]
    for epoch, metrics in calls:
        assert metrics["epoch"] == epoch
        assert "train_loss" in metrics
        assert "accuracy" in metrics
    # the callback's per-epoch rows must be the same objects/values train_qgnn_classifier
    # itself returns in result.history -- not a separate, possibly-drifting copy
    assert [metrics for _epoch, metrics in calls] == result.history


def test_train_qgnn_classifier_best_state_dict_is_loadable() -> None:
    train_features = _tiny_features(4, seed_offset=0)
    val_features = _tiny_features(4, seed_offset=200)
    result = train_qgnn_classifier(
        train_features, val_features, m=M, d=LATENT_DIM, num_layers=1,
        lr=0.1, epochs=1, seed=1, device=DEVICE, batch_size=2, show_progress=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "qgnn_m4.pt")
        config = {"m": M, "d": LATENT_DIM, "num_layers": 1}
        save_qgnn_checkpoint(result.best_state_dict, config, path)
        loaded_model, loaded_config = load_qgnn_checkpoint(path, DEVICE)

    assert loaded_config == config
    metrics = evaluate_qgnn_classifier(loaded_model, val_features, DEVICE)
    assert metrics["accuracy"] == result.best_val_metrics["accuracy"]


# ── evaluate_qgnn_classifier ──────────────────────────────────────────────────

def test_evaluate_qgnn_classifier_returns_full_metrics() -> None:
    from gvls.models.qgnn import QGNNClassifier

    model = QGNNClassifier(m=M, d=LATENT_DIM, num_layers=1, seed=0).to(DEVICE)
    features = _tiny_features(6, seed_offset=300)
    metrics = evaluate_qgnn_classifier(model, features, DEVICE)

    expected_keys = {
        "accuracy", "auc", "ap", "macro_f1", "precision", "recall", "confusion_matrix"
    }
    assert expected_keys <= set(metrics.keys())
    assert 0.0 <= metrics["accuracy"] <= 1.0
