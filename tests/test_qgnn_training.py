import tempfile
from pathlib import Path

import numpy as np
import torch

from gvls.compression.jet_sweep import (
    build_pooled_gvls,
    load_gvls_checkpoint,
    save_gvls_checkpoint,
)
from gvls.data.jets import NUM_FEATURES, PDGIDS, build_jet_graph
from gvls.qgnn_training import (
    JetFeatures,
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
