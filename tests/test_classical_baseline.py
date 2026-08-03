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
from gvls.eval.classical_baseline import evaluate_classical_baselines, jet_features_to_array
from gvls.qgnn_training import JetFeatures, extract_latent_features

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


def _random_features(n: int, seed: int) -> list[JetFeatures]:
    rng = np.random.default_rng(seed)
    return [
        JetFeatures(
            z_tilde=torch.from_numpy(rng.standard_normal((M, LATENT_DIM)).astype(np.float32)),
            a_z=torch.from_numpy(rng.random((M, M)).astype(np.float32)),
            label=int(i % 2),
        )
        for i in range(n)
    ]


def _separable_features(n: int, seed: int) -> list[JetFeatures]:
    # label=1 jets have z_tilde shifted well away from label=0's -- trivially
    # linearly separable, a sanity check that fitting actually works rather
    # than just returning plausible-looking numbers regardless of signal.
    rng = np.random.default_rng(seed)
    features = []
    for i in range(n):
        label = i % 2
        offset = 10.0 if label == 1 else -10.0
        z = rng.standard_normal((M, LATENT_DIM)).astype(np.float32) + offset
        a = rng.random((M, M)).astype(np.float32)
        features.append(
            JetFeatures(z_tilde=torch.from_numpy(z), a_z=torch.from_numpy(a), label=label)
        )
    return features


# ── jet_features_to_array ────────────────────────────────────────────────────

def test_jet_features_to_array_shapes() -> None:
    features = _random_features(10, seed=0)
    x, y = jet_features_to_array(features)
    expected_dim = M * LATENT_DIM + M * (M - 1) // 2
    assert x.shape == (10, expected_dim)
    assert y.shape == (10,)


def test_jet_features_to_array_labels_match() -> None:
    features = _random_features(6, seed=1)
    _, y = jet_features_to_array(features)
    assert list(y) == [f.label for f in features]


def test_jet_features_to_array_uses_upper_triangle_only() -> None:
    # a_z symmetric with a distinctive value off the diagonal -- exactly
    # M*(M-1)/2 entries should appear in the flattened feature vector.
    z = torch.zeros(M, LATENT_DIM)
    a = torch.eye(M) * 0.0
    a[0, 1] = a[1, 0] = 7.0
    features = [JetFeatures(z_tilde=z, a_z=a, label=0)]
    x, _ = jet_features_to_array(features)
    a_part = x[0, M * LATENT_DIM :]
    assert (a_part == 7.0).sum() == 1  # counted once, not twice


def test_jet_features_to_array_rejects_empty() -> None:
    with pytest.raises(ValueError):
        jet_features_to_array([])


# ── evaluate_classical_baselines ─────────────────────────────────────────────

def test_evaluate_classical_baselines_returns_expected_keys() -> None:
    train = _random_features(20, seed=2)
    test = _random_features(10, seed=3)
    result = evaluate_classical_baselines(train, test, seed=0)
    assert set(result.keys()) == {"logreg", "mlp"}
    for metrics in result.values():
        for key in ("accuracy", "auc", "macro_f1"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_classical_baselines_finds_trivial_separation() -> None:
    train = _separable_features(40, seed=4)
    test = _separable_features(20, seed=5)
    result = evaluate_classical_baselines(train, test, seed=0)
    assert result["logreg"]["accuracy"] > 0.95
    assert result["mlp"]["accuracy"] > 0.95


# ── End-to-end: real GVLS checkpoint -> extract_latent_features -> baselines ─

def test_classical_baseline_pipeline_end_to_end() -> None:
    """Mirrors classical_baseline_diagnostic.py's per-trial loop exactly
    (minus the Hydra/CLI wrapper): build+save a real GVLS checkpoint, load
    it back, extract frozen features, fit classical baselines -- validates
    the actual code path the diagnostic script runs before a remote run."""
    torch.manual_seed(0)
    model = build_pooled_gvls(IN_CHANNELS, LATENT_DIM, K, M, _base_cfg())
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
        loaded_model, _ = load_gvls_checkpoint(path, DEVICE)

    train_jets = [_synthetic_jet(20 + i, seed=i) for i in range(12)]
    test_jets = [_synthetic_jet(20 + i, seed=100 + i) for i in range(8)]
    train_features = extract_latent_features(loaded_model, train_jets, DEVICE)
    test_features = extract_latent_features(loaded_model, test_jets, DEVICE)

    result = evaluate_classical_baselines(train_features, test_features, seed=0)
    assert set(result.keys()) == {"logreg", "mlp"}
    for metrics in result.values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
