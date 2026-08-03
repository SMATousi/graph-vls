import numpy as np
import pytest
import torch

from gvls.eval.metrics import (
    auc_ap,
    bits_per_edge,
    classification_metrics,
    node_accuracy,
    select_best_threshold,
)

# ── auc_ap ────────────────────────────────────────────────────────────────────

def test_auc_ap_perfect() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    auc, ap = auc_ap(y_true, y_score)
    assert auc == pytest.approx(1.0)
    assert ap == pytest.approx(1.0)


def test_auc_ap_inverted() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    auc, ap = auc_ap(y_true, y_score)
    assert auc == pytest.approx(0.0)


def test_auc_ap_random_predictor() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(size=n)
    auc, ap = auc_ap(y_true, y_score)
    assert abs(auc - 0.5) < 0.02
    assert abs(ap - 0.5) < 0.02


def test_auc_ap_accepts_tensors() -> None:
    y_true = torch.tensor([1, 0, 1, 0])
    y_score = torch.tensor([0.9, 0.1, 0.8, 0.2])
    auc, ap = auc_ap(y_true, y_score)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ap <= 1.0


def test_auc_ap_returns_floats() -> None:
    auc, ap = auc_ap(np.array([1, 0]), np.array([1.0, 0.0]))
    assert isinstance(auc, float)
    assert isinstance(ap, float)


# ── node_accuracy ─────────────────────────────────────────────────────────────

def test_node_accuracy_perfect() -> None:
    y = np.array([0, 1, 2, 1])
    assert node_accuracy(y, y) == pytest.approx(1.0)


def test_node_accuracy_all_wrong() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert node_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_node_accuracy_partial() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 0, 0])
    assert node_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_node_accuracy_accepts_tensors() -> None:
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])
    assert node_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_node_accuracy_returns_float() -> None:
    assert isinstance(node_accuracy(np.array([0]), np.array([0])), float)


# ── bits_per_edge ─────────────────────────────────────────────────────────────

def test_bits_per_edge_perfect_logits() -> None:
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    logits = np.array([100.0, 100.0, -100.0, -100.0])
    assert bits_per_edge(y_true, logits) == pytest.approx(0.0, abs=1e-6)


def test_bits_per_edge_random_logits() -> None:
    # logit=0 → sigmoid=0.5 → BCE = log(2) nats = 1.0 bit for any label
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    logits = np.zeros(4)
    assert bits_per_edge(y_true, logits) == pytest.approx(1.0, abs=1e-9)


def test_bits_per_edge_accepts_tensors() -> None:
    y_true = torch.tensor([1.0, 0.0])
    logits = torch.tensor([100.0, -100.0])
    assert bits_per_edge(y_true, logits) == pytest.approx(0.0, abs=1e-6)


def test_bits_per_edge_returns_float() -> None:
    assert isinstance(bits_per_edge(np.array([1.0]), np.array([1.0])), float)


def test_bits_per_edge_nonnegative() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100).astype(float)
    logits = rng.standard_normal(100)
    assert bits_per_edge(y_true, logits) >= 0.0


# ── classification_metrics ────────────────────────────────────────────────────

def test_classification_metrics_perfect_logits() -> None:
    y_true = np.array([1, 1, 0, 0])
    logits = np.array([100.0, 100.0, -100.0, -100.0])
    m = classification_metrics(y_true, logits)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)
    assert m["ap"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_all_wrong() -> None:
    y_true = np.array([1, 1, 0, 0])
    logits = np.array([-100.0, -100.0, 100.0, 100.0])
    m = classification_metrics(y_true, logits)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["confusion_matrix"] == [[0, 2], [2, 0]]


def test_classification_metrics_returns_all_expected_keys() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=20)
    logits = rng.standard_normal(20)
    m = classification_metrics(y_true, logits)
    expected_keys = {
        "accuracy", "auc", "ap", "macro_f1", "precision", "recall", "confusion_matrix"
    }
    assert expected_keys <= set(m.keys())
    for key in expected_keys - {"confusion_matrix"}:
        assert 0.0 <= m[key] <= 1.0


def test_classification_metrics_accepts_tensors() -> None:
    y_true = torch.tensor([1, 0, 1, 0])
    logits = torch.tensor([2.0, -2.0, 1.5, -1.5])
    m = classification_metrics(y_true, logits)
    assert m["accuracy"] == pytest.approx(1.0)


def test_classification_metrics_threshold_affects_predictions() -> None:
    # probs ~= [0.574, 0.525, 0.488, 0.426]; threshold=0.5 misclassifies index
    # 2 (true=1, prob=0.488 < 0.5); threshold=0.45 gets everything right.
    y_true = np.array([1, 1, 1, 0])
    logits = np.array([0.3, 0.1, -0.05, -0.3])
    default_threshold = classification_metrics(y_true, logits, threshold=0.5)
    lower_threshold = classification_metrics(y_true, logits, threshold=0.45)
    assert default_threshold["accuracy"] == pytest.approx(0.75)
    assert lower_threshold["accuracy"] == pytest.approx(1.0)


# ── select_best_threshold ────────────────────────────────────────────────────

def test_select_best_threshold_finds_perfect_threshold_when_available() -> None:
    # Same scenario as test_classification_metrics_threshold_affects_predictions:
    # the fixed 0.5 default misses a perfect-accuracy threshold that exists
    # among the data's own probabilities.
    y_true = np.array([1, 1, 1, 0])
    logits = np.array([0.3, 0.1, -0.05, -0.3])
    threshold = select_best_threshold(y_true, logits, metric="accuracy")
    tuned = classification_metrics(y_true, logits, threshold=threshold)
    assert tuned["accuracy"] == pytest.approx(1.0)


def test_select_best_threshold_never_worse_than_default() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=50)
    logits = rng.standard_normal(50)
    threshold = select_best_threshold(y_true, logits, metric="accuracy")
    tuned_acc = classification_metrics(y_true, logits, threshold=threshold)["accuracy"]
    default_acc = classification_metrics(y_true, logits, threshold=0.5)["accuracy"]
    assert tuned_acc >= default_acc


def test_select_best_threshold_supports_macro_f1() -> None:
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=50)
    logits = rng.standard_normal(50)
    threshold = select_best_threshold(y_true, logits, metric="macro_f1")
    assert 0.0 <= threshold <= 1.0


def test_select_best_threshold_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError):
        select_best_threshold(np.array([1, 0]), np.array([1.0, -1.0]), metric="bogus")
