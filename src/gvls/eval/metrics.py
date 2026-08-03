from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor

ArrayLike = np.ndarray | Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auc_ap(y_true: ArrayLike, y_score: ArrayLike) -> tuple[float, float]:
    """AUC-ROC and Average Precision for binary edge prediction.

    Args:
        y_true:  Binary labels (0 or 1), shape (N,).
        y_score: Continuous scores (higher = more likely positive), shape (N,).

    Returns:
        (auc, ap) as floats in [0, 1].
    """
    yt = _to_numpy(y_true).ravel()
    ys = _to_numpy(y_score).ravel()
    auc = float(roc_auc_score(yt, ys))
    ap = float(average_precision_score(yt, ys))
    return auc, ap


def node_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of correctly classified nodes.

    Args:
        y_true: Integer class labels, shape (N,).
        y_pred: Predicted class labels, shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_pred).ravel()
    return float((yt == yp).mean())


def bits_per_edge(adj_true: ArrayLike, adj_logits: ArrayLike) -> float:
    """Mean binary cross-entropy per edge-pair, expressed in bits.

    Measures the coding cost of the predicted adjacency distribution.
    Lower is better; 0.0 is perfect, 1.0 corresponds to a random (0.5) predictor.

    Args:
        adj_true:   Binary ground-truth adjacency values (0 or 1), shape (N,).
        adj_logits: Raw (pre-sigmoid) logits for each edge pair, shape (N,).

    Returns:
        Mean bits per edge-pair as a float.
    """
    yt = _to_numpy(adj_true).ravel().astype(np.float64)
    yl = _to_numpy(adj_logits).ravel().astype(np.float64)
    # Numerically stable BCE: max(l,0) - y*l + log(1 + exp(-|l|))
    bce_nats = np.maximum(yl, 0.0) - yt * yl + np.log1p(np.exp(-np.abs(yl)))
    return float(bce_nats.mean() / np.log(2.0))


def classification_metrics(
    y_true: ArrayLike, y_logits: ArrayLike, threshold: float = 0.5
) -> dict[str, Any]:
    """Full binary-classification metrics from raw (pre-sigmoid) logits (T4.5/T4.6).

    Used for the QGNN's quark/gluon jet classification task -- one scalar
    logit per jet, exactly what `QGNNClassifier.forward` and `node_accuracy`'s
    inputs already look like, generalized here beyond just accuracy since a
    single scalar (accuracy or F1 alone) isn't enough to judge a classifier
    trained via parameter-shift on a noiseless simulator.

    Args:
        y_true:    Binary labels (0 or 1), shape (N,).
        y_logits:  Raw (pre-sigmoid) logits, shape (N,).
        threshold: Probability strictly above which a jet is predicted positive.

    Returns:
        Dict with accuracy, auc, ap, macro_f1, precision, recall (all floats,
        precision/recall/macro_f1 in [0, 1] with zero_division=0), and
        confusion_matrix as a nested list [[tn, fp], [fn, tp]].
    """
    yt = _to_numpy(y_true).ravel().astype(np.int64)
    yl = _to_numpy(y_logits).ravel().astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-yl))
    y_pred = (probs > threshold).astype(np.int64)

    auc, ap = auc_ap(yt, probs)
    cm = confusion_matrix(yt, y_pred, labels=[0, 1])

    return {
        "accuracy": float((yt == y_pred).mean()),
        "auc": auc,
        "ap": ap,
        "macro_f1": float(f1_score(yt, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(yt, y_pred, zero_division=0)),
        "recall": float(recall_score(yt, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def select_best_threshold(
    y_true: ArrayLike, y_logits: ArrayLike, metric: str = "accuracy"
) -> float:
    """Pick the probability threshold that maximizes `metric` on a held-out split.

    `classification_metrics`'s default `threshold=0.5` assumes the raw
    logits are already well-calibrated around a 0.5 decision boundary --
    not guaranteed for a classifier trained via SPSA/parameter-shift on a
    small, noisy dataset (a real observation, not hypothetical: a 5-seed
    QGNN jet-classification repeat sweep saw AUC/AP -- threshold-independent
    ranking metrics -- stay tight across seeds (e.g. AUC 0.684 +/- 0.009)
    while fixed-0.5 accuracy/macro-F1/recall swung far more (recall alone
    ranged 0.644-0.961) -- the signature of a threshold-calibration problem,
    not a ranking-quality one; see specs/phase4/validation.md V-11).

    Only ever call this on a validation split, never on the split whose
    score you intend to report -- tuning the threshold against the test set
    would be leakage.

    Candidates are every distinct predicted probability (plus 0.0 and 1.0):
    accuracy/F1/etc. as a function of threshold is a step function that only
    changes value at these points, so this is an exact search over all
    achievable operating points, not an approximate grid. Ties are broken by
    the candidate closest to 0.5 -- the least aggressive departure from the
    un-tuned default among equally-good options.
    """
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError(f"unsupported metric {metric!r}, expected 'accuracy' or 'macro_f1'")

    yt = _to_numpy(y_true).ravel().astype(np.int64)
    yl = _to_numpy(y_logits).ravel().astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-yl))

    candidates = np.unique(np.concatenate([probs, [0.0, 1.0]]))
    best_threshold = 0.5
    best_score = -1.0
    for candidate in candidates:
        y_pred = (probs > candidate).astype(np.int64)
        if metric == "accuracy":
            score = float((yt == y_pred).mean())
        else:
            score = float(f1_score(yt, y_pred, average="macro", zero_division=0))
        if score > best_score or (
            score == best_score and abs(candidate - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(candidate)
    return best_threshold
