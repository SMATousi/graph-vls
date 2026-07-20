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
