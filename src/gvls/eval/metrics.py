from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
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
