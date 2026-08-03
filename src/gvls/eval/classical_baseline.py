"""Classical-baseline diagnostic on frozen GVLS features (T4.10 followup, validation.md V-11).

Motivation: after fixing GVLS's data-starved pretraining (validation.md
V-11), the QGNN's test accuracy rose to ~66.9% but is still ~7 points short
of Lorentz-EQGNN's 74.00%, and train accuracy (~69.8%) isn't saturating --
something is still capping fit quality, not just generalization. This module
answers *where* that cap sits: train a plain classical classifier on the
exact same frozen `(z_tilde, A_z)` features the QGNN sees, using the exact
same per-trial 800-jet training subsets and fixed test set. If the classical
baseline also plateaus around ~67-70%, the ceiling is upstream in GVLS's
compression (the features themselves aren't more separable than that,
regardless of classifier); if it does meaningfully better, the QGNN's own
training (circuit capacity, SPSA gradient noise) is leaving accuracy on the
table.

Two classifiers, chosen to bracket the question: `LogisticRegression` (a
linear ceiling -- if this alone reaches ~67%, the features are already
"easy") and a shallow `MLPClassifier` (one small hidden layer, roughly
matching the QGNN's own shallow single re-uploading layer in spirit, to
probe for exploitable nonlinear structure a linear model would miss).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from gvls.qgnn_training import JetFeatures


def jet_features_to_array(features: list[JetFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten `(z_tilde, A_z)` into one feature vector per jet.

    `z_tilde` (M, d) is flattened row-major; `A_z` (M, M) contributes only
    its upper triangle (k=1, excluding the diagonal) since the latent graph
    is undirected and symmetric -- this gives the classical model access to
    the same raw information the QGNN circuit has (z_tilde as rotation-angle
    inputs, A_z as entangling-gate coefficients), without double-counting
    each edge.
    """
    if not features:
        raise ValueError("features must be non-empty")
    m = features[0].z_tilde.shape[0]
    iu, ju = np.triu_indices(m, k=1)
    rows = []
    for f in features:
        z = f.z_tilde.numpy().reshape(-1)
        a = f.a_z.numpy()[iu, ju]
        rows.append(np.concatenate([z, a]))
    x = np.stack(rows).astype(np.float64)
    y = np.array([f.label for f in features], dtype=np.int64)
    return x, y


def _score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_classical_baselines(
    train_features: list[JetFeatures],
    test_features: list[JetFeatures],
    seed: int,
    mlp_hidden_units: int = 16,
) -> dict[str, dict[str, Any]]:
    """Fit logistic regression + a shallow MLP on frozen GVLS features, score on test.

    Returns `{"logreg": {...}, "mlp": {...}}`, each with accuracy/auc/macro_f1
    -- directly comparable to `evaluate_qgnn_classifier`'s own metric keys
    (same names, subset of them) for a side-by-side reading against
    `results/qgnn/qg_jets_metrics_lorentz800_summary.json`.
    """
    x_train, y_train = jet_features_to_array(train_features)
    x_test, y_test = jet_features_to_array(test_features)

    logreg = LogisticRegression(max_iter=2000, random_state=seed)
    logreg.fit(x_train, y_train)
    logreg_metrics = _score_predictions(
        y_test, logreg.predict(x_test), logreg.decision_function(x_test)
    )

    mlp = MLPClassifier(hidden_layer_sizes=(mlp_hidden_units,), max_iter=2000, random_state=seed)
    mlp.fit(x_train, y_train)
    mlp_metrics = _score_predictions(y_test, mlp.predict(x_test), mlp.predict_proba(x_test)[:, 1])

    return {"logreg": logreg_metrics, "mlp": mlp_metrics}
