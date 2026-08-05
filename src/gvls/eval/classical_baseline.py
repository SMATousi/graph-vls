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

# T5.2 (specs/phase5/): which parts of the frozen latent representation a
# classical model is allowed to see. "z_a" is the pre-T5.2 behaviour and the
# default, so existing results stay reproducible; the rest exist to measure
# what the pooled posterior's variance is worth, since it was previously
# discarded before any classifier saw it.
FEATURE_SETS = ("z_a", "z_a_logvar", "z_a_mu_logvar", "logvar_only")


def jet_features_to_array(
    features: list[JetFeatures], feature_set: str = "z_a"
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a jet's frozen latent representation into one feature vector.

    `z_tilde` (M, d) is flattened row-major; `A_z` (M, M) contributes only
    its upper triangle (k=1, excluding the diagonal) since the latent graph
    is undirected and symmetric -- this gives the classical model access to
    the same raw information the QGNN circuit has (z_tilde as rotation-angle
    inputs, A_z as entangling-gate coefficients), without double-counting
    each edge.

    `feature_set` (T5.2) selects what else is included:

    * ``"z_a"``            -- z_tilde + A_z upper triangle (default, the
                             pre-T5.2 behaviour and what every result before
                             specs/phase5/ was measured on)
    * ``"z_a_logvar"``     -- adds the pooled posterior's log-variance
    * ``"z_a_mu_logvar"``  -- adds the pooled posterior mean as well. `mu` is
                             largely redundant with `z_tilde` (which is `mu`
                             pushed through latent message passing at eval
                             time), so this mostly probes whether message
                             passing is discarding anything
    * ``"logvar_only"``    -- the variance alone, as a direct answer to
                             "does the posterior's spread carry any class
                             information at all?"

    Raises ValueError if a variance-bearing set is requested but the features
    were extracted before T5.2 (i.e. `log_var is None`), rather than silently
    falling back to a narrower feature set and reporting it as the wider one.
    """
    if not features:
        raise ValueError("features must be non-empty")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got '{feature_set}'")

    needs_log_var = feature_set != "z_a"
    needs_mu = feature_set == "z_a_mu_logvar"
    if needs_log_var and features[0].log_var is None:
        raise ValueError(
            f"feature_set='{feature_set}' needs log_var, but these JetFeatures have none "
            "(extracted before T5.2). Re-run extract_latent_features."
        )
    if needs_mu and features[0].mu is None:
        raise ValueError(
            f"feature_set='{feature_set}' needs mu, but these JetFeatures have none "
            "(extracted before T5.2). Re-run extract_latent_features."
        )

    m = features[0].z_tilde.shape[0]
    iu, ju = np.triu_indices(m, k=1)
    rows = []
    for f in features:
        if feature_set == "logvar_only":
            rows.append(f.log_var.numpy().reshape(-1))  # type: ignore[union-attr]
            continue
        parts = [f.z_tilde.numpy().reshape(-1), f.a_z.numpy()[iu, ju]]
        if needs_mu:
            parts.append(f.mu.numpy().reshape(-1))  # type: ignore[union-attr]
        if needs_log_var:
            parts.append(f.log_var.numpy().reshape(-1))  # type: ignore[union-attr]
        rows.append(np.concatenate(parts))
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
    feature_set: str = "z_a",
) -> dict[str, dict[str, Any]]:
    """Fit logistic regression + a shallow MLP on frozen GVLS features, score on test.

    Returns `{"logreg": {...}, "mlp": {...}}`, each with accuracy/auc/macro_f1
    -- directly comparable to `evaluate_qgnn_classifier`'s own metric keys
    (same names, subset of them) for a side-by-side reading against
    `results/qgnn/qg_jets_metrics_lorentz800_summary.json`.

    `feature_set` (T5.2) defaults to `"z_a"`, the pre-T5.2 behaviour, so
    existing callers and results are unaffected. See `jet_features_to_array`.
    """
    x_train, y_train = jet_features_to_array(train_features, feature_set)
    x_test, y_test = jet_features_to_array(test_features, feature_set)

    logreg = LogisticRegression(max_iter=2000, random_state=seed)
    logreg.fit(x_train, y_train)
    logreg_metrics = _score_predictions(
        y_test, logreg.predict(x_test), logreg.decision_function(x_test)
    )

    mlp = MLPClassifier(hidden_layer_sizes=(mlp_hidden_units,), max_iter=2000, random_state=seed)
    mlp.fit(x_train, y_train)
    mlp_metrics = _score_predictions(y_test, mlp.predict(x_test), mlp.predict_proba(x_test)[:, 1])

    return {"logreg": logreg_metrics, "mlp": mlp_metrics}
