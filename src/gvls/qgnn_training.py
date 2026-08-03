"""Two-stage supervised QGNN training on frozen GVLS features (T4.5).

Stage 1 (T4.3, already done): pretrain `PooledGVLS` unsupervised, freeze it.
Stage 2 (this module): run the frozen model once over every jet to extract
`(z_tilde, A_z)` (no gradient -- Design Decision 8), then train only
`QGNNClassifier`'s circuit parameters (`theta`, `b_i`, the readout rotation)
supervised on the quark/gluon label, gradient-accumulated over minibatches of
jets (same per-jet loop pattern T4.2 validated for GVLS's own pretraining).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from gvls.data.jets import JetGraph
from gvls.eval.metrics import classification_metrics, select_best_threshold
from gvls.models.pooling import PooledGVLS
from gvls.models.qgnn import QGNNClassifier


@dataclass
class JetFeatures:
    """One jet's frozen (z_tilde, A_z) pair plus its label."""

    z_tilde: Tensor  # (M, latent_dim)
    a_z: Tensor       # (M, M)
    label: int


def extract_latent_features(
    model: PooledGVLS, jets: list[JetGraph], device: torch.device
) -> list[JetFeatures]:
    """Run a frozen PooledGVLS once over every jet, no gradient (Design Decision 8).

    `model` is not modified and no optimizer step ever touches it here --
    freezing happens simply by never constructing an optimizer over its
    parameters and always calling it under `torch.no_grad()`.
    """
    model.eval()
    features: list[JetFeatures] = []
    with torch.no_grad():
        for jet in jets:
            x = jet.x.to(device)
            edge_index = jet.edge_index.to(device)
            _mu, _log_var, _z, a_z, z_tilde, _s, _recon_logits = model(x, edge_index)
            features.append(
                JetFeatures(z_tilde=z_tilde.cpu(), a_z=a_z.cpu(), label=int(jet.y.item()))
            )
    return features


def qgnn_jet_loss(model: QGNNClassifier, features: JetFeatures, device: torch.device) -> Tensor:
    """BCE-with-logits loss for one jet's frozen (z_tilde, A_z) against its
    label. Superseded by `qgnn_batch_loss` in `train_qgnn_classifier`'s main
    loop (T4.8) -- kept as the known-correct per-jet reference the batched
    path's gradient-parity test (`tests/test_qgnn_training.py`) checks against.
    """
    z_tilde = features.z_tilde.to(device)
    a_z = features.a_z.to(device)
    label = torch.tensor([float(features.label)], device=device)
    logit = model(z_tilde, a_z)
    return F.binary_cross_entropy_with_logits(logit, label)


def collate_jet_features(features: list[JetFeatures]) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a minibatch of `JetFeatures` into batched `(B,M,d)`/`(B,M,M)`/`(B,)`
    tensors (T4.8). Valid because every jet is pooled to the same fixed `M`
    (`plan.md` Design Decision 3) -- every jet's `z_tilde`/`a_z` shape is
    identical, so stacking needs no padding/masking, unlike the classical
    encoder/pooling stack's per-jet-`N` constraint (Design Decision 7).
    """
    z_tildes = torch.stack([f.z_tilde for f in features])
    a_zs = torch.stack([f.a_z for f in features])
    labels = torch.tensor([float(f.label) for f in features])
    return z_tildes, a_zs, labels


def qgnn_batch_loss(
    model: QGNNClassifier, features: list[JetFeatures], device: torch.device
) -> Tensor:
    """BCE-with-logits loss (mean-reduced) for a minibatch of jets, via one
    batched `QGNNClassifier` call (T4.8) instead of a per-jet loop."""
    z_tildes, a_zs, labels = collate_jet_features(features)
    logits = model(z_tildes.to(device), a_zs.to(device))
    return F.binary_cross_entropy_with_logits(logits, labels.to(device))


@torch.no_grad()
def compute_qgnn_logits(
    model: QGNNClassifier, features: list[JetFeatures], device: torch.device, batch_size: int = 256
) -> tuple[Tensor, Tensor]:
    """Raw (pre-sigmoid) logits + labels for a QGNNClassifier over a set of jets.

    Factored out of `evaluate_qgnn_classifier` (T4.10 followup, validation.md
    V-11) so `select_best_threshold`'s validation-only threshold search can
    reuse the same batched forward pass without duplicating it.
    """
    model.eval()
    all_logits: list[Tensor] = []
    for start in range(0, len(features), batch_size):
        chunk = features[start : start + batch_size]
        z_tildes, a_zs, _ = collate_jet_features(chunk)
        all_logits.append(model(z_tildes.to(device), a_zs.to(device)).cpu())
    logits = torch.cat(all_logits)
    labels = torch.tensor([f.label for f in features], dtype=torch.float32)
    return logits, labels


@torch.no_grad()
def evaluate_qgnn_classifier(
    model: QGNNClassifier,
    features: list[JetFeatures],
    device: torch.device,
    batch_size: int = 256,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Full classification metrics (accuracy, AUC, AP, macro-F1, precision,
    recall, confusion matrix) for a QGNNClassifier over a set of jets.

    Batched (T4.8): evaluates `batch_size` jets per circuit call instead of
    one call per jet, bounding a single Estimator job's size (chunked rather
    than one call for the whole split, per `plan.md` Design Decision 10).

    `threshold` (T4.10 followup, validation.md V-11) defaults to 0.5 for
    backward compatibility -- pass a validation-selected threshold
    (`select_best_threshold`) when scoring a held-out test set instead, so a
    miscalibrated (not necessarily poorly-ranked) classifier isn't penalized
    by an arbitrary cutoff.
    """
    logits, labels = compute_qgnn_logits(model, features, device, batch_size)
    return classification_metrics(labels, logits, threshold=threshold)


_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Construct the training-loop optimizer by name (T4.10).

    `adamw` is the default (`plan.md` Design Decision 12, matching the
    Lorentz-EQGNN literature baseline's protocol); `adam` remains selectable
    for the pre-T4.10 configuration or any other comparison.
    """
    try:
        optimizer_cls = _OPTIMIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown optimizer {name!r}, expected one of {sorted(_OPTIMIZERS)}"
        ) from None
    return optimizer_cls(params, lr=lr)


@dataclass
class QGNNTrainingResult:
    best_state_dict: dict[str, Tensor]
    best_epoch: int
    best_val_metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    best_train_metrics: dict[str, Any] = field(default_factory=dict)
    best_threshold: float = 0.5


def train_qgnn_classifier(
    train_features: list[JetFeatures],
    val_features: list[JetFeatures],
    m: int,
    d: int,
    num_layers: int,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    show_progress: bool = True,
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
    gradient_method: str = "spsa",
    spsa_epsilon: float = 1e-6,
    spsa_batch_size: int = 1,
    optimizer: str = "adamw",
) -> QGNNTrainingResult:
    """Train QGNNClassifier's circuit parameters via Adam/AdamW (T4.5, FR-5).

    `optimizer` (new, T4.10, `plan.md` Design Decision 12) selects between
    `"adamw"` (default, matching the Lorentz-EQGNN literature baseline's
    protocol) and `"adam"` (the pre-T4.10 default, still available). This is
    purely a classical training-loop choice -- it has no bearing on the
    quantum circuit's `gradient_method` (SPSA vs. parameter-shift, T4.9),
    which is an orthogonal, separately-configurable concern.

    Each minibatch is one batched `QGNNClassifier` call (T4.8) via
    `qgnn_batch_loss`/`collate_jet_features` -- unlike the classical GVLS
    stack (Design Decision 7), the QGNN's inputs are already fixed-size
    (`M x d`, `M x M`) post-pooling for every jet, so there is no per-jet-`N`
    obstacle to batching here; the original per-jet loop this superseded
    (`qgnn_jet_loss` in a Python `for` loop, one `TorchConnector` call per
    jet) was carried over from that classical-stack pattern without
    re-examining whether the constraint actually applied. See `plan.md`
    Design Decision 10 for the full diagnosis.
    Tracks train/val loss and the full metric suite each epoch; returns the
    state dict from whichever epoch had the best validation accuracy (always
    judged at the fixed `threshold=0.5` `best_val_metrics` already used, so
    epoch selection is unaffected by the threshold tuning below).

    `result.best_threshold` (T4.10 followup, validation.md V-11) is a
    validation-selected decision threshold (`select_best_threshold`,
    `metric="accuracy"`) computed once from the best epoch's own val logits
    -- report/apply this at test time (`evaluate_qgnn_classifier`'s
    `threshold` argument) instead of the default 0.5, since a small,
    SPSA/parameter-shift-trained classifier's raw logits aren't guaranteed
    to be well-calibrated around 0.5 even when their ranking (AUC/AP) is
    good. `best_train_metrics` is computed at this tuned threshold (not
    0.5) so it's reported on the same basis as the eventual tuned test
    accuracy; `best_val_metrics` stays at 0.5 -- it's the per-epoch
    selection criterion's own record, not a number this function tunes
    against itself.

    `on_epoch_end`, if given, is called once per epoch as
    `on_epoch_end(epoch, metrics)` with that epoch's `{"epoch", "train_loss",
    **val_metrics}` row -- mirrors `train_pooled_gvls_on_jets`'s callback
    (`src/gvls/compression/jet_sweep.py`), added there after stage-1 GVLS
    pretraining was found to log nothing to W&B until training finished
    (`specs/phase4/validation.md` V-5). This training loop had the same gap:
    the caller previously only saw `result.history` after this function
    returned, so a long run that crashed or was killed partway (a real risk
    at this stage's current per-jet, unbatched training cost, see T4.8) would
    report zero training metrics.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    torch.manual_seed(seed)
    model = QGNNClassifier(
        m=m,
        d=d,
        num_layers=num_layers,
        seed=seed,
        gradient_method=gradient_method,
        spsa_epsilon=spsa_epsilon,
        spsa_batch_size=spsa_batch_size,
    ).to(device)
    optim = _build_optimizer(optimizer, model.parameters(), lr)
    shuffle_generator = torch.Generator().manual_seed(seed)

    best_val_accuracy = -1.0
    best_state_dict: dict[str, Tensor] = {
        k: v.clone() for k, v in model.state_dict().items()
    }
    best_epoch = -1
    best_val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    epoch_iter = tqdm(range(epochs), desc="train QGNN", disable=not show_progress)
    for epoch in epoch_iter:
        model.train()
        perm = torch.randperm(len(train_features), generator=shuffle_generator).tolist()
        running_loss, n_seen = 0.0, 0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            batch = [train_features[i] for i in batch_idx]
            optim.zero_grad()
            loss = qgnn_batch_loss(model, batch, device)  # mean over the minibatch
            loss.backward()
            optim.step()
            running_loss += loss.item() * len(batch_idx)  # de-mean back to a sum
            n_seen += len(batch_idx)
        train_loss = running_loss / max(n_seen, 1)

        val_metrics = evaluate_qgnn_classifier(model, val_features, device)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_metrics)
        epoch_iter.set_postfix(train_loss=train_loss, val_acc=val_metrics["accuracy"])

        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_metrics)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics

    # T4.10 (FR-6 amendment): report train accuracy (and the full metric
    # suite) alongside val/test, matching the literature table's convention
    # of reporting both -- computed once, on the best-val-accuracy weights,
    # not re-derived from the (val-only) per-epoch history.
    model.load_state_dict(best_state_dict)

    # T4.10 followup (validation.md V-11): select a validation-only decision
    # threshold from the best epoch's own val logits (never the test set --
    # that would be leakage), then report train metrics on that same
    # calibrated basis rather than the raw 0.5 default.
    val_logits, val_labels = compute_qgnn_logits(model, val_features, device)
    best_threshold = select_best_threshold(val_labels, val_logits, metric="accuracy")

    train_logits, train_labels = compute_qgnn_logits(model, train_features, device)
    best_train_metrics = classification_metrics(
        train_labels, train_logits, threshold=best_threshold
    )

    return QGNNTrainingResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        history=history,
        best_train_metrics=best_train_metrics,
        best_threshold=best_threshold,
    )


def save_qgnn_checkpoint(
    state_dict: dict[str, Tensor], config: dict[str, Any], path: str
) -> None:
    """Persist a QGNNClassifier's weights plus its (m, d, num_layers) config."""
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "config": config}, path)


def load_qgnn_checkpoint(path: str, device: torch.device) -> tuple[QGNNClassifier, dict[str, Any]]:
    """Inverse of `save_qgnn_checkpoint`: rebuild the ansatz, load weights.

    `gradient_method` only affects `.backward()`, never `forward()`, so it
    has no bearing on correctness here (this model is only ever `.eval()`'d
    for inference) -- restored anyway for provenance/completeness, defaulting
    to "spsa" for older checkpoints saved before T4.8's SPSA follow-up.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = QGNNClassifier(
        m=int(config["m"]),
        d=int(config["d"]),
        num_layers=int(config["num_layers"]),
        gradient_method=config.get("gradient_method", "spsa"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config
