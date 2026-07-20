"""Two-stage supervised QGNN training on frozen GVLS features (T4.5).

Stage 1 (T4.3, already done): pretrain `PooledGVLS` unsupervised, freeze it.
Stage 2 (this module): run the frozen model once over every jet to extract
`(z_tilde, A_z)` (no gradient -- Design Decision 8), then train only
`QGNNClassifier`'s circuit parameters (`theta`, `b_i`, the readout rotation)
supervised on the quark/gluon label, gradient-accumulated over minibatches of
jets (same per-jet loop pattern T4.2 validated for GVLS's own pretraining).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from gvls.data.jets import JetGraph
from gvls.eval.metrics import classification_metrics
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
    """BCE-with-logits loss for one jet's frozen (z_tilde, A_z) against its label."""
    z_tilde = features.z_tilde.to(device)
    a_z = features.a_z.to(device)
    label = torch.tensor([float(features.label)], device=device)
    logit = model(z_tilde, a_z)
    return F.binary_cross_entropy_with_logits(logit, label)


@torch.no_grad()
def evaluate_qgnn_classifier(
    model: QGNNClassifier, features: list[JetFeatures], device: torch.device
) -> dict[str, Any]:
    """Full classification metrics (accuracy, AUC, AP, macro-F1, precision,
    recall, confusion matrix) for a QGNNClassifier over a set of jets."""
    model.eval()
    labels = torch.tensor([f.label for f in features], dtype=torch.float32)
    logits = torch.cat(
        [model(f.z_tilde.to(device), f.a_z.to(device)).cpu() for f in features]
    )
    return classification_metrics(labels, logits)


@dataclass
class QGNNTrainingResult:
    best_state_dict: dict[str, Tensor]
    best_epoch: int
    best_val_metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)


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
) -> QGNNTrainingResult:
    """Train QGNNClassifier's circuit parameters via Adam (T4.5, FR-5).

    Jets are iterated one at a time (T4.2's pattern, reused here for the
    quantum classifier for the same reason: each jet's circuit reads a
    different (z_tilde, A_z), so there is no batched tensor to build), with
    gradients accumulated over a minibatch before each `optimizer.step()`.
    Tracks train/val loss and the full metric suite each epoch; returns the
    state dict from whichever epoch had the best validation accuracy.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    torch.manual_seed(seed)
    model = QGNNClassifier(m=m, d=d, num_layers=num_layers, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            optimizer.zero_grad()
            batch_loss = 0.0
            for idx in batch_idx:
                loss = qgnn_jet_loss(model, train_features[idx], device)
                (loss / len(batch_idx)).backward()
                batch_loss += loss.item()
            optimizer.step()
            running_loss += batch_loss
            n_seen += len(batch_idx)
        train_loss = running_loss / max(n_seen, 1)

        val_metrics = evaluate_qgnn_classifier(model, val_features, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        epoch_iter.set_postfix(train_loss=train_loss, val_acc=val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics

    return QGNNTrainingResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        history=history,
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
    """Inverse of `save_qgnn_checkpoint`: rebuild the ansatz, load weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = QGNNClassifier(
        m=int(config["m"]), d=int(config["d"]), num_layers=int(config["num_layers"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config
