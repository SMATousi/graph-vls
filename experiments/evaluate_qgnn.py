"""Full classification-metric evaluation of the trained QGNN on held-out test jets (T4.6).

Reports accuracy, AUC, average precision, macro-F1, precision, recall, and
the confusion matrix on the test split (untouched by both T4.3/T4.5's
pretraining/training), plus the qubit count (M) and circuit depth
(num_layers) actually used.

NOTE: T4.6's *literature-comparison* deliverable (a published QGNN accuracy
number on this or a comparable dataset) is intentionally NOT included here.
plan.md is explicit that identifying one is real, unresolved research
legwork, not a placeholder to fill in mechanically, and specs/phase4/
requirements.md NFR-5 forbids fabricating a comparison number before that
search is actually done. This script reports GVLS+QGNN's own numbers only --
its absence here means "not yet looked for," not "none exists."

T4.10 (plan.md Design Decision 12) adds train accuracy and wall-clock
training/inference time to the report, read back from the QGNN checkpoint's
saved config (`train_accuracy`/`training_time_s`, persisted by
experiments/train_qgnn.py) plus a freshly-measured test-set inference time --
matching sota-table.png's Table II columns for a direct row-by-row
comparison. Per NFR-5, these wall-clock numbers are our own hardware's, not
matched to the literature table's -- reported plainly, not implying parity.
Checkpoints saved before T4.10 lack the two training-side fields; this script
reports them as unavailable rather than fabricating a number.

T4.10 followup (validation.md V-11): the primary reported test metrics now
use the validation-selected decision threshold persisted in the QGNN
checkpoint's config (`train_qgnn_classifier`'s `best_threshold`) instead of
the fixed 0.5 default -- a 5-seed repeat sweep found accuracy/macro-F1/
recall swinging far more across seeds than the threshold-independent AUC/AP
did, the signature of a miscalibrated cutoff rather than a poorly-ranked
classifier. The fixed-0.5 accuracy is still computed and reported alongside
(as `test_accuracy_fixed_threshold_0.5`) so the size of that effect is
visible, not hidden. Checkpoints saved before this change lack a persisted
threshold and fall back to 0.5 (both numbers then coincide).

Usage:
    python experiments/evaluate_qgnn.py
    python experiments/evaluate_qgnn.py gvls_checkpoint_path=checkpoints/gvls_jets_m6.pt \
        qgnn_checkpoint_path=checkpoints/qgnn_jets_m6.pt
"""

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.compression.jet_sweep import load_gvls_checkpoint
from gvls.data.jets import load_split_from_config
from gvls.eval.metrics import classification_metrics
from gvls.qgnn_training import (
    compute_qgnn_logits,
    extract_latent_features,
    load_qgnn_checkpoint,
)


@hydra.main(version_base=None, config_path="../configs", config_name="qgnn_evaluate_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading GVLS checkpoint from {cfg.gvls_checkpoint_path}...")
    gvls_model, gvls_config = load_gvls_checkpoint(str(cfg.gvls_checkpoint_path), device)
    print(f"Loading QGNN checkpoint from {cfg.qgnn_checkpoint_path}...")
    qgnn_model, qgnn_config = load_qgnn_checkpoint(str(cfg.qgnn_checkpoint_path), device)
    m, num_layers = int(qgnn_config["m"]), int(qgnn_config["num_layers"])
    print(f"  GVLS M={gvls_config['num_clusters']}  QGNN M={m}  num_layers={num_layers}")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    protocol = data_cfg.get("protocol", "balanced")
    print(f"Loading qg_jets (protocol={protocol}, seed={cfg.data.seed})...")
    split = load_split_from_config(data_cfg)
    print(f"Evaluating on {len(split.test)} held-out test jets (untouched by training)...")

    # T4.10 (validation.md V-10): data_cfg (notably num_jets/protocol) is
    # logged so this run is distinguishable in the W&B UI from other
    # dataset-size/protocol configurations, not just via num_test_jets.
    # wandb.name/group/tags default to the pre-T4.10 behavior unless overridden.
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name or f"qgnn-eval-M{m}",
        group=cfg.wandb.group or "qgnn-jet-classification",
        tags=list(cfg.wandb.tags),
        job_type="evaluation",
        config={
            "m": m,
            "num_layers": num_layers,
            "num_test_jets": len(split.test),
            "data": data_cfg,
        },
    )

    test_features = extract_latent_features(gvls_model, split.test, device)

    # T4.10 followup (validation.md V-11): validation-selected threshold,
    # persisted by train_qgnn.py -- falls back to 0.5 for checkpoints saved
    # before this change (in which case both metrics dicts below coincide).
    threshold = qgnn_config.get("threshold")
    if threshold is None:
        threshold = 0.5
        threshold_source = "default (checkpoint predates threshold tuning)"
    else:
        threshold = float(threshold)
        threshold_source = "validation-selected (train_qgnn.py)"

    inference_start = time.perf_counter()
    test_logits, test_labels = compute_qgnn_logits(qgnn_model, test_features, device)
    inference_time_s = time.perf_counter() - inference_start

    metrics = classification_metrics(test_labels, test_logits, threshold=threshold)
    metrics_fixed = classification_metrics(test_labels, test_logits, threshold=0.5)

    # T4.10: read back what train_qgnn.py measured/persisted at training time.
    # `.get(...)` defaults to None for checkpoints saved before T4.10 --
    # reported as "not available" below rather than fabricated (NFR-5).
    optimizer_name = qgnn_config.get("optimizer")
    train_accuracy = qgnn_config.get("train_accuracy")
    training_time_s = qgnn_config.get("training_time_s")

    wandb.log({f"test_{key}": val for key, val in metrics.items() if key != "confusion_matrix"})
    wandb.log(
        {
            "inference_time_s": inference_time_s,
            "test_accuracy_fixed_threshold_0.5": metrics_fixed["accuracy"],
            "threshold": threshold,
        }
    )
    if train_accuracy is not None:
        wandb.log({"train_accuracy": train_accuracy, "training_time_s": training_time_s})

    print("\nTest-set metrics (at the validation-selected threshold):")
    for key in ("accuracy", "auc", "ap", "macro_f1", "precision", "recall"):
        print(f"  {key:12s}: {metrics[key]:.4f}")
    print(
        "  confusion_matrix (rows=true, cols=pred, label 0=quark/1=gluon): "
        f"{metrics['confusion_matrix']}"
    )
    print(f"  threshold used: {threshold:.4f} ({threshold_source})")
    print(
        f"  accuracy at fixed threshold=0.5 (for comparison): {metrics_fixed['accuracy']:.4f}  "
        f"(delta: {metrics['accuracy'] - metrics_fixed['accuracy']:+.4f})"
    )
    def _fmt(val: float | str | None, spec: str = "") -> str:
        # checkpoints saved before T4.10 lack train_accuracy/training_time_s
        return "not available (checkpoint predates T4.10)" if val is None else format(val, spec)

    print(f"  qubit_count (M): {m}")
    print(f"  circuit_depth (num_layers): {num_layers}")
    print(f"  optimizer: {_fmt(optimizer_name)}")
    print(f"  train_accuracy: {_fmt(train_accuracy, '.4f')}")
    print(f"  training_time_s: {_fmt(training_time_s, '.2f')}")
    print(f"  inference_time_s: {inference_time_s:.2f}  ({len(split.test)} test jets)")
    print(
        "\nNOTE (NFR-5): wall-clock timing above is measured on this run's own hardware, "
        "not matched to sota-table.png's Table II hardware -- directional comparison only."
    )
    print(
        "\nNOTE: no literature QGNN comparison number is included here -- "
        "that search (plan.md T4.6) has not been done yet. Comparison target identified "
        "2026-07-30 (Lorentz-EQGNN, sota-table.png Table II) -- see plan.md Design Decision 12."
    )

    results = {
        "m": m,
        "num_layers": num_layers,
        "optimizer": optimizer_name,
        "num_test_jets": len(split.test),
        "train_accuracy": train_accuracy,
        "test_accuracy": metrics["accuracy"],
        "test_accuracy_fixed_threshold_0.5": metrics_fixed["accuracy"],
        "threshold": threshold,
        "training_time_s": training_time_s,
        "inference_time_s": inference_time_s,
        **metrics,
    }
    results_path = Path(str(cfg.results_path))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {results_path}")

    artifact = wandb.Artifact(
        name=f"qgnn-jets-m{m}-test-metrics",
        type="evaluation",
        metadata={key: val for key, val in results.items() if key != "confusion_matrix"},
    )
    artifact.add_file(str(results_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
