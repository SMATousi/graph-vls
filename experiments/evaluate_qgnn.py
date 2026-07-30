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
from omegaconf import DictConfig

import wandb
from gvls.compression.jet_sweep import load_gvls_checkpoint
from gvls.data.jets import load_qg_jets, split_jets
from gvls.qgnn_training import (
    evaluate_qgnn_classifier,
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

    print(f"Loading {cfg.data.num_jets} qg_jets (seed={cfg.data.seed})...")
    jets = load_qg_jets(
        num_jets=int(cfg.data.num_jets),
        k_graph_cap=int(cfg.data.k_graph_cap),
        seed=int(cfg.data.seed),
    )
    split = split_jets(
        jets,
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )
    print(f"Evaluating on {len(split.test)} held-out test jets (untouched by training)...")

    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=f"qgnn-eval-M{m}",
        group="qgnn-jet-classification",
        job_type="evaluation",
        config={"m": m, "num_layers": num_layers, "num_test_jets": len(split.test)},
    )

    test_features = extract_latent_features(gvls_model, split.test, device)
    inference_start = time.perf_counter()
    metrics = evaluate_qgnn_classifier(qgnn_model, test_features, device)
    inference_time_s = time.perf_counter() - inference_start

    # T4.10: read back what train_qgnn.py measured/persisted at training time.
    # `.get(...)` defaults to None for checkpoints saved before T4.10 --
    # reported as "not available" below rather than fabricated (NFR-5).
    optimizer_name = qgnn_config.get("optimizer")
    train_accuracy = qgnn_config.get("train_accuracy")
    training_time_s = qgnn_config.get("training_time_s")

    wandb.log({f"test_{key}": val for key, val in metrics.items() if key != "confusion_matrix"})
    wandb.log({"inference_time_s": inference_time_s})
    if train_accuracy is not None:
        wandb.log({"train_accuracy": train_accuracy, "training_time_s": training_time_s})

    print("\nTest-set metrics:")
    for key in ("accuracy", "auc", "ap", "macro_f1", "precision", "recall"):
        print(f"  {key:12s}: {metrics[key]:.4f}")
    print(
        "  confusion_matrix (rows=true, cols=pred, label 0=quark/1=gluon): "
        f"{metrics['confusion_matrix']}"
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
