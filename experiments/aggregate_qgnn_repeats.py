"""Aggregate repeated QGNN evaluation runs into mean +/- std (T4.10 seed-repeat study).

The Lorentz-EQGNN literature baseline (sota-table.png Table II, Quark-Gluon,
4 qubits) reports test accuracy as a mean +/- std (74.00% +/- 0.26%),
implying multiple training runs. A single seeded run of our own pipeline is
not directly comparable to that -- this script reads N per-seed results JSON
files (as written by evaluate_qgnn.py, one per QGNN-training repeat) and
computes the same mean +/- std for each metric.

Only the QGNN classifier is retrained per repeat -- the frozen GVLS
checkpoint (and the train/val/test jet split) is shared across all repeats,
since GVLS pretraining is a deterministic, frozen upstream feature extractor
(Design Decision 8; extract_latent_features has no gradient). This means the
reported std captures the QGNN classifier's own training-seed variance
(circuit weight init, minibatch order, SPSA gradient noise) only, not full
end-to-end pipeline variance -- an explicit, documented scope narrowing
(NFR-5), not an oversight. See specs/phase4/validation.md V-10.

Usage:
    python experiments/aggregate_qgnn_repeats.py \
        "results/qgnn/qg_jets_metrics_lorentz800_seed*.json" \
        --output results/qgnn/qg_jets_metrics_lorentz800_summary.json
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

import wandb

NUMERIC_KEYS = [
    "test_accuracy",
    "test_accuracy_fixed_threshold_0.5",  # T4.10 followup, validation.md V-11
    "threshold",
    "auc",
    "ap",
    "macro_f1",
    "precision",
    "recall",
    "train_accuracy",
    "training_time_s",
    "inference_time_s",
]

LORENTZ_EQGNN_TEST_ACCURACY = 0.7400
LORENTZ_EQGNN_TEST_ACCURACY_STD = 0.0026


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help="glob pattern matching per-seed results JSON files")
    parser.add_argument("--output", required=True, help="path to write the aggregated summary JSON")
    parser.add_argument("--wandb-project", default="graph-vls")
    parser.add_argument(
        "--wandb-mode", default="offline", choices=["offline", "online", "disabled"]
    )
    parser.add_argument("--wandb-group", default="qgnn-jet-classification")
    parser.add_argument("--wandb-name", default="qgnn-repeats-summary")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    runs = [json.loads(Path(p).read_text()) for p in paths]
    print(f"Aggregating {len(runs)} runs:")
    for p in paths:
        print(f"  {p}")

    summary: dict = {"num_runs": len(runs), "source_files": [str(p) for p in paths]}
    for key in NUMERIC_KEYS:
        values = [r[key] for r in runs if r.get(key) is not None]
        if len(values) != len(runs):
            # A run predates a metric (e.g. checkpoints saved before T4.10
            # lack train_accuracy/training_time_s) -- skip rather than
            # silently averaging over a partial set (NFR-5).
            continue
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_values"] = values

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    if "test_accuracy_mean" in summary:
        print(
            f"Test accuracy: {summary['test_accuracy_mean']:.4f} +/- "
            f"{summary['test_accuracy_std']:.4f}  (n={len(runs)})"
        )
        print(
            f"Lorentz-EQGNN (literature): {LORENTZ_EQGNN_TEST_ACCURACY:.4f} +/- "
            f"{LORENTZ_EQGNN_TEST_ACCURACY_STD:.4f}  "
            "(sota-table.png Table II, Quark-Gluon, 4 qubits)"
        )
    if "test_accuracy_fixed_threshold_0.5_mean" in summary:
        # T4.10 followup (validation.md V-11): how much validation-selected
        # threshold tuning moved the reported number, averaged across trials.
        fixed_mean = summary["test_accuracy_fixed_threshold_0.5_mean"]
        delta = summary["test_accuracy_mean"] - fixed_mean
        print(
            f"Fixed threshold=0.5 accuracy (for comparison): {fixed_mean:.4f} +/- "
            f"{summary['test_accuracy_fixed_threshold_0.5_std']:.4f}  "
            f"(delta from tuned: {delta:+.4f})"
        )
    print(f"Written to {out_path}")

    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            name=args.wandb_name,
            group=args.wandb_group,
            job_type="summary",
            config={"num_runs": len(runs), "source_files": summary["source_files"]},
        )
        wandb.log(
            {
                k: v
                for k, v in summary.items()
                if isinstance(v, (int, float))
            }
        )
        wandb.log(
            {
                "lorentz_eqgnn_test_accuracy": LORENTZ_EQGNN_TEST_ACCURACY,
                "lorentz_eqgnn_test_accuracy_std": LORENTZ_EQGNN_TEST_ACCURACY_STD,
            }
        )
        artifact = wandb.Artifact(name="qgnn-repeats-summary", type="evaluation-summary")
        artifact.add_file(str(out_path))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
