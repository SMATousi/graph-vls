"""Classical-baseline diagnostic on frozen GVLS features (T4.10 followup, validation.md V-11).

After fixing GVLS's data-starved pretraining (validation.md V-11), the QGNN's
5-trial mean test accuracy rose from ~59.1% to ~66.9% -- real progress, but
still ~7 points short of Lorentz-EQGNN's 74.00%, and train accuracy (~69.8%)
isn't saturating either. This script answers *where* that remaining gap
sits: does a plain classical classifier trained on the exact same frozen
`(z_tilde, A_z)` features (same 5 per-trial 800-jet training subsets, same
fixed test set as the QGNN comparability run) also plateau around ~67-70%
(ceiling is upstream in GVLS's compression -- the features aren't more
separable than that, regardless of classifier), or does it do meaningfully
better (the QGNN's own training -- circuit capacity, SPSA gradient noise --
is leaving accuracy on the table)?

Two classifiers per trial: `LogisticRegression` (a linear ceiling) and a
shallow `MLPClassifier` (one small hidden layer, roughly matching the QGNN's
own shallow single re-uploading layer in spirit). See
`gvls/eval/classical_baseline.py` for the actual fitting/scoring logic.

Requires the trained GVLS checkpoint (checkpoints/gvls_jets_m4_lorentz800.pt
by default) -- run experiments/pretrain_gvls_jets_final.py (or the full
scripts/run_qgnn_lorentz_comparability.sh pipeline) first if it doesn't
exist yet.

Usage:
    python experiments/classical_baseline_diagnostic.py
    python experiments/classical_baseline_diagnostic.py \
        gvls_checkpoint_path=checkpoints/gvls_jets_m6_lorentz800.pt
"""

import json
import statistics
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gvls.compression.jet_sweep import load_gvls_checkpoint
from gvls.data.jets import load_split_from_config
from gvls.eval.classical_baseline import evaluate_classical_baselines
from gvls.qgnn_training import extract_latent_features

# Reference point: results/qgnn/qg_jets_metrics_lorentz800_summary.json's
# test_accuracy_mean/_std after the GVLS-full-pool-pretraining fix.
QGNN_TEST_ACCURACY_MEAN = 0.6688
QGNN_TEST_ACCURACY_STD = 0.0030


@hydra.main(
    version_base=None, config_path="../configs", config_name="qgnn_classical_baseline_config"
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")  # classical models only -- no benefit from GPU/quantum sim here

    print(f"Loading frozen GVLS checkpoint from {cfg.gvls_checkpoint_path}...")
    gvls_model, gvls_config = load_gvls_checkpoint(str(cfg.gvls_checkpoint_path), device)
    m = int(gvls_config["num_clusters"])
    print(f"  GVLS M={m}")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    seeds = list(cfg.train_subset_seeds)
    print(f"Running {len(seeds)} trials (seeds={seeds}), mirroring the QGNN comparability sweep...")

    per_model: dict[str, list[dict]] = {"logreg": [], "mlp": []}
    test_features = None
    for seed in seeds:
        trial_cfg = dict(data_cfg)
        trial_cfg["train_subset_seed"] = int(seed)
        split = load_split_from_config(trial_cfg)
        if test_features is None:
            # val/test come from the outer (data.seed-fixed) partition and
            # are identical across trials -- extract once, reuse.
            test_features = extract_latent_features(gvls_model, split.test, device)
        train_features = extract_latent_features(gvls_model, split.train, device)

        trial_metrics = evaluate_classical_baselines(
            train_features,
            test_features,
            seed=int(seed),
            mlp_hidden_units=int(cfg.mlp_hidden_units),
        )
        for name, metrics in trial_metrics.items():
            per_model[name].append(metrics)
        print(
            f"  seed={seed}  logreg_acc={trial_metrics['logreg']['accuracy']:.4f}  "
            f"mlp_acc={trial_metrics['mlp']['accuracy']:.4f}"
        )

    summary: dict = {"num_trials": len(seeds), "seeds": seeds}
    for name, rows in per_model.items():
        for key in ("accuracy", "auc", "macro_f1"):
            values = [row[key] for row in rows]
            summary[f"{name}_{key}_mean"] = statistics.mean(values)
            summary[f"{name}_{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

    print("\n=== Classical baseline diagnostic (T4.10 followup, validation.md V-11) ===")
    print(
        f"Logistic regression: accuracy={summary['logreg_accuracy_mean']:.4f} +/- "
        f"{summary['logreg_accuracy_std']:.4f}  auc={summary['logreg_auc_mean']:.4f}"
    )
    print(
        f"Shallow MLP ({cfg.mlp_hidden_units} hidden units): "
        f"accuracy={summary['mlp_accuracy_mean']:.4f} +/- {summary['mlp_accuracy_std']:.4f}  "
        f"auc={summary['mlp_auc_mean']:.4f}"
    )
    print(
        f"QGNN (for comparison): accuracy={QGNN_TEST_ACCURACY_MEAN:.4f} +/- "
        f"{QGNN_TEST_ACCURACY_STD:.4f}"
    )
    best_classical = max(summary["logreg_accuracy_mean"], summary["mlp_accuracy_mean"])
    if best_classical <= QGNN_TEST_ACCURACY_MEAN + QGNN_TEST_ACCURACY_STD:
        print(
            "\n-> Classical baselines do NOT clearly beat the QGNN: the ~67% ceiling looks "
            "like a GVLS feature-separability limit, not a QGNN training deficiency."
        )
    else:
        print(
            "\n-> A classical baseline clearly beats the QGNN: the frozen features support "
            "higher accuracy than the QGNN is currently extracting -- look at QGNN training "
            "(gradient_method, num_layers) next, not GVLS."
        )

    results_path = Path(str(cfg.results_path))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {results_path}")


if __name__ == "__main__":
    main()
