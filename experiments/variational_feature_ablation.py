"""T5.2 (specs/phase5/): what is the pooled posterior's variance actually worth?

`extract_latent_features` used to keep only `(z_tilde, A_z)` and discard the
pooled posterior's `mu`/`log_var`. Since extraction runs under `model.eval()`,
`z_tilde` is the deterministic mean path -- so no classifier in this project
had ever seen a variance, and GVLS's variational machinery was severed from
the metric it is supposed to improve. That is a large part of why raising
`beta` only ever cost accuracy in the (k, beta, prior) sweep
(specs/phase5/validation.md V-0 Part B): the model pays the regularization
and the metric receives none of the information.

This script measures the ablation FR-2 requires, under exactly the protocol
`classical_baseline_diagnostic.py` uses (same frozen checkpoint, same 5
balanced-800 training subsets, same fixed 1250-jet test set), so its numbers
drop straight into the same comparison table:

    z_a            z_tilde + A_z upper triangle  (pre-T5.2 baseline)
    z_a_logvar     + pooled log-variance
    z_a_mu_logvar  + pooled mean as well
    logvar_only    the variance alone -- does it carry class information?

Reported against both bars this phase is judged on (NFR-1): the current GVLS
baseline and the `N`-only bar (logistic regression on particle count alone).

Requires a trained GVLS checkpoint -- run experiments/pretrain_gvls_jets_final.py
(or scripts/run_qgnn_lorentz_comparability.sh) first.

Usage:
    python experiments/variational_feature_ablation.py
    python experiments/variational_feature_ablation.py \
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

# specs/phase5/validation.md V-0 Part C. NFR-1: every task reports against
# both, not only against GVLS's own history.
N_ONLY_BAR = 0.7552  # logistic regression on particle count alone
GVLS_BASELINE = 0.7034  # (z_tilde, A_z), the pre-T5.2 feature set


@hydra.main(
    version_base=None, config_path="../configs", config_name="variational_feature_ablation_config"
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")  # classical models only

    print(f"Loading frozen GVLS checkpoint from {cfg.gvls_checkpoint_path}...")
    gvls_model, gvls_config = load_gvls_checkpoint(str(cfg.gvls_checkpoint_path), device)
    print(f"  GVLS M={int(gvls_config['num_clusters'])}, d={int(gvls_config['latent_dim'])}")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    seeds = list(cfg.train_subset_seeds)
    feature_sets = list(cfg.feature_sets)
    print(f"{len(feature_sets)} feature sets x {len(seeds)} trials (seeds={seeds})")

    # Extract once per trial and reuse across feature sets: the extraction is
    # identical, only which columns get handed to the classifier changes. This
    # also means every feature set sees the *same* frozen representation, so
    # differences are attributable to the columns and nothing else.
    test_features = None
    train_features_by_seed = {}
    for seed in seeds:
        trial_cfg = dict(data_cfg)
        trial_cfg["train_subset_seed"] = int(seed)
        split = load_split_from_config(trial_cfg)
        if test_features is None:
            test_features = extract_latent_features(gvls_model, split.test, device)
        train_features_by_seed[int(seed)] = extract_latent_features(
            gvls_model, split.train, device
        )

    summary: dict = {"num_trials": len(seeds), "seeds": seeds, "feature_sets": {}}
    for feature_set in feature_sets:
        per_model: dict[str, list[dict]] = {"logreg": [], "mlp": []}
        for seed in seeds:
            trial_metrics = evaluate_classical_baselines(
                train_features_by_seed[int(seed)],
                test_features,
                seed=int(seed),
                mlp_hidden_units=int(cfg.mlp_hidden_units),
                feature_set=feature_set,
            )
            for name, metrics in trial_metrics.items():
                per_model[name].append(metrics)

        entry: dict = {}
        for name, rows in per_model.items():
            for key in ("accuracy", "auc", "macro_f1"):
                values = [row[key] for row in rows]
                entry[f"{name}_{key}_mean"] = statistics.mean(values)
                entry[f"{name}_{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary["feature_sets"][feature_set] = entry
        print(
            f"  {feature_set:15s} logreg={entry['logreg_accuracy_mean']:.4f}"
            f"+/-{entry['logreg_accuracy_std']:.4f}  mlp={entry['mlp_accuracy_mean']:.4f}"
            f"  auc={entry['logreg_auc_mean']:.4f}"
        )

    print("\n=== T5.2: variational-output ablation (specs/phase5/) ===")
    print(f"{'feature set':<16}{'logreg':>10}{'std':>9}{'mlp':>10}{'auc':>9}  vs baseline")
    baseline = summary["feature_sets"].get("z_a", {}).get("logreg_accuracy_mean")
    for feature_set, entry in summary["feature_sets"].items():
        delta = (
            f"{entry['logreg_accuracy_mean'] - baseline:+.4f}" if baseline is not None else "n/a"
        )
        print(
            f"{feature_set:<16}{entry['logreg_accuracy_mean']:>10.4f}"
            f"{entry['logreg_accuracy_std']:>9.4f}{entry['mlp_accuracy_mean']:>10.4f}"
            f"{entry['logreg_auc_mean']:>9.4f}  {delta}"
        )

    # The `*_n` / `n_only` rows are the T5.3 *control*: they contain the raw
    # particle count. Judging "did we clear the N-only bar?" using a feature
    # set that contains N is circular -- an earlier version of this block did
    # exactly that and reported the bar cleared on the strength of `z_a_n`.
    # The bar is about what the *latent representation* encodes.
    encoded = {
        name: entry
        for name, entry in summary["feature_sets"].items()
        if not (name.endswith("_n") or name == "n_only")
    }
    best_encoded_name, best_encoded = max(
        encoded.items(), key=lambda kv: kv[1]["logreg_accuracy_mean"]
    )
    best_acc = best_encoded["logreg_accuracy_mean"]
    best_std = best_encoded["logreg_accuracy_std"]

    print(f"\nBars (NFR-1): GVLS baseline {GVLS_BASELINE:.4f}, N-only {N_ONLY_BAR:.4f}")
    print(f"Best ENCODED feature set (no raw N): {best_encoded_name} = {best_acc:.4f}")
    if best_acc - best_std > N_ONLY_BAR:
        print("-> the encoded representation clears the N-only bar outright.")
    elif best_acc + best_std > N_ONLY_BAR:
        print(
            f"-> the encoded representation ties the N-only bar "
            f"({best_acc:.4f} +/- {best_std:.4f} vs {N_ONLY_BAR:.4f}) -- a tie, not a win."
        )
    elif baseline is not None and best_acc > baseline:
        print(
            f"-> improves on the GVLS baseline ({baseline:.4f}) but is still below the "
            f"N-only bar; the latent does not yet encode what fixed-M pooling discards."
        )
    else:
        print(f"-> no encoded feature set improves on the GVLS baseline ({baseline}).")

    controls = {k: v for k, v in summary["feature_sets"].items() if k not in encoded}
    if controls:
        best_control_name, best_control = max(
            controls.items(), key=lambda kv: kv[1]["logreg_accuracy_mean"]
        )
        gap = best_control["logreg_accuracy_mean"] - best_acc
        print(
            f"Control (raw N appended): {best_control_name} = "
            f"{best_control['logreg_accuracy_mean']:.4f}, "
            f"{'still ahead by' if gap > 0 else 'behind by'} {abs(gap):.4f}"
        )

    results_path = Path(str(cfg.results_path))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {results_path}")


if __name__ == "__main__":
    main()
