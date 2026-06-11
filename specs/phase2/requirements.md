# Phase 2 — Requirements

## Functional Requirements

### FR-1: Optuna study
- Must use Optuna ≥ 3.0 with the default TPE sampler
- Each trial instantiates a fresh GVLS model with hyperparameters drawn from the search space, trains it for `nas.epochs_per_trial` epochs on the target dataset, and returns the best val AUC observed during that trial as the objective
- The study maximises val AUC
- Each trial seeds PyTorch with `trial.number * 42` for reproducibility
- The study must be **resumable**: if `optuna_studies/{dataset}.db` already exists, additional trials are appended to the same study

### FR-2: Search space
Architecture parameters:

| Parameter | Type | Range / Choices |
|---|---|---|
| `latent_dim` | categorical | 16, 32, 64, 128 |
| `hidden_dim` | categorical | 32, 64, 128, 256 |
| `mp_rounds` | int | 0, 1, 2 |
| `graph_method` | categorical | attention, fgp (NRI excluded by default — see NFR-1) |
| `k` | categorical | 5, 10, 20, 50 |
| `prior` | categorical | isotropic, graph_mrf |

Training parameters:

| Parameter | Type | Range |
|---|---|---|
| `lr` | log-uniform float | [1e-4, 5e-2] |
| `beta` | log-uniform float | [1e-5, 0.1] |
| `lambda_` | log-uniform float | [0.1, 10.0] *(only suggested when `prior=graph_mrf`)* |

NRI can be enabled via `nas.include_nri=true` (off by default; requires sufficient RAM — see NFR-1).

### FR-3: Pruning
- Use Optuna `MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)`
- Report val AUC as an intermediate value at `epochs_per_trial // 2` (the halfway checkpoint)
- Trials that are pruned at this checkpoint still count toward `n_trials` for progress tracking

### FR-4: Output
- Best trial config saved to `configs/best/{dataset_name}.yaml` after the study completes; this file must be loadable as `configs/model/gvls.yaml` (same schema)
- Full Optuna study persisted to `optuna_studies/{dataset_name}.db` (SQLite backend)
- Top-5 trials (by val AUC) printed to stdout in a table

### FR-5: W&B summary
- A single W&B run per NAS session logs: `best_val_auc`, `n_trials_completed`, `n_trials_pruned`, and the best hyperparameter config as a flat dict

### FR-6: Hydra config
- NAS script uses its own Hydra root config `configs/nas_config.yaml`
- NAS-specific settings live in `configs/nas/default.yaml`
- Dataset, train split, and seed are inherited from the existing `configs/data/` and `configs/train/` groups

---

## Non-Functional Requirements

### NFR-1: Speed
- NRI graph method excluded from default search space because the O(N²·2d) pair tensor requires ≈1.9 GB for Cora (N=2708, d=128) and would OOM or slow the search dramatically
- Each trial on Cora at `epochs_per_trial=100` must complete in under 90 seconds on CPU

### NFR-2: Test budget
- The test suite uses `n_trials=3, epochs_per_trial=20` so NAS tests complete in under 60 seconds total

### NFR-3: Code style
- `ruff check src/` passes with zero violations after Phase 2 code is added

---

## New Dependencies

```
optuna>=3.0
```

Add to `[project] dependencies` in `pyproject.toml`.
