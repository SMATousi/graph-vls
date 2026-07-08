# Phase 2 — Plan

## Objective

Run a hyperparameter and architecture search over the GVLS model using Optuna, producing a best configuration per dataset. The search covers both architectural choices (latent graph method, message-passing rounds, latent dimension) and training choices (lr, beta, prior). Cora is the automated benchmark; CiteSeer and PubMed are run manually by the researcher.

---

## Scope

### In scope
- Optuna TPE search with MedianPruner over the search space defined in `requirements.md`
- Automated run on Cora; manual runs on CiteSeer and PubMed
- Best config persisted per dataset and loadable by `train_gvls.py`
- W&B summary run per NAS session

### Out of scope
- Multi-objective optimisation (e.g., jointly optimise AUC and training time)
- NAS over decoder architectures (inner-product decoder is fixed)
- Graph-level or node classification tasks (Phase 3)

---

## File Map

```
src/gvls/
  nas/
    search_space.py     # T2.1 — suggest_* helpers that populate a trial config
    objective.py        # T2.2 — Optuna objective: train GVLS, return best val AUC
configs/
  nas/
    default.yaml        # T2.1 — n_trials, epochs_per_trial, timeout, include_nri
  nas_config.yaml       # T2.1 — root Hydra config for nas.py
  best/                 # T2.4 — written by nas.py; one yaml per dataset
experiments/
  nas.py                # T2.3 — Hydra entry point, creates study, runs trials
optuna_studies/         # T2.4 — SQLite db files, one per dataset (gitignored)
tests/
  test_nas.py           # T2.1–T2.3 — smoke test: 3 trials on Cora, 20 epochs
```

---

## Tasks

### T2.1 — Search space and config

**File:** `src/gvls/nas/search_space.py`

Implement `suggest_config(trial) -> dict`:
- Calls `trial.suggest_*` for every parameter in the search space (FR-2)
- `lambda_` is only suggested if `prior == 'graph_mrf'`; otherwise fixed to 1.0
- Returns a flat `dict` with the same keys as `configs/model/gvls.yaml`

**File:** `configs/nas/default.yaml`
```yaml
n_trials: 50
epochs_per_trial: 100
timeout: 7200          # seconds; safety cap (2 hours)
include_nri: false     # enable NRI graph method (requires ~2 GB RAM for Cora)
study_dir: optuna_studies
```

**File:** `configs/nas_config.yaml`
```yaml
defaults:
  - data: cora
  - train: default
  - nas: default
  - _self_

wandb:
  project: graph-vls
  mode: offline
```

---

### T2.2 — Objective function

**File:** `src/gvls/nas/objective.py`

Implement `make_objective(data, split, cfg, device) -> Callable[[Trial], float]`:

Returns a closure `objective(trial)`:
1. Call `suggest_config(trial)` to get hyperparameters
2. Build GVLS model from suggested config
3. Train for `cfg.nas.epochs_per_trial` epochs using the same loop as `train_gvls.py`
   - pos_weight computed once outside the trial (from adj_true)
   - W&B logging disabled inside trials
4. At `epochs_per_trial // 2`, call `trial.report(val_auc, step)` and `trial.should_prune()`; raise `optuna.TrialPruned` if pruned
5. Return the best val AUC seen across all epochs of the trial

Design notes:
- Data and split are loaded once outside the closure; only model and optimizer are recreated per trial
- `adj_true` and `pos_weight` are computed once and shared across trials
- Each trial seeds with `trial.number * 42` before model init and training

---

### T2.3 — NAS entry point

**File:** `experiments/nas.py`

Hydra entry point using `nas_config.yaml`:

1. Load dataset and split edges (reuse Phase 0 utilities)
2. Create or load Optuna study:
   ```python
   study = optuna.create_study(
       study_name=f"gvls-{dataset_name}",
       storage=f"sqlite:///{cfg.nas.study_dir}/{dataset_name}.db",
       direction="maximize",
       sampler=optuna.samplers.TPESampler(seed=cfg.train.seed),
       pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
       load_if_exists=True,
   )
   ```
3. Run `study.optimize(objective, n_trials=cfg.nas.n_trials, timeout=cfg.nas.timeout)`
4. Print top-5 trials table
5. Save best config to `configs/best/{dataset_name}.yaml`
6. Log W&B summary run

---

### T2.4 — Run on Cora and store results ✅

Execute:
```bash
python experiments/nas.py data=cora
```

Collect `configs/best/cora.yaml`. Verify that retraining `train_gvls.py` with
`model=best/cora` achieves val AUC within ±0.02 of the best trial's val AUC.

CiteSeer and PubMed: run manually (`data=citeseer`, `data=pubmed`) when compute is available.

**Completed 2026-06-11:** CiteSeer NAS ran 2026-06-10 (50 trials, best val AUC=0.9407); PubMed NAS ran 2026-06-11 (51 trials, best val AUC=0.9518). Both `configs/best/{citeseer,pubmed}.yaml` written and used for full retraining across all split ratios — see `specs/phase2/validation.md` V-5.

---

## Deliverable ✅

`experiments/nas.py data=cora` completes 50 trials without crashing, at least one trial achieves val AUC > 0.7, `configs/best/cora.yaml` is written, and the Optuna SQLite database is populated.

**Extended beyond original scope:** the same pipeline was also run manually for CiteSeer and PubMed, completing the full three-dataset NAS deliverable ahead of the roadmap's original per-phase split (Cora automated / CiteSeer & PubMed deferred). See `specs/phase2/validation.md` V-5 and V-6.
