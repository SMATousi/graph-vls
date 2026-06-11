# Phase 2 — Validation

## Exit Criteria

---

## V-1: Search Space and Config ⬜

| Check | Pass condition | Result |
|---|---|---|
| `suggest_config` returns correct keys | Output dict has all keys from `gvls.yaml` schema | ⬜ |
| `lambda_` conditional | `lambda_` is present when `prior=graph_mrf`; fixed to 1.0 when `prior=isotropic` | ⬜ |
| All categorical values valid | `graph_method` ∈ {attention, fgp}, `prior` ∈ {isotropic, graph_mrf}, etc. | ⬜ |

---

## V-2: Objective Function ⬜

| Check | Pass condition | Result |
|---|---|---|
| Single trial completes | One Optuna trial on Cora (20 epochs) returns a finite float | ⬜ |
| Pruning fires | A trial with bad intermediate AUC raises `TrialPruned` when `should_prune()` is called | ⬜ |
| Best val AUC returned | Returned value equals the maximum val AUC observed across all epochs of the trial | ⬜ |

---

## V-3: NAS Entry Point ⬜

| Check | Pass condition | Result |
|---|---|---|
| Study created | `optuna_studies/cora.db` exists after first run | ⬜ |
| Study resumable | Running nas.py twice appends trials rather than starting fresh | ⬜ |
| Top-5 table printed | stdout contains a ranked table of completed trials | ⬜ |
| W&B run logged | Single W&B offline run with `best_val_auc`, `n_trials_completed`, best config | ⬜ |

---

## V-4: Cora Search Results ⬜

| Check | Pass condition | Result |
|---|---|---|
| 50 trials complete | Study has ≥ 50 completed trials (pruned trials are excluded from this count) | ⬜ |
| Pruner is active | ≥ 20% of trials pruned at the halfway checkpoint | ⬜ |
| Best val AUC | At least one trial achieves val AUC > 0.7 | ⬜ |
| `configs/best/cora.yaml` written | File exists and is a valid `gvls.yaml`-schema config | ⬜ |
| Retrain reproducibility | Retraining `train_gvls.py` with best config achieves val AUC within ±0.02 of best trial | ⬜ |

---

## V-5: CiteSeer and PubMed (manual) ⬜

These runs are executed manually by the researcher; not gated by CI.

| Check | Pass condition | Result |
|---|---|---|
| CiteSeer: 50 trials complete | `optuna_studies/citeseer.db` populated | ⬜ |
| CiteSeer: best config saved | `configs/best/citeseer.yaml` written | ⬜ |
| PubMed: 50 trials complete | `optuna_studies/pubmed.db` populated | ⬜ |
| PubMed: best config saved | `configs/best/pubmed.yaml` written | ⬜ |

---

## V-6: Code Quality ⬜

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All tests pass (including `test_nas.py`) | ⬜ |
| `ruff check src/` | Zero violations | ⬜ |
| `optuna>=3.0` in `pyproject.toml` | Dependency declared | ⬜ |
