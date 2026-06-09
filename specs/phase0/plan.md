# Phase 0 — Plan

## Objective

Stand up the data, evaluation, and experiment infrastructure needed for all later phases. **No baseline models are implemented.** Baseline numbers come from a SOTA paper (citation TBD by author) that already benchmarks GAE, LGAE, ARGA, GIC, sGraph, GNAE, and VGNAE on the same datasets and splits we will use.

---

## Scope

### In scope
- Dataset loading and deterministic train/val/test splitting
- Evaluation metric implementations (AUC, AP, accuracy, bits-per-edge)
- Experiment config system and run logging
- Project directory skeleton

### Out of scope
- Implementing any graph autoencoder model
- Reproducing baseline numbers from scratch (reported numbers are sufficient)

---

## Tasks

### T0.1 — Project skeleton
Create the top-level directory structure:
```
src/gvls/
  data/        # dataset wrappers and split utilities
  eval/        # metric functions
  models/      # empty, populated in Phase 1
  losses/      # empty, populated in Phase 1
experiments/   # one script per paper experiment
configs/
  data/
  train/
  model/
  experiment/
tests/
```

### T0.2 — Data pipeline
- Wrap PyG `Planetoid` for Cora, CiteSeer, PubMed
- Wrap PyG `TUDataset` for MUTAG, PROTEINS, IMDB-B (used from Phase 2 onward; load only)
- Implement deterministic edge splitting: given a training ratio r ∈ {0.2, 0.4, 0.8}, split edges into train/val/test with a fixed seed; negative edges sampled at 1:1 ratio
- Splits must be reproducible: same seed → same split on every run

### T0.3 — Evaluation metrics
Implement in `src/gvls/eval/metrics.py`:
- `auc_ap(y_true, y_score) → (auc, ap)` — wraps `sklearn.metrics.roc_auc_score` and `average_precision_score`
- `node_accuracy(y_true, y_pred) → float`
- `bits_per_edge(adj_true, adj_logits) → float` — binary cross-entropy in nats converted to bits, normalized by number of edges

All functions take numpy arrays or tensors; return Python floats.

### T0.4 — Config and logging
- Hydra config structure under `configs/` with one base config per concern (data, train, model)
- W&B run initialized from Hydra config; logs dataset name, split ratio, seed, and all metrics
- A minimal `experiments/smoke_test.py` that loads each dataset, runs the split, and logs metric placeholders — confirms the full stack works end-to-end

---

## Deliverable

A single `experiments/smoke_test.py` run that:
1. Loads Cora, CiteSeer, PubMed at all three training ratios
2. Computes AUC/AP on a dummy all-ones predictor (expected AUC ≈ 0.5)
3. Logs results to W&B without error
