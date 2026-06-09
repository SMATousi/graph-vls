# Phase 0 — Plan ✅ Completed 2026-06-09

## Objective

Stand up the data, evaluation, and experiment infrastructure needed for all later phases. **No baseline models are implemented.** Baseline numbers come from Ahn & Kim, "Variational Graph Normalized Autoencoders," CIKM 2021, which benchmarks GAE, LGAE, ARGA, GIC, sGraph, GNAE, and VGNAE on the same datasets and splits we will use.

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

### T0.1 — Project skeleton ✅
Created the top-level directory structure:
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
notebooks/
```
Also added `pyproject.toml` (setuptools build, ruff + mypy config) and updated `.gitignore`.

**Fix applied:** `.gitignore` originally used `data/` (matched `src/gvls/data/`); corrected to `/data/` to anchor to repo root.

### T0.2 — Data pipeline ✅
- `src/gvls/data/datasets.py`: `load_planetoid(name)` and `load_tu_dataset(name)` wrappers
- `src/gvls/data/splits.py`: `split_edges(data, train_ratio, seed)` returning an `EdgeSplit` dataclass
  - Canonical undirected edges (i < j) shuffled with seeded RNG
  - `train_edge_index` stores both directions for message passing
  - Negative edges sampled via rejection, added to `pos_set` to prevent duplicates
- 11 unit tests in `tests/test_data.py` — all pass

**Bug found and fixed:** initial test graph (10 nodes, ~30 edges) was too dense for ratio=0.2 — the negative sampler entered an infinite loop because only 15 non-existing edges were available but 24 were needed. Fixed by switching to a 50-node sparse graph (path + skip-2, 97 edges, 1128 non-existing).

### T0.3 — Evaluation metrics ✅
Implemented in `src/gvls/eval/metrics.py`:
- `auc_ap(y_true, y_score) → (float, float)` — sklearn wrappers
- `node_accuracy(y_true, y_pred) → float`
- `bits_per_edge(adj_true, adj_logits) → float` — numerically stable BCE via `max(l,0) - y·l + log1p(exp(-|l|))`, divided by log(2)

All accept numpy arrays or PyTorch tensors. 15 unit tests in `tests/test_metrics.py` — all pass.

### T0.4 — Config and logging ✅
- Hydra configs: `configs/config.yaml` (root), `configs/data/{cora,citeseer,pubmed}.yaml`, `configs/train/default.yaml`, `configs/model/default.yaml`
- `experiments/smoke_test.py`: loads dataset, splits edges, runs dummy random predictor, logs all metrics to W&B
- W&B defaults to `mode: offline`; switch to `online` via `wandb.mode=online`

---

## Deliverable ✅

`experiments/smoke_test.py` verified on Cora at default settings (80% train, seed 42):

```
nodes=2708  train=4222  val=528  test=528
val  auc=0.4981  ap=0.4931
test auc=0.5018  ap=0.5092  bpe=1.0000
```

Multirun CLI (`-m data=cora,citeseer,pubmed train.split_ratio=0.2,0.4,0.8`) is wired up and ready.
