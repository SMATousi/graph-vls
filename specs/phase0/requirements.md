# Phase 0 — Requirements

## Functional Requirements

### FR-1: Dataset loading
- Must load Cora, CiteSeer, and PubMed via PyG `Planetoid` with a single dataset name argument
- Must load MUTAG, PROTEINS, IMDB-B via PyG `TUDataset` (for future phases; no processing required in Phase 0)
- Raw data must be cached locally after first download; subsequent runs must not re-download

### FR-2: Edge splitting
- Must support training ratios r ∈ {0.2, 0.4, 0.8}
- Positive test/val edges: held-out real edges not seen during training
- Negative test/val edges: non-existing edges sampled uniformly at 1:1 ratio with positives
- Split must be deterministic given a seed; seed must be configurable
- Train edges used for message passing; val/test edges used only for evaluation (no leakage)

### FR-3: Evaluation metrics
- `auc_ap`: takes binary labels and continuous scores; returns (AUC-ROC, AP) as floats in [0, 1]
- `node_accuracy`: takes integer class labels and predictions; returns accuracy as float
- `bits_per_edge`: takes adjacency ground truth and per-edge logits; returns mean bits per edge
- All metric functions must be unit-tested against known-correct values

### FR-4: Config management
- Each experiment run must be fully specified by a Hydra config (no hardcoded hyperparameters)
- CLI overrides must work: `python experiments/smoke_test.py data=cora train.split_ratio=0.4`
- Config must be logged as a W&B artifact on each run

### FR-5: Experiment logging
- Each run logs: dataset, split_ratio, seed, metric values
- Run group and tags configurable from CLI
- W&B project name: `graph-vls`

---

## Non-Functional Requirements

### NFR-1: Reproducibility
- Same config + seed must produce byte-identical splits and metric values across machines (no non-deterministic ops in Phase 0 code)

### NFR-2: Test coverage
- Metric functions: 100% line coverage
- Data split: at least one test verifying no train/test edge overlap and correct negative sampling ratio

### NFR-3: Code style
- `ruff` passes with zero warnings
- `mypy --strict` passes on `src/gvls/data/` and `src/gvls/eval/`

---

## Dependencies

```
torch>=2.1
torch_geometric>=2.4
scikit-learn>=1.3
hydra-core>=1.3
wandb>=0.16
pytest>=7.4
ruff>=0.1
mypy>=1.6
```

No model-specific dependencies (no `torch_scatter`/`torch_sparse` required until Phase 1).
