# Phase 3 — Requirements

## Functional Requirements

### FR-1: Compression metrics
- `reconstruction_f1(adj_true, adj_logits, threshold=0.5) -> float`: binary F1 over the upper-triangle node-pair predictions
- `dim_compression_ratio(d, F) -> float`: `d / F`, where `F` is input feature dimensionality
- `edge_compression_ratio(A_z, num_input_edges) -> float`: non-zero upper-triangle count in A_z divided by input edge count `|E|`
- `bits_per_edge` (Phase 0) must be reusable over the full node-pair set; for graphs where `N²` is intractable to materialize densely (PubMed, N=19717), must support a sampled estimate over a large random subset of pairs, with the sample size configurable
- All metric functions accept numpy arrays or PyTorch tensors, consistent with Phase 0's `eval/metrics.py` conventions

### FR-2: Full-graph split mode
- `full_graph_split(data, seed) -> EdgeSplit`: all real edges become `train_edge_index` (both directions); no val/test edges are produced
- Must be deterministic given a seed
- Distinct from `split_edges` (Phase 0), which is retained unchanged for link-prediction use (Phase 1/2)

### FR-3: Rate-distortion sweep
- Must sweep `latent_dim ∈ {4, 8, 16, 32, 64, 128}` × `k ∈ {1, 2, 3, 5, 10, 20}` (36 combinations per dataset)
- All other hyperparameters (`graph_method`, `prior`, `mp_rounds`, `lr`, `beta`, `lambda_`) are fixed per dataset to that dataset's Phase 2 NAS-best config (`configs/best/{dataset}.yaml`) — only `latent_dim` and `k` vary in the sweep
- Each grid point trains on the full-graph split (FR-2) for a fixed epoch budget (200, matching Phase 1's default) — no early stopping, since there is no held-out validation signal for a compression/memorization objective
- Must compute and persist, per grid point: `reconstruction_f1`, `bits_per_edge`, `dim_compression_ratio`, `edge_compression_ratio`, latent graph density (`|A_z|` non-zero fraction), and the raw counts (`N`, `F`, `|E|`, `d`, `k`, `|A_z|`)
- Results written to `results/compression/{dataset}.csv`, one row per grid point
- Must select and persist a compression-optimal config: smallest `(d, k)` pair (by `d/F + |A_z|/|E|`, ties broken by smallest `d`) with `reconstruction_f1 ≥ 0.90`, written to `configs/compression/{dataset}.yaml` in the same schema as `configs/model/gvls.yaml`
- Must log each run to W&B under group tag `compression-sweep-{dataset}`

### FR-4: Decoder fallback (conditional)
- Only implemented if T3.4's trigger fires: `reconstruction_f1` at `(latent_dim=128, k=20)` is below 0.90 for a given dataset
- If triggered: `LatentGraphDecoder` performs one additional message-passing round over A_z (same residual form as the Phase 1 latent message passing) with its own learned weight matrix, applied to z̃ before the inner-product decode
- Must report a head-to-head F1 comparison against the baseline inner-product decoder at the same `(d, k)`

### FR-5: Node classification (secondary)
- Loads a frozen, trained GVLS model (Phase 2 NAS-best config) and extracts z̃
- Trains a linear probe and a 2-layer MLP head (hidden width configurable, default 64) on z̃ → class logits
- Uses the standard Planetoid public semi-supervised split (20 labels/class)
- Reports accuracy and macro-F1 on the test split

---

## Non-Functional Requirements

### NFR-1: Reproducibility
- Fixed seed for every sweep grid point (`seed = dataset_seed`, no per-trial variation — this is a grid sweep, not a stochastic search)
- Same config + seed must reproduce the same rate-distortion CSV row within floating-point tolerance

### NFR-2: Scale
- PubMed's full node-pair count (`N²≈389M`) makes dense bits-per-edge computation memory-prohibitive; FR-1's sampled estimator must be used for PubMed, with sample size large enough that repeated runs agree within ±0.01 bits/edge
- Each grid point (200 epochs, full-graph) must complete in a reasonable wall-clock budget on CPU; if PubMed grid points are too slow, GPU or a reduced epoch budget for PubMed specifically is acceptable (document the deviation if taken)

### NFR-3: Test coverage
- Every new module (`compression.py`, `full_graph_split`, `compression_sweep.py`, conditionally `decoder.py`, `node_probe.py`) has at least one shape/correctness test
- Test budget: `test_compression_sweep.py` uses a 2×2 grid at 10 epochs so it completes in under 60 seconds

### NFR-4: Code style
- `ruff check src/` passes with zero warnings after each task
- New eval code lives under `src/gvls/eval/`; new data-split code extends `src/gvls/data/splits.py`; conditional decoder code lives under `src/gvls/models/`

---

## New Dependencies

No new top-level dependencies expected. `results/compression/*.csv` can be written with the standard library `csv` module or `pandas` if already convenient in the environment — no new dependency required either way.
