# Phase 3 — Requirements

## Functional Requirements

### FR-1: Compression metrics
- `reconstruction_f1(adj_true, adj_logits, threshold=0.5) -> float`: binary F1 over the upper-triangle node-pair predictions. **Unchanged by T3.6** — `adj_logits` is always the unpooled `N×N` reconstruction (`Â = S·σ(z̃_pooled z̃_pooledᵀ)·Sᵀ` when pooling is active, or the plain inner product when `M=N`), so this metric compares against the same full-size ground-truth adjacency regardless of whether pooling is enabled.
- `dim_compression_ratio(d, F) -> float`: `d / F`, where `F` is input feature dimensionality
- `edge_compression_ratio(A_z, num_input_edges) -> float`: non-zero upper-triangle count in A_z divided by input edge count `|E|`. **Note (T3.6):** once pooling is active, `A_z` is `M×M`, not `N×N` — this ratio still divides by the same input `|E|`, so it now reflects both the node-count and edge-density reduction together.
- `node_compression_ratio(M, N) -> float` **(new, T3.6)**: `M / N`, where `M` is the number of latent (pooled) nodes and `N` is the number of input nodes. Reported as a separate ratio, consistent with Design Decision 1 (no collapsed single score).
- `assignment_storage_bits(N, M) -> float` **(new, T3.6)**: `N * ceil(log2(M))`, the storage cost of the hardened (argmax) per-node cluster assignment — i.e. one index per input node into `M` clusters — used for honest total-size accounting since `S` is required to reconstruct the full graph and must be counted as part of the compressed representation, not treated as free.
- `bits_per_edge` (Phase 0) must be reusable over the full node-pair set; for graphs where `N²` is intractable to materialize densely (PubMed, N=19717), must support a sampled estimate over a large random subset of pairs, with the sample size configurable
- All metric functions accept numpy arrays or PyTorch tensors, consistent with Phase 0's `eval/metrics.py` conventions

### FR-2: Full-graph split mode
- `full_graph_split(data, seed) -> EdgeSplit`: all real edges become `train_edge_index` (both directions); no val/test edges are produced
- Must be deterministic given a seed
- Distinct from `split_edges` (Phase 0), which is retained unchanged for link-prediction use (Phase 1/2)

### FR-3: Rate-distortion sweep
- Must sweep `latent_dim ∈ {4, 8, 16, 32, 64, 128}` × `k ∈ {2, 3, 5, 10, 20}` (30 combinations per dataset). **`k=1` removed 2026-07-13**: at `k=1`, `LatentGraphLearner` keeps only each node's single highest-scoring neighbour, which after symmetrization produces a near-spanning-tree `A_z` (edge count ≈ `M−1`, confirmed empirically on PubMed's pooling rerun — see `specs/phase3/validation.md` V-8) rather than a meaningfully relational latent graph. With `reconstruction_f1` now flat across the whole `k` range (post the ELBO fix, V-3), `select_compression_optimal`'s fallback path had no signal to avoid it, so it kept winning by floating-point noise, not because `k=1` is a better latent graph.
- All other hyperparameters (`graph_method`, `prior`, `mp_rounds`, `lr`, `beta`, `lambda_`) are fixed per dataset to that dataset's Phase 2 NAS-best config (`configs/best/{dataset}.yaml`) — only `latent_dim` and `k` vary in the sweep
- Each grid point trains on the full-graph split (FR-2) for a fixed epoch budget (200, matching Phase 1's default) — no early stopping, since there is no held-out validation signal for a compression/memorization objective
- Must compute and persist, per grid point: `reconstruction_f1`, `bits_per_edge`, `dim_compression_ratio`, `edge_compression_ratio`, latent graph density (`|A_z|` non-zero fraction), and the raw counts (`N`, `F`, `|E|`, `d`, `k`, `|A_z|`)
- Results written to `results/compression/{dataset}.csv`, one row per grid point
- Must select and persist a compression-optimal config: smallest `(d, k)` pair (by `d/F + |A_z|/|E|`, ties broken by smallest `d`) with `reconstruction_f1 ≥ 0.90`, written to `configs/compression/{dataset}.yaml` in the same schema as `configs/model/gvls.yaml`
- Must log each run to W&B under group tag `compression-sweep-{dataset}`

### FR-4: A_z-conditioned decoder — **superseded 2026-07-09, revived 2026-07-13**
- Was to be built if T3.4's trigger fired: `reconstruction_f1` at `(latent_dim=128, k=20)` below 0.90. It fired for all three datasets, but the response was changed: instead of adding a second message-passing round to the decoder at fixed `M=N`, the project pivoted to node-count pooling (FR-6/FR-7) — kept as a record of that decision in `specs/phase3/plan.md`.
- **Revived 2026-07-13.** T3.6's pooling approach did not close the fidelity gap (all three datasets remain flat plateaus below 0.90 even after the ELBO KL-normalization fix — see `specs/phase3/validation.md` V-3), which is exactly the condition under which the original plan said to revisit this. Implemented as `LatentGraphDecoder` (`src/gvls/models/decoder.py`): one unconditional extra round of `LatentMessagePassing` over `A_z`, applied to `z_tilde` before the inner-product decode, giving `A_z` a guaranteed path into the reconstruction independent of the encoder's `mp_rounds` config (zero for CiteSeer's and PubMed's Phase 2 NAS-best configs, which is why `A_z` was inert or near-inert for those two datasets in the first place).
- `train_gvls_full_graph`/`evaluate_compression` (`src/gvls/compression/sweep.py`) take a `decoder: "inner_product" | "graph_conditioned"` parameter (default `"inner_product"`, unchanged behavior); `experiments/compression_sweep.py` exposes it as `experiment.decoder`. Head-to-head comparison at matched `(d,k)`: run the sweep once per decoder value. The `graph_conditioned` run writes to suffixed paths (`results/compression/{name}_graph_decoder.csv`, `configs/compression/{name}_graph_decoder.yaml`) so it doesn't clobber the `inner_product` baseline.
- **Status: implemented and unit-tested (`tests/test_decoder.py`, `tests/test_compression_sweep.py`); the production head-to-head sweep has not been run** — see `specs/phase3/validation.md` V-4.

### FR-5: Node classification (secondary)
- Loads a frozen, trained GVLS model (Phase 2 NAS-best config) and extracts z̃
- Trains a linear probe and a 2-layer MLP head (hidden width configurable, default 64) on z̃ → class logits
- Uses the standard Planetoid public semi-supervised split (20 labels/class)
- Reports accuracy and macro-F1 on the test split

### FR-6: Learned node-count pooling (new, T3.6)
- `LatentGraphPooling(input_dim, M, method="linear")`: computes per-node assignment logits (a learned linear layer on the encoder's `z`, one logit per cluster) and applies a row-wise softmax to produce `S ∈ [0,1]^{N×M}`
- Pools the per-node Gaussians into `M` cluster-level Gaussians via moment matching (mixture-of-Gaussians → single Gaussian per cluster), not a naive weighted average of means alone:
  - `μ_pooled = Sᵀμ` (weighted mean of assigned nodes' means, per cluster)
  - `σ²_pooled = Sᵀ(σ²) + Sᵀ(μ²) − μ_pooled²` (law of total variance: within-cluster variance plus between-node variance of the means, so the pooled Gaussian's variance reflects genuine uncertainty about which input nodes were merged, not just their average uncertainty)
- The latent graph learner (`LatentGraphLearner` from Phase 1) is reused unchanged, operating on the `M` pooled embeddings instead of `N` — i.e. `A_z` is now `M×M`
- Latent message passing (Phase 1's residual form) likewise reuses the existing implementation, now over the `M`-node `A_z`
- `M` is specified as a ratio of `N` (`M = round(pool_ratio * N)`, `pool_ratio ∈ {0.5, 0.25, 0.125, 0.0625}`), clamped to `M ≥ 2`
- **Required, not optional (added 2026-07-09 after an empirical cold-start collapse — see plan.md T3.6 and validation.md V-7):** two auxiliary training losses are needed alongside the ELBO, both standard components of DiffPool (Ying et al. 2018), or `S` collapses to a near-uniform blur that produces a degenerate, constant reconstruction regardless of `pool_ratio`:
  - `assignment_entropy(S) = mean_i[-Σ_m S_{i,m} log S_{i,m}]`: minimized during training to encourage each node's assignment to specialize rather than stay diffuse
  - `assignment_link_loss(S, A) = BCE(S·Sᵀ, A)`, excluding the diagonal (`S`'s self-similarity is naturally high for a confident assignment, but `A`'s diagonal is always 0 — including it would fight the entropy term): compares the assignment's *implied* clustering directly against the real input adjacency, giving `S` a gradient signal that doesn't have to travel through the entire pool → latent-graph → message-passing → unpool chain
  - Total training loss: `elbo(...) + entropy_weight · assignment_entropy(S) + aux_link_weight · assignment_link_loss(S, A)`, with `entropy_weight=0.1, aux_link_weight=5.0` as defaults (selected via a manual sweep on Cora past the point of diminishing returns — F1 rose from a degenerate 0.667 to ≈0.78 at `pool_ratio=0.5`)

### FR-7: Unpooling decode (new, T3.6)
- Reconstruction reuses the *same* `S` learned during pooling (not re-learned or re-derived at decode time): `Â (N×N) = S · (z̃_pooled z̃_pooledᵀ) · Sᵀ` — computed at the *logit* level (no sigmoid before unpooling), consistent with this codebase's convention of `elbo()` and `reconstruction_f1()` always operating on pre-sigmoid logits rather than probabilities; the sigmoid is applied once, by the caller, at eval time
- `reconstruction_f1` (FR-1) is computed on this unpooled `Â` against the true `N×N` adjacency — the metric itself is unchanged; only what feeds into `adj_logits` differs
- For storage-cost accounting (`assignment_storage_bits`, FR-1), `S` is hardened post-training via row-wise argmax to a single per-node cluster index — the soft weights are only needed during training for differentiable pooling, not for the final compressed artifact

### FR-8: Node-count sweep (new, T3.6)
- Must sweep `pool_ratio ∈ {0.5, 0.25, 0.125, 0.0625}` (i.e. `M/N`), holding `(d, k)` fixed at each dataset's T3.3 compression-optimal config (`configs/compression/{dataset}.yaml`) — this isolates node-count compression's effect on fidelity, independent of the dimensionality/edge-sparsity axes T3.3 already swept
- Must compute and persist, per grid point: everything FR-3 already persists, plus `node_compression_ratio`, `assignment_storage_bits`, and the raw count `M`
- `entropy_weight` and `aux_link_weight` (FR-6) are held fixed across the whole sweep (not swept per grid point) — they are a training-stability fix, not a compression axis under study
- Results written to `results/compression/{dataset}_pooling.csv`, one row per grid point
- Must log each run to W&B under group tag `compression-pooling-sweep-{dataset}`

---

## Non-Functional Requirements

### NFR-1: Reproducibility
- Fixed seed for every sweep grid point (`seed = dataset_seed`, no per-trial variation — this is a grid sweep, not a stochastic search)
- Same config + seed must reproduce the same rate-distortion CSV row within floating-point tolerance

### NFR-2: Scale
- PubMed's full node-pair count (`N²≈389M`) makes dense bits-per-edge computation memory-prohibitive; FR-1's sampled estimator must be used for PubMed, with sample size large enough that repeated runs agree within ±0.01 bits/edge
- Each grid point (200 epochs, full-graph) must complete in a reasonable wall-clock budget on CPU; if PubMed grid points are too slow, GPU or a reduced epoch budget for PubMed specifically is acceptable (document the deviation if taken)

### NFR-3: Test coverage
- Every new module (`compression.py`, `full_graph_split`, `compression_sweep.py`, `pooling.py`, `node_probe.py`) has at least one shape/correctness test
- Test budget: `test_compression_sweep.py` uses a 2×2 grid at 10 epochs so it completes in under 60 seconds
- `test_pooling.py` (T3.6) verifies: `S` rows sum to 1 (valid softmax assignment), pooled Gaussian shapes are `(M, d)`, gradients reach both the assignment logits and the pooled Gaussian parameters, unpooled `Â` has shape `(N, N)`, `assignment_entropy` is high for a uniform `S` and near-zero for a one-hot `S`, and `assignment_link_loss` is low when `S`'s implied clustering matches the true adjacency and high when it contradicts it

### NFR-4: Code style
- `ruff check src/` passes with zero warnings after each task
- New eval code lives under `src/gvls/eval/`; new data-split code extends `src/gvls/data/splits.py`; pooling code lives under `src/gvls/models/pooling.py`

---

## New Dependencies

No new top-level dependencies expected. `results/compression/*.csv` can be written with the standard library `csv` module or `pandas` if already convenient in the environment — no new dependency required either way.
