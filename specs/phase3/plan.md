# Phase 3 — Plan

## Objective

Build and evaluate the **graph compression** pipeline: quantify how compact the GVLS latent representation (z̃, A_z) is relative to the input graph (X, A), and trace a rate-distortion curve (reconstruction fidelity vs. compression) across latent dimension `d`, latent-graph sparsity `k`, and — as of the 2026-07-09 reframing — latent node count `M`. This is the direct prerequisite for the planned QGNN integration (mission.md, `reports/midterm_report.md` §6) — the QGNN consumes (z̃, A_z), so its qubit/gate budget is set by how small we can make that representation while still reconstructing the graph faithfully.

Graph compression is the **priority** of this phase. Node classification is included as a lower-priority secondary task. Graph-level tasks (MUTAG/PROTEINS/IMDB-B) are deferred out of Phase 3 entirely (see Scope). Link prediction across all splits was already completed as an extension of Phase 2 (`README.md`, `reports/midterm_report.md`) and is not repeated here.

**Reframing (2026-07-09):** T3.1–T3.3 (below) swept `d` and `k` but always at `M=N` — the latent graph had as many nodes as the input graph, so "compression" only ever meant smaller per-node dimensionality and sparser edges at a fixed node count. That's no longer the goal: the latent graph of distributions should be smaller than the input graph in **node count** too. This reverses `mission.md`'s prior "not a hierarchical pooling model" decision (see mission.md changelog). T3.4 (a decoder tweak that would have stayed at `M=N`) is superseded by **T3.6**, a learned pooling mechanism. T3.1–T3.3's results remain valid and are kept as the `M=N` baseline that T3.6 compares against.

---

## Scope

### In scope
- Compression metrics: dimensionality ratio (d/F), edge-count ratio (|A_z|/|E|), **node-count ratio (M/N, new T3.6)**, reconstruction fidelity (F1, bits-per-edge) on the **full input graph** (no held-out split — see design decision below)
- A full-graph training/eval mode (`train_ratio=1.0`) distinct from the link-prediction split used in Phases 0–2
- A rate-distortion sweep over `(latent_dim, k)`, holding each dataset's Phase 2 NAS-best `graph_method`/`prior`/`mp_rounds`/`lr`/`beta`/`lambda_` fixed
- A **compression-oriented** `k` sweep, separate from the AUC-optimal `k` found by Phase 2 NAS (which can produce a *denser* A_z than the input graph — see Design Decisions)
- **Learned node-count pooling (T3.6, new priority):** a DiffPool-style soft assignment that reduces the latent graph to `M ≪ N` nodes, with unpooling at decode time, and a sweep over `M/N` holding `(d, k)` fixed at each dataset's T3.3 compression-optimal point
- Node classification: linear probe + 2-layer MLP head on frozen z̃ (secondary priority)
- **A_z-conditioned decoder (T3.4, revived 2026-07-13 — see Design Decision 8):** an optional decode path (`experiment.decoder=graph_conditioned`) that gives `A_z` a guaranteed path into the reconstruction, for a head-to-head comparison against the plain inner-product decoder now that T3.6 alone hasn't closed the fidelity gap

### Out of scope
- Graph-level tasks (MUTAG/PROTEINS/IMDB-B pooling + classification) — deferred to Phase 4 or later; not connected to the compression/QGNN priority and requires new dataset preprocessing. Note: that pooling is a *different* mechanism (whole-graph embedding for graph classification) from T3.6's node-count pooling (which compresses a single graph's own latent representation) — the two are unrelated despite both being called "pooling."
- Re-running link prediction across splits — already done via the Phase 2 extension (`configs/best/*.yaml` retrained on 20/40/80% splits, results in `README.md`)
- Any quantum circuit implementation (QGNN itself) — GVLS remains a classical preprocessing stage; the quantum consumer is future work per the midterm report's roadmap
- Ablations (graph method, prior, mp_rounds, β, hard vs. soft assignment) — Phase 4

---

## Design Decisions (resolved before writing this spec)

1. **Compression metric.** Report **separate ratio metrics**, not a single collapsed score: `d/F` (embedding-dimension ratio), `|A_z|/|E|` (edge-count ratio), `M/N` (node-count ratio, new T3.6), plus reconstruction fidelity (F1, bits-per-edge) at each point. This keeps the comparison transparent and avoids an arbitrary weighting between "smaller" and "faithful."
2. **`k` strategy.** Phase 2's NAS chose `k` (and `d`) to maximize link-prediction AUC — e.g. Cora's best config uses `k=20`, which after union-symmetrization can produce **more** edges in A_z than the input graph has (`|A_z|` can reach up to `~2·k·N/2`, vs. Cora's `|E|=5429` at `N=2708`). An AUC-optimal config is not necessarily a compression-optimal one. Phase 3 runs a **separate `k` sweep** (`k ∈ {1,2,3,5,10,20}`) specifically to trace the rate-distortion curve, independent of the Phase 2 NAS objective. The two configs (`configs/best/{dataset}.yaml` for link prediction, `configs/compression/{dataset}.yaml` for compression) are expected to differ and both are kept.
3. **Fidelity protocol.** Reconstruction fidelity is measured on the **full input graph**, with no train/val/test held-out split (`train_ratio=1.0`). This matches how compression is normally judged — how well can you decompress exactly what you encoded — rather than generalization to unseen edges, which is a link-prediction question already answered in Phase 2. Because there is no held-out signal, "best epoch" selection uses a fixed epoch budget (matching Phase 1's 200-epoch default) and the final-epoch checkpoint; overfitting to the training graph is expected and desired for a compression objective.
4. **Decoder — superseded 2026-07-09, revived 2026-07-13 (see Design Decision 8).** The original plan was to start with the existing inner-product decoder `σ(z̃_i · z̃_j)` unchanged, and only build an A_z-conditioned alternative if the sweep showed the decoder was the bottleneck (T3.4's trigger). It fired for all three datasets (see `specs/phase3/validation.md` V-4), but rather than build that decoder tweak — which would have kept `M=N` — the project pivoted to node-count pooling instead (T3.6, Design Decision 5). The inner-product decoder itself was used unchanged throughout T3.6; what changed there is that it operates on `M` pooled nodes and is unpooled via `S` afterward, not that its functional form changes. T3.6 alone didn't close the fidelity gap, which is the condition under which the plan said to revisit this — see Design Decision 8.
5. **Node-count reduction (new, T3.6).** Introduce a learned soft assignment `S ∈ [0,1]^{N×M}` (DiffPool-style, row-softmax over a linear scoring function on `z`) that pools the `N` per-node Gaussians into `M ≪ N` cluster-level Gaussians via moment matching. The latent graph `A_z` and message passing operate on these `M` pooled nodes (reusing Phase 1's `LatentGraphLearner` unchanged, just at smaller scale). Reconstruction unpools via the same `S`, at the logit level: `Â = S·(z̃_pooled z̃_pooledᵀ)·Sᵀ`. `M` is swept as a ratio of `N` holding `(d, k)` fixed at each dataset's T3.3 compression-optimal config, isolating the node-count axis. `S` itself must be counted as part of the compressed representation's storage cost (hardened to one argmax index per node for accounting, per FR-7), rather than treated as free — otherwise the "compression" claim would be misleading.
6. **Auxiliary losses on S — added 2026-07-09 after an empirical cold-start collapse, not part of the original design.** The first pass at T3.6 (Design Decision 5 above, unmodified) converged to a degenerate always-predict-edge classifier on *every* grid point, across all three datasets: `S` starts near-uniform at initialization (a plain linear layer over `M` classes gives near-uniform softmax before training), which collapses the pooled `M×M` similarity matrix to an almost-constant value; because `S`'s rows sum to 1, unpooling a near-constant matrix produces an *exactly* constant `N×N` reconstruction, which in turn gives `S` no gradient to escape the collapse — a self-reinforcing dead end. This is a known failure mode of naive soft-assignment pooling (it is why DiffPool, Ying et al. 2018, ships with exactly the two auxiliary losses added here: `assignment_entropy(S)`, to encourage each node's assignment to specialize, and `assignment_link_loss(S, A) = BCE(S·Sᵀ, A)` excluding the diagonal, to give `S` a direct gradient signal from the real input graph that doesn't depend on anything downstream of it). Validated on Cora at `pool_ratio=0.5`: F1 rose from the degenerate 0.667 to ≈0.78 with `entropy_weight=0.1, aux_link_weight=5.0` (chosen via a small manual sweep, past the point of diminishing returns — see `specs/phase3/validation.md` V-7 for the full sweep). In hindsight this risk should have been anticipated at spec-writing time rather than discovered empirically; noted here so the eventual paper's method section doesn't present the auxiliary losses as an incidental implementation detail — they are load-bearing.
7. **ELBO KL-normalization bug — found and fixed 2026-07-11, a second and independent cause of residual collapse at large `M`.** After Design Decision 6's fix, PubMed's two largest pool sizes (`M=9858`, `M=4929`) still collapsed to the same F1=0.6667 signature. Root cause: `kl_isotropic` and `kl_graph_mrf` (`src/gvls/losses/elbo.py`) returned a raw, un-normalized `torch.sum()` over the node count they were computed over, while `elbo()`'s reconstruction term is mean-reduced over all `N²` pairs — so `β·KL`'s absolute magnitude scaled linearly with `M` (or `N`, pre-pooling) while `recon_loss`'s did not. This was invisible at a fixed `N` (Phase 0–2, and T3.3's `M=N` baseline), but T3.6's `pool_ratio` sweep varies `M` across nearly two orders of magnitude, exposing it — most severely for PubMed, whose NAS-best config uses `prior=graph_mrf` (independently ~18–20× more KL magnitude than `isotropic` at matched inputs) and the largest `β` of the three datasets. At PubMed's two largest `M`, `β·KL` reached 12–24× the reconstruction loss's scale, pulling `μ` toward the prior and collapsing `z̃_pooled` to a near-constant matrix — the same symptom as Design Decision 6's failure mode, but a different, compounding cause, independent of `S`'s behavior. Also gives a plausible (not fully isolated) explanation for T3.3's previously-unexplained PubMed "capacity hurts" pattern (F1 falling monotonically with `d`; `d` scales `kl_graph_mrf`'s magnitude the same way, just over a much smaller range). Fixed by normalizing both KL functions by node count. See `specs/phase3/validation.md` V-8 for the full evidence chain (analytical magnitude check, causal beta ablation, and the retrospective fit to the T3.3 anomaly) and status (fixed and unit-tested; production reruns pending).
8. **T3.4 revived, and `k=1` removed from the sweep grid — both 2026-07-13, after the confirmed reruns showed the fidelity gap persists and surfaced a new structural concern.** Two things prompted this: (a) with the ELBO bug fixed, all three datasets' compression grids are flat plateaus below the 0.90 floor (Cora ≤0.828, CiteSeer ≤0.819, PubMed ≤0.777) — T3.6's pooling did not close the gap on its own, which is exactly the condition the original T3.4 plan set for revisiting the decoder (see Design Decision 4, T3.4 below); (b) PubMed's post-fix compression-optimal config landed on `k=1`, and inspecting what `k=1` actually produces — `LatentGraphLearner` keeps only each node's single highest-scoring neighbour, symmetrized — showed `A_z` edge counts of almost exactly `M−1` at every pool size tested (a near-spanning-tree, not a meaningfully relational graph), which won `select_compression_optimal`'s F1-argmax fallback only because F1 is now statistically flat across the whole `k` range post-fix, not because `k=1` is architecturally better. Response: (1) revive T3.4 (`LatentGraphDecoder`, `src/gvls/models/decoder.py`) rather than fix `select_compression_optimal`'s fallback tie-breaking, since the more direct question — is the plain inner-product decoder the actual bottleneck — is what the original plan wanted answered first; (2) remove `k=1` from `configs/experiment/compression_sweep.yaml`'s grid so future sweeps can't select it. Both are process/scope decisions, not further bug fixes.

---

## File Map

```
src/gvls/
  eval/
    compression.py         # T3.1 — reconstruction_f1, dim/edge/node compression ratios
  data/
    splits.py               # T3.2 — extend with full_graph_split (train_ratio=1.0)
  compression/
    sweep.py                 # T3.3 — train_gvls_full_graph, evaluate_compression,
                              #        write_results_csv, select_compression_optimal
                              #        (added; not in the original file map — see
                              #        T3.3's "Implementation note")
    pooling_sweep.py          # T3.6 (new) — sweep over pool_ratio, reusing sweep.py's
                              #              train/eval helpers with pooling enabled
  models/
    pooling.py                # T3.6 (new) — LatentGraphPooling: assignment S, moment-
                              #              matched Gaussian pooling, unpool-via-S decode
    decoder.py                 # T3.4 (revived 2026-07-13) — LatentGraphDecoder:
                              #              one extra MP round over A_z before decode
configs/
  train/
    full_graph.yaml          # T3.2 — train_ratio=1.0, fixed epoch budget, no val/test
  compression/
    cora.yaml                # T3.3 — compression-optimal config (written by sweep)
    citeseer.yaml             # T3.3
    pubmed.yaml               # T3.3
  experiment/
    compression_sweep.yaml   # T3.3 — grid definition (d values, k values)
    pooling_sweep.yaml        # T3.6 (new) — pool_ratio grid definition
  compression_sweep_config.yaml  # T3.3 — root Hydra config (data/train/experiment defaults)
experiments/
  compression_sweep.py       # T3.3 — thin Hydra CLI wrapper around gvls.compression.sweep
  pooling_sweep.py            # T3.6 (new) — thin Hydra CLI wrapper around pooling_sweep.py
  node_probe.py               # T3.5 — linear/MLP probe on frozen z̃
tests/
  test_compression_metrics.py  # T3.1 — extended with node_compression_ratio,
                              #        assignment_storage_bits tests (T3.6)
  test_full_graph_split.py     # T3.2
  test_compression_sweep.py    # T3.3 — smoke test, tiny grid
  test_pooling.py                # T3.6 (new) — assignment/pooling/unpooling shape+gradient tests
  test_node_probe.py            # T3.5
  test_decoder.py                # T3.4 (revived 2026-07-13) — LatentGraphDecoder shape/
                              #              gradient/A_z-sensitivity tests
```

**T3.4 status:** `src/gvls/models/decoder.py` and `tests/test_decoder.py` were marked "removed, no longer planned" (2026-07-09, Design Decision 4) when the project pivoted to T3.6's pooling instead. **Revived 2026-07-13** — see Design Decision 8 and the T3.4 section below.

---

## Tasks

### T3.1 — Compression metrics

**File:** `src/gvls/eval/compression.py`

- `reconstruction_f1(adj_true, adj_logits, threshold=0.5) -> float`: binarize `sigmoid(adj_logits) > threshold`, compute F1 against the true adjacency over the upper triangle (`i<j`)
- `dim_compression_ratio(d, F) -> float`: returns `d / F`
- `edge_compression_ratio(A_z, num_input_edges) -> float`: counts non-zero upper-triangle entries in A_z (post top-k + symmetrization), returns that count divided by `num_input_edges`
- Reuse `bits_per_edge` from `src/gvls/eval/metrics.py` (Phase 0), applied over all node pairs (or a large random sample of pairs for datasets where N² is too large to materialize densely, e.g. PubMed with N=19717 → N²≈389M pairs)

Tests (`tests/test_compression_metrics.py`):
- `reconstruction_f1` = 1.0 on a perfect predictor
- `reconstruction_f1` = 0.0 on an all-zero predictor with real edges present
- `dim_compression_ratio(32, 1433)` ≈ 0.0223
- `edge_compression_ratio` counts correctly on a toy 5-node A_z with known non-zero pattern
- `bits_per_edge` sampling path (PubMed-scale) returns a finite value close to the exact full-pair computation on a small graph where both are tractable

---

### T3.2 — Full-graph split mode

**File:** `src/gvls/data/splits.py` (extend)

- `full_graph_split(data, seed) -> EdgeSplit`: `train_edge_index` = all real edges (both directions, for message passing); no val/test edges. Negative sampling for fidelity evaluation happens at eval time, not at split time (evaluation samples a fresh 1:1 negative set each call, consistent with `auc_ap`/`bits_per_edge` conventions).
- **File:** `configs/train/full_graph.yaml` — `epochs: 200` (matches Phase 1 default), no `val_check` / early stopping (no held-out signal to check against)

Tests (`tests/test_full_graph_split.py`):
- `train_edge_index` contains every real edge (both directions) and nothing else
- No val/test edges produced
- Determinism: same seed → identical split

---

### T3.3 — Rate-distortion sweep

**File:** `configs/experiment/compression_sweep.yaml`
```yaml
latent_dim: [4, 8, 16, 32, 64, 128]
k: [1, 2, 3, 5, 10, 20]
# graph_method, prior, mp_rounds, lr, beta, lambda_ inherited from configs/best/{dataset}.yaml
```

**File:** `experiments/compression_sweep.py`

Hydra entry point:
1. Load dataset, build `full_graph_split`
2. Load the dataset's Phase 2 NAS-best config (`configs/best/{dataset}.yaml`) for the fixed hyperparameters (`graph_method`, `prior`, `mp_rounds`, `lr`, `beta`, `lambda_`)
3. For each `(latent_dim, k)` in the grid: build GVLS, train for `configs/train/full_graph.yaml` epochs, then compute `reconstruction_f1`, `bits_per_edge`, `dim_compression_ratio`, `edge_compression_ratio`, and latent graph density on the trained model
4. Write one row per grid point to `results/compression/{dataset}.csv`
5. Log each run to W&B (group tag `compression-sweep-{dataset}`)
6. Select the **compression-optimal config**: smallest `(d, k)` pair meeting a fidelity floor (F1 ≥ 0.90 — revisit once real numbers are in), write to `configs/compression/{dataset}.yaml`

**Execution order:** Cora first (fastest, validates the pipeline), then PubMed (largest graph — the clearest demonstration of compression value for the QGNN motivation, per the midterm report), then CiteSeer.

**Compute risk flagged before running the full sweep:** PubMed's Phase 2 NAS-best config uses `prior=graph_mrf`, whose KL term calls `torch.linalg.slogdet` on an N×N matrix (`kl_graph_mrf` in `src/gvls/losses/elbo.py`) — O(N³). Measured on this machine: ~6.2s per call at PubMed's N=19717. At 200 epochs × 36 grid points, that's ~12.4 hours for PubMed alone if run serially on CPU. Cora (N=2708) and CiteSeer (N=3327) are unaffected (`prior=isotropic` for both, and even if graph_mrf were used the O(N³) cost is negligible at that scale). Before launching the full production sweep, decide whether to (a) accept the ~12-hour PubMed run, (b) reduce PubMed's epoch budget per NFR-2's allowance, or (c) run on GPU if available.

Tests (`tests/test_compression_sweep.py`):
- Smoke test: 2×2 grid (`latent_dim=[8,16]`, `k=[2,5]`) on Cora, 10 epochs, completes without error and writes 4 rows to the results CSV

**Implementation note:** grid-point training/evaluation logic lives in `src/gvls/compression/sweep.py` (not inline in `experiments/compression_sweep.py`), mirroring the `src/gvls/nas/objective.py` split used in Phase 2 — this keeps the core logic unit-testable against a tiny synthetic graph without going through Hydra config resolution or downloading a real dataset. `experiments/compression_sweep.py` is a thin Hydra CLI wrapper around it. This deviates from the file map above, which didn't list a `src/gvls/compression/` module.

**Cora run complete (2026-07-07):** all 36 grid points ran successfully. Full results in `results/compression/cora.csv`, summarized in `README.md` and `specs/phase3/validation.md` V-3. Headline: F1 is flat (0.813–0.828) across the *entire* grid regardless of `d` or `k` — the fidelity floor (0.90) is not met anywhere, which fires T3.4's trigger below. `k` (not `d`) is what actually controls edge-count compression: `k=1` yields `|A_z|` at ~37% of the input's edge count, while `k≥2` makes A_z denser than the input graph.

**PubMed run complete (2026-07-08), run on a remote A100:** all 36 grid points ran successfully — the O(N³) `graph_mrf` KL cost was manageable on GPU as anticipated. Full results in `results/compression/pubmed.csv`, summarized in `README.md` and `specs/phase3/validation.md` V-3. Headline, and a materially different pattern from Cora: **F1 actively decreases with more capacity** (mean F1 by `k`: 0.745→0.716 from `k=1`→`k=20`; at fixed `k=20`, F1 by `d`: 0.742→0.673 from `d=4`→`d=128`). PubMed's own Phase 2 NAS-best architecture (`d=128, k=20`, tuned for link-prediction AUC) lands on the single worst point in the compression grid. `bits_per_edge` is also degenerate (≈1.0 exactly everywhere), flagged as an open question rather than root-caused — see README for the `pos_weight`-based hypothesis.

**Superseded 2026-07-13 — rerun after the ELBO KL-normalization fix (see Design Decision 7, V-8).** This "capacity hurts" pattern and the degenerate `bits_per_edge` were both symptoms of the KL bug, not real effects. The rerun (`results/compression/pubmed.csv`, commit `eec133b`) shows F1 rising slightly with `d` (0.7725→0.7771) instead of falling, `k` now having ~zero effect (mean F1 flat at 0.7734 across all six `k`, the same "`A_z` too weak to matter" conclusion as CiteSeer), and `bits_per_edge` a real, non-degenerate ~1.13–1.14. `configs/compression/pubmed.yaml` changed to `d=128, k=1` (F1=0.7772) as a result — full before/after in `specs/phase3/validation.md` V-3 and V-8. The original findings above are kept as the historical record of the bug's symptom. **Follow-up, not yet done:** T3.6's PubMed pooling sweep was confirmed working against the *old* `d=16,k=2` config and should be rerun once more against this new `d=128,k=1` point.

**CiteSeer run complete (2026-07-09):** all 36 grid points ran successfully (after a transient GitHub rate-limit issue downloading the raw Planetoid files — resolved by fetching the last file from a jsdelivr mirror; unrelated to the sweep code). Full results in `results/compression/citeseer.csv`. Headline: **`k` has essentially zero effect on F1, for a mechanistic reason** — CiteSeer's NAS-best config is `mp_rounds=0, prior=isotropic`, so `A_z` never reaches `z̃` (mp_rounds=0 skips message passing) or the KL term (isotropic ignores `A_z`) — it has no path into the loss or the output at all. F1 is bit-identical across all 6 `k` values at `d∈{4,8,128}` (tiny ~1e-4 residuals at `d∈{16,32,64}`, likely floating-point non-determinism, not a real `k` effect). F1 ceiling (0.819) and range are close to Cora's. **All three datasets are now done, and none meet the 0.90 fidelity floor anywhere in their grids** — see the cross-dataset synthesis in `specs/phase3/validation.md` V-3.

---

### T3.4 — A_z-conditioned decoder — **superseded 2026-07-09, revived 2026-07-13**

**Trigger condition:** build this only if T3.3's results show the inner-product decoder is the bottleneck — concretely, if reconstruction F1 at the **largest** tested capacity (`d=128, k=20`) is **below 0.90**. That signals the ceiling itself is weak (a decoder/architecture limitation), not that we're pushing the compression ratio too far (which would show up as a normal fidelity drop-off at *small* `d`/`k`, not a flat low curve even at large capacity).

**Status (2026-07-09): triggered for all three datasets (Cora 2026-07-07, PubMed 2026-07-08, CiteSeer 2026-07-09), but not built — superseded instead.** Cora: F1 at `d=128, k=20` was 0.8235, flat-and-low across the whole grid. PubMed: F1 at `d=128, k=20` was 0.673, the worst point in the grid, part of a monotonic decline. CiteSeer: F1 at `d=128, k=20` was 0.8140, and — more fundamentally — `A_z` is provably inert for CiteSeer's NAS-best config (`mp_rounds=0, prior=isotropic` gives it no path into the loss). Three for three fired, but the project's compression goal was reframed on 2026-07-09 to prioritize node-count reduction (T3.6) over refining the decoder at a fixed `M=N`. This task was kept here as a record; the plan said to revisit only if T3.6 alone doesn't close the fidelity gap.

**PubMed's 2026-07-09 number above is now stale (see V-8/T3.3's 2026-07-13 rerun note above)** — after the ELBO KL fix, F1 at `d=128, k=20` is 0.7768, no longer the worst point in the grid but essentially tied for the best. The trigger still fires (0.7768 < 0.90), but for the same reason as Cora and CiteSeer now (a flat plateau), not because more capacity was actively hurting.

**Revived 2026-07-13 — see Design Decision 8.** T3.6's pooling alone did not close the fidelity gap: even after the ELBO fix, all three datasets sit as flat plateaus below 0.90 (Cora ≤0.828, CiteSeer ≤0.819, PubMed ≤0.777), and PubMed's pooling sweep additionally re-collapsed at its smallest `M` under the new compression-optimal config (`d=128,k=1` — see V-8). That's precisely the condition the original plan set for revisiting this task, so it's revived rather than superseded.

**File:** `src/gvls/models/decoder.py` — implemented. `LatentGraphDecoder(latent_dim)`: applies one unconditional extra round of `LatentMessagePassing` to `z_tilde` over `A_z`, then decodes via the inner product on the result (`z_decode = z_tilde + D⁻¹A_z z_tilde W`, `recon_logits = z_decode @ z_decodeᵀ`). Unlike the base model's own `mp_rounds` (which is 0 for CiteSeer's and PubMed's Phase 2 NAS-best configs — the reason `A_z` was inert/near-inert for those two datasets in T3.3's original findings), this round always runs, so `A_z` is guaranteed a path into the reconstruction regardless of the encoder config.

**Integration:** `train_gvls_full_graph`/`evaluate_compression` (`src/gvls/compression/sweep.py`) take a new `decoder: "inner_product" | "graph_conditioned"` parameter, default `"inner_product"` (byte-for-byte unchanged behavior and output paths). `experiments/compression_sweep.py` exposes this as `experiment.decoder`; a `graph_conditioned` run writes to `results/compression/{name}_graph_decoder.csv` / `configs/compression/{name}_graph_decoder.yaml` instead of the default paths, so a head-to-head comparison run doesn't overwrite the `inner_product` baseline that T3.6 and other consumers depend on.

**Status: implemented and unit-tested** (`tests/test_decoder.py` — shape, gradient flow to both `z_tilde` and the decoder's own weight, sensitivity to `A_z`, and a zero-`A_z` sanity check that it reduces to the plain inner product; `tests/test_compression_sweep.py` — end-to-end smoke test through the sweep's train/eval path, plus an invalid-`decoder`-string test). Manually smoke-tested end-to-end through the actual `experiments/compression_sweep.py` CLI (3 epochs, Cora, `d=8,k=2`) to confirm the suffixed output paths work and the baseline files are untouched. **The production head-to-head sweep (all three datasets, both decoders, full grid) has not been run** — see `specs/phase3/validation.md` V-4.

---

### T3.6 — Node-count compression via learned pooling (new, supersedes T3.4)

**Motivation:** T3.1–T3.3 measured fidelity vs. `(d, k)` entirely at `M=N` — the latent graph always had as many nodes as the input. The reframed goal is a latent graph of distributions that is smaller than the input graph in node count too, not just per-node dimensionality. This reverses `mission.md`'s prior "not a hierarchical pooling model" stance (see mission.md changelog, 2026-07-09).

**File:** `src/gvls/models/pooling.py`

- `LatentGraphPooling(input_dim, M)`: a learned linear scoring layer on `z` produces per-node logits over `M` clusters; row-softmax gives the assignment `S ∈ [0,1]^{N×M}`
- Pools Gaussians via moment matching, not a naive weighted mean:
  - `μ_pooled = Sᵀμ`
  - `σ²_pooled = Sᵀ(σ²) + Sᵀ(μ²) − μ_pooled²` (within-cluster variance + between-node variance of the means)
- Reuses Phase 1's `LatentGraphLearner` unchanged, applied to the `M` pooled embeddings (`A_z` is now `M×M`)
- Reuses Phase 1's residual message passing unchanged, operating over the `M`-node `A_z`
- Unpool at decode: `Â (N×N) = S · (z̃_pooled z̃_pooledᵀ) · Sᵀ`, using the *same* `S` from pooling (not re-derived), computed at the logit level (no sigmoid before unpooling — see Design Decision 5)
- `assignment_entropy(S)` and `assignment_link_loss(S, A)` (both in `pooling.py`): two auxiliary losses required to avoid a cold-start collapse — see Design Decision 6 and the "Cold-start collapse" subsection below

**Cold-start collapse, diagnosed and fixed 2026-07-09 (see Design Decision 6):** the first version of T3.6, with no auxiliary losses, converged to a degenerate always-predict-edge classifier on every grid point of every dataset (F1 stuck at exactly 2/3, the value produced by a trivial constant-positive predictor on the 1:1 balanced eval set) — not a real rate-distortion result. Root cause: `S` starts near-uniform, which collapses the pooled `M×M` similarity to an almost-constant value, which — because `S`'s rows sum to 1 — makes the unpooled reconstruction *exactly* constant, which then gives `S` no gradient to specialize. Fixed by adding `entropy_weight · assignment_entropy(S) + aux_link_weight · assignment_link_loss(S, A)` to the training loss (defaults `entropy_weight=0.1, aux_link_weight=5.0`, selected via a manual sweep on Cora at `pool_ratio=0.5`: F1 rose from 0.667 to ≈0.78, plateauing for `aux_link_weight ≥ 5`). The three `*_pooling.csv` result files produced before this fix are stale and must be regenerated — see `specs/phase3/validation.md` V-7.

**Second, independent bug found 2026-07-11 — see Design Decision 7 and `specs/phase3/validation.md` V-8.** After the fix above, the regenerated `pubmed_pooling.csv` still showed the exact pre-fix collapse signature (F1=0.6667) at `pool_ratio=0.5`/`0.25` (`M=9858`/`4929`), while `pool_ratio=0.125`/`0.0625` escaped it. Root-caused to `kl_isotropic`/`kl_graph_mrf` (`src/gvls/losses/elbo.py`) being un-normalized sums over node count, so `β·KL`'s magnitude scaled linearly with `M` while the mean-reduced `recon_loss` did not — for PubMed's `graph_mrf` prior and comparatively large `β`, this pushed `β·KL` to 12–24× the reconstruction loss's scale at the two largest `M`, pulling `μ` toward the prior and collapsing the pooled reconstruction independent of whatever `S` was doing. Fixed by normalizing both KL functions by node count (implemented and unit-tested). **Confirmed 2026-07-12**: the rerun `pubmed_pooling.csv` shows both previously-collapsed grid points (`M=9858`, `M=4929`) fixed (F1 0.6667→0.748/0.750), with Cora/CiteSeer moving by <1% (negative control) and PubMed's whole pooling grid now within 0.008–0.016 F1 of its `M=N` baseline — see validation.md V-8 for the full before/after table. T3.3's PubMed `(d,k)` baseline (`results/compression/pubmed.csv`) has not yet been rerun with the fix; that's the direct test of whether this bug also explains T3.3's "capacity hurts" anomaly (§V-3).

**File:** `src/gvls/compression/pooling_sweep.py`

Hydra entry point, mirroring `sweep.py`'s structure:
1. Load dataset, build `full_graph_split` (T3.2, unchanged)
2. Load each dataset's T3.3 compression-optimal config (`configs/compression/{dataset}.yaml`) for fixed `(d, k)` and other hyperparameters
3. For each `pool_ratio ∈ {0.5, 0.25, 0.125, 0.0625}`: compute `M = max(2, round(pool_ratio * N))`, build GVLS with pooling enabled, train for the full-graph epoch budget with the auxiliary losses above, compute `reconstruction_f1` (on the unpooled `Â`), `node_compression_ratio`, `assignment_storage_bits`, plus all of T3.3's existing metrics
4. Write one row per grid point to `results/compression/{dataset}_pooling.csv`
5. Log each run to W&B (group tag `compression-pooling-sweep-{dataset}`)

**Execution order:** Cora first (validates the pipeline, matching T3.3's precedent), then CiteSeer, then PubMed (largest graph, most informative for the QGNN motivation, most compute cost — save for last once the mechanism is validated).

Tests (`tests/test_pooling.py`):
- `S` rows sum to 1 (valid softmax) for a toy `(N=10, M=3)` case
- Pooled `(μ, log_var)` have shape `(M, d)`
- Gradients reach both the assignment scoring layer and the pooled Gaussian parameters
- Unpooled `Â` has shape `(N, N)` regardless of `M`
- `assignment_entropy` is high (≈`log M`) for a uniform `S` and near-zero for a one-hot `S`; gradient reaches the pre-softmax logits
- `assignment_link_loss` is low when `S`'s implied clustering matches the true adjacency and high when it contradicts it; gradient reaches the pre-softmax logits; a `pos_weight` argument upweights positives
- Smoke test: tiny grid (`pool_ratio=[0.5, 0.25]`) on Cora, 10 epochs, completes without error and writes rows to the results CSV

---

### T3.5 — Node classification probe (secondary priority)

**File:** `experiments/node_probe.py`

- Load each dataset's Phase 2 NAS-best GVLS (link-prediction config, `configs/best/{dataset}.yaml`), freeze it, extract z̃
- Train (a) a linear probe and (b) a 2-layer MLP head on z̃ → class logits
- Semi-supervised setting: standard Planetoid public split (20 labels/class)
- Report accuracy and macro-F1

Tests (`tests/test_node_probe.py`):
- Probe trains without error on a synthetic z̃ + labels
- Accuracy = 1.0 on a trivially separable synthetic case

**Priority note:** this task may slip to Phase 4 if compression work (T3.1–T3.3, T3.6) runs long — graph compression is the phase's priority deliverable.

---

## Deliverable

- `results/compression/{cora,citeseer,pubmed}.csv` populated with the full `(d, k)` rate-distortion grid (T3.1–T3.3, `M=N` baseline)
- `results/compression/{cora,citeseer,pubmed}_pooling.csv` populated with the `pool_ratio` (`M/N`) sweep (T3.6)
- `configs/compression/{cora,citeseer,pubmed}.yaml` written (compression-optimal configs, `M=N`)
- A clear comparison, per dataset, of: input graph size (N, F, |E|) vs. compressed representation size (M, d, |A_z|) at the compression-optimal `(d, k, M)` point, with `d/F`, `|A_z|/|E|`, `M/N`, F1, and bits-per-edge reported
- T3.4 explicitly recorded as superseded, with the F1 numbers that had triggered it, and the reasoning for pivoting to T3.6 instead
- T3.6 built and evaluated on all three datasets, reporting how fidelity trades off against node-count reduction independent of the `(d, k)` axes T3.3 already covers
- Node classification results (T3.5) for at least Cora, if time allows within the phase
