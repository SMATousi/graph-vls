# Phase 3 — Plan

## Objective

Build and evaluate the **graph compression** pipeline: quantify how compact the GVLS latent representation (z̃, A_z) is relative to the input graph (X, A), and trace a rate-distortion curve (reconstruction fidelity vs. compression) across latent dimension `d` and latent-graph sparsity `k`. This is the direct prerequisite for the planned QGNN integration (mission.md, `reports/midterm_report.md` §6) — the QGNN consumes (z̃, A_z), so its qubit/gate budget is set by how small we can make that representation while still reconstructing the graph faithfully.

Graph compression is the **priority** of this phase. Node classification is included as a lower-priority secondary task. Graph-level tasks (MUTAG/PROTEINS/IMDB-B) are deferred out of Phase 3 entirely (see Scope). Link prediction across all splits was already completed as an extension of Phase 2 (`README.md`, `reports/midterm_report.md`) and is not repeated here.

---

## Scope

### In scope
- Compression metrics: dimensionality ratio (d/F), edge-count ratio (|A_z|/|E|), reconstruction fidelity (F1, bits-per-edge) on the **full input graph** (no held-out split — see design decision below)
- A full-graph training/eval mode (`train_ratio=1.0`) distinct from the link-prediction split used in Phases 0–2
- A rate-distortion sweep over `(latent_dim, k)`, holding each dataset's Phase 2 NAS-best `graph_method`/`prior`/`mp_rounds`/`lr`/`beta`/`lambda_` fixed
- A **compression-oriented** `k` sweep, separate from the AUC-optimal `k` found by Phase 2 NAS (which can produce a *denser* A_z than the input graph — see Design Decisions)
- Conditional stretch task: an explicit A_z-conditioned decoder, built only if the existing inner-product decoder proves to be the fidelity bottleneck
- Node classification: linear probe + 2-layer MLP head on frozen z̃ (secondary priority)

### Out of scope
- Graph-level tasks (MUTAG/PROTEINS/IMDB-B pooling + classification) — deferred to Phase 4 or later; not connected to the compression/QGNN priority and requires new dataset preprocessing
- Re-running link prediction across splits — already done via the Phase 2 extension (`configs/best/*.yaml` retrained on 20/40/80% splits, results in `README.md`)
- Any quantum circuit implementation (QGNN itself) — GVLS remains a classical preprocessing stage; the quantum consumer is future work per the midterm report's roadmap
- Ablations (graph method, prior, mp_rounds, β) — Phase 4

---

## Design Decisions (resolved before writing this spec)

1. **Compression metric.** Report **separate ratio metrics**, not a single collapsed score: `d/F` (embedding-dimension ratio), `|A_z|/|E|` (edge-count ratio), plus reconstruction fidelity (F1, bits-per-edge) at each point. This keeps the comparison transparent and avoids an arbitrary weighting between "smaller" and "faithful."
2. **`k` strategy.** Phase 2's NAS chose `k` (and `d`) to maximize link-prediction AUC — e.g. Cora's best config uses `k=20`, which after union-symmetrization can produce **more** edges in A_z than the input graph has (`|A_z|` can reach up to `~2·k·N/2`, vs. Cora's `|E|=5429` at `N=2708`). An AUC-optimal config is not necessarily a compression-optimal one. Phase 3 runs a **separate `k` sweep** (`k ∈ {1,2,3,5,10,20}`) specifically to trace the rate-distortion curve, independent of the Phase 2 NAS objective. The two configs (`configs/best/{dataset}.yaml` for link prediction, `configs/compression/{dataset}.yaml` for compression) are expected to differ and both are kept.
3. **Fidelity protocol.** Reconstruction fidelity is measured on the **full input graph**, with no train/val/test held-out split (`train_ratio=1.0`). This matches how compression is normally judged — how well can you decompress exactly what you encoded — rather than generalization to unseen edges, which is a link-prediction question already answered in Phase 2. Because there is no held-out signal, "best epoch" selection uses a fixed epoch budget (matching Phase 1's 200-epoch default) and the final-epoch checkpoint; overfitting to the training graph is expected and desired for a compression objective.
4. **Decoder.** Start with the existing inner-product decoder `σ(z̃_i · z̃_j)` unchanged (z̃ already reflects A_z's influence via Phase 1's latent message passing). Only build the alternative — an explicit decoder that conditions on A_z again at decode time — **if** the sweep results show the current decoder is the bottleneck (see T3.4's trigger condition). Both paths are documented here so the fallback doesn't require a new spec round-trip.

---

## File Map

```
src/gvls/
  eval/
    compression.py         # T3.1 — reconstruction_f1, dim/edge compression ratios
  data/
    splits.py               # T3.2 — extend with full_graph_split (train_ratio=1.0)
  compression/
    sweep.py                 # T3.3 — train_gvls_full_graph, evaluate_compression,
                              #        write_results_csv, select_compression_optimal
                              #        (added; not in the original file map — see
                              #        T3.3's "Implementation note")
  models/
    decoder.py               # T3.4 (conditional) — A_z-conditioned decoder
configs/
  train/
    full_graph.yaml          # T3.2 — train_ratio=1.0, fixed epoch budget, no val/test
  compression/
    cora.yaml                # T3.3 — compression-optimal config (written by sweep)
    citeseer.yaml             # T3.3
    pubmed.yaml               # T3.3
  experiment/
    compression_sweep.yaml   # T3.3 — grid definition (d values, k values)
  compression_sweep_config.yaml  # T3.3 — root Hydra config (data/train/experiment defaults)
experiments/
  compression_sweep.py       # T3.3 — thin Hydra CLI wrapper around gvls.compression.sweep
  node_probe.py               # T3.5 — linear/MLP probe on frozen z̃
tests/
  test_compression_metrics.py  # T3.1
  test_full_graph_split.py     # T3.2
  test_compression_sweep.py    # T3.3 — smoke test, tiny grid
  test_decoder.py               # T3.4 (conditional)
  test_node_probe.py            # T3.5
```

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

**CiteSeer run complete (2026-07-09):** all 36 grid points ran successfully (after a transient GitHub rate-limit issue downloading the raw Planetoid files — resolved by fetching the last file from a jsdelivr mirror; unrelated to the sweep code). Full results in `results/compression/citeseer.csv`. Headline: **`k` has essentially zero effect on F1, for a mechanistic reason** — CiteSeer's NAS-best config is `mp_rounds=0, prior=isotropic`, so `A_z` never reaches `z̃` (mp_rounds=0 skips message passing) or the KL term (isotropic ignores `A_z`) — it has no path into the loss or the output at all. F1 is bit-identical across all 6 `k` values at `d∈{4,8,128}` (tiny ~1e-4 residuals at `d∈{16,32,64}`, likely floating-point non-determinism, not a real `k` effect). F1 ceiling (0.819) and range are close to Cora's. **All three datasets are now done, and none meet the 0.90 fidelity floor anywhere in their grids** — see the cross-dataset synthesis in `specs/phase3/validation.md` V-3.

---

### T3.4 — A_z-conditioned decoder (conditional / stretch)

**Trigger condition:** build this only if T3.3's results show the inner-product decoder is the bottleneck — concretely, if reconstruction F1 at the **largest** tested capacity (`d=128, k=20`) is **below 0.90**. That signals the ceiling itself is weak (a decoder/architecture limitation), not that we're pushing the compression ratio too far (which would show up as a normal fidelity drop-off at *small* `d`/`k`, not a flat low curve even at large capacity).

**Status: triggered for all three datasets (Cora 2026-07-07, PubMed 2026-07-08, CiteSeer 2026-07-09).** Cora: F1 at `d=128, k=20` was 0.8235, flat-and-low across the whole grid. PubMed: F1 at `d=128, k=20` was 0.673, the worst point in the grid, part of a monotonic decline. CiteSeer: F1 at `d=128, k=20` was 0.8140, and — more fundamentally — `A_z` is provably inert for CiteSeer's NAS-best config (`mp_rounds=0, prior=isotropic` gives it no path into the loss). Three for three. Not yet implemented; this is now the clear next step.

**File:** `src/gvls/models/decoder.py` (only if triggered)

- `LatentGraphDecoder(latent_dim, mp_rounds=1)`: one extra round of message passing over A_z applied to z̃ before decoding — same form as T1.3's residual message passing (`z̃' = z̃ + D⁻¹ A_z z̃ W_dec`), separate learned weights `W_dec`
- Decode as `σ(z̃'_i · z̃'_j)`
- Compare F1 against the baseline inner-product decoder at matched `(d, k)` on Cora

Tests (`tests/test_decoder.py`, only if triggered):
- Output shape `(N, N)`
- Gradients reach `W_dec`
- F1 improvement over baseline decoder is reported (no fixed pass/fail — informational)

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

**Priority note:** this task may slip to Phase 4 if compression work (T3.1–T3.4) runs long — graph compression is the phase's priority deliverable.

---

## Deliverable

- `results/compression/{cora,citeseer,pubmed}.csv` populated with the full `(d, k)` rate-distortion grid
- `configs/compression/{cora,citeseer,pubmed}.yaml` written (compression-optimal configs)
- A clear comparison, per dataset, of: input graph size (N, F, |E|) vs. compressed representation size (N, d, |A_z|) at the compression-optimal point, with `d/F`, `|A_z|/|E|`, F1, and bits-per-edge reported
- T3.4 built and evaluated only if its trigger condition fires; otherwise explicitly recorded as "not triggered" with the F1 numbers that justified skipping it
- Node classification results (T3.5) for at least Cora, if time allows within the phase
