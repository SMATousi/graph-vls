# Phase 3 — Validation

## Exit Criteria

---

## V-1: Compression Metrics ✅

| Check | Pass condition | Result |
|---|---|---|
| `reconstruction_f1` perfect predictor | Returns 1.0 | ✅ `test_reconstruction_f1_perfect` |
| `reconstruction_f1` all-zero predictor | Returns 0.0 with real edges present | ✅ `test_reconstruction_f1_all_zero_predictor` |
| `dim_compression_ratio` correctness | `dim_compression_ratio(32, 1433)` ≈ 0.0223 | ✅ `test_dim_compression_ratio_value` |
| `edge_compression_ratio` correctness | Matches hand-counted non-zero pattern on a toy 5-node A_z | ✅ `test_edge_compression_ratio_toy_graph` |
| `bits_per_edge` sampled ≈ exact | Sampled estimator agrees with exact full-pair computation (±0.01 bits/edge) on a small graph | ✅ `test_bits_per_edge_sampled_matches_exact_on_small_graph` (exact match when sample covers all pairs) |

19 tests in `tests/test_compression_metrics.py`, all passing.

---

## V-2: Full-Graph Split ✅

| Check | Pass condition | Result |
|---|---|---|
| All real edges present | `train_edge_index` contains every real edge (both directions) | ✅ `test_train_edge_index_contains_every_real_edge_both_directions` |
| No val/test edges | `full_graph_split` produces no held-out edges | ✅ `test_no_val_test_edges` |
| Determinism | Same seed → identical split across runs | ✅ `test_determinism` |

5 tests in `tests/test_full_graph_split.py`, all passing.

---

## V-3: Rate-Distortion Sweep 🔶 In Progress (Cora ✅, CiteSeer ⬜, PubMed ⬜ running remotely)

| Check | Pass condition | Result |
|---|---|---|
| Cora grid complete | 36/36 `(d, k)` combinations trained and logged to `results/compression/cora.csv` | ✅ 36/36, 200 epochs each |
| PubMed grid complete | 36/36 combinations logged to `results/compression/pubmed.csv` | ⬜ running on a remote A100 (PubMed's NAS-best `prior=graph_mrf` KL is O(N³)/epoch — too slow on CPU, see `specs/phase3/plan.md`) |
| CiteSeer grid complete | 36/36 combinations logged to `results/compression/citeseer.csv` | ⬜ not started |
| Compression-optimal configs written | `configs/compression/{cora,citeseer,pubmed}.yaml` exist and are valid `gvls.yaml`-schema configs | ✅ Cora written (fallback pick, see below); CiteSeer/PubMed ⬜ |
| Fidelity floor met at some grid point | At least one `(d, k)` per dataset achieves `reconstruction_f1 ≥ 0.90` | ❌ **Cora: not met** — best F1 across all 36 points is 0.828 (d=16, k=20); CiteSeer/PubMed ⬜ pending |
| W&B logging | `compression-sweep-{dataset}` group contains 36 runs per dataset with all FR-3 metrics | ✅ Cora (36 runs logged); CiteSeer/PubMed ⬜ |

### Cora findings (full results: `results/compression/cora.csv`)

- **F1 is flat across the entire grid** (0.813–0.828, a 1.5-point spread over all 36 `(d,k)` combinations) — more latent capacity or a denser latent graph buys essentially nothing. This points at the plain inner-product decoder as the bottleneck rather than the compression ratio.
- **`k`, not `d`, controls edge compression.** At `k=1`, `|A_z|` is ~37% of `|E|` (genuine compression); every `k≥2` makes A_z denser than the input graph (up to 7.5× at the NAS-best `k=20`).
- Since no point met the 0.90 floor, `configs/compression/cora.yaml` was written via the fallback (highest raw F1: d=16, k=20) — see `select_compression_optimal` in `src/gvls/compression/sweep.py`. Note this fallback pick is one of the *least* compressed points in the grid (edge ratio 7.37×); `d=8, k=1` reaches nearly the same F1 (0.825) at a fraction of the size on both axes and is arguably the more useful operating point for the QGNN use case. Full write-up in `README.md`.

### Headline comparison

For each dataset, at the compression-optimal `(d, k)` (Cora: fallback pick, floor not met):

| Dataset | N | F | \|E\| | d | \|A_z\| | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 1433 | 5278 | 16 | 38891 | 0.0112 | 7.369 | 0.828 | 1.094 |
| CiteSeer | 3327 | 3703 | 4732 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| PubMed | 19717 | 500 | 44338 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

Note: Cora's `|E|=5278` here (vs. `5429` in Phase 0/1's link-prediction table) because `full_graph_split` removes self-loops the same way `split_edges` does, but the two counts were computed independently — both are correct for their respective splits.

---

## V-4: Decoder Fallback (conditional) 🔶 Triggered for Cora, not yet implemented

Triggered only if `reconstruction_f1` at `(d=128, k=20)` is below 0.90 for a given dataset.

| Check | Pass condition | Result |
|---|---|---|
| Trigger evaluated | Recorded per-dataset whether the trigger fired, with the F1 numbers | ✅ Cora: **triggered** (F1=0.8235 at d=128,k=20, < 0.90); CiteSeer/PubMed ⬜ pending their sweeps |
| If triggered: decoder implemented | `LatentGraphDecoder` shape/gradient tests pass | ⬜ not yet built — awaiting decision on whether to implement now or after CiteSeer/PubMed sweeps confirm the same pattern |
| If triggered: F1 comparison reported | Head-to-head F1 at matched `(d, k)`, baseline vs. A_z-conditioned decoder | ⬜ |

---

## V-5: Node Classification (secondary) ⬜

| Check | Pass condition | Result |
|---|---|---|
| Linear probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| MLP probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| CiteSeer / PubMed probes | Reported if time allows (secondary priority — may slip to Phase 4) | ⬜ |

---

## V-6: Code Quality ✅

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All tests pass (including new Phase 3 tests) | ✅ 115/115 |
| `ruff check src/` | Zero violations | ✅ |
| `test_compression_sweep.py` runtime | Completes in under 60 seconds (2×2 grid, 10 epochs) | ✅ full file (7 tests) runs in ~2s |
