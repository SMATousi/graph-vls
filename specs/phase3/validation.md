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

## V-3: Rate-Distortion Sweep 🔶 In Progress (Cora ✅, PubMed ✅, CiteSeer ⬜)

| Check | Pass condition | Result |
|---|---|---|
| Cora grid complete | 36/36 `(d, k)` combinations trained and logged to `results/compression/cora.csv` | ✅ 36/36, 200 epochs each |
| PubMed grid complete | 36/36 combinations logged to `results/compression/pubmed.csv` | ✅ 36/36, run on a remote A100 (PubMed's NAS-best `prior=graph_mrf` KL is O(N³)/epoch — too slow on CPU, see `specs/phase3/plan.md`) |
| CiteSeer grid complete | 36/36 combinations logged to `results/compression/citeseer.csv` | ⬜ not started |
| Compression-optimal configs written | `configs/compression/{cora,citeseer,pubmed}.yaml` exist and are valid `gvls.yaml`-schema configs | ✅ Cora, PubMed written (both fallback picks, see below); CiteSeer ⬜ |
| Fidelity floor met at some grid point | At least one `(d, k)` per dataset achieves `reconstruction_f1 ≥ 0.90` | ❌ **Cora: not met** (best F1 0.828); ❌ **PubMed: not met** (best F1 0.761); CiteSeer ⬜ pending |
| W&B logging | `compression-sweep-{dataset}` group contains 36 runs per dataset with all FR-3 metrics | ✅ Cora, PubMed (36 runs logged each); CiteSeer ⬜ |

### Cora findings (full results: `results/compression/cora.csv`)

- **F1 is flat across the entire grid** (0.813–0.828, a 1.5-point spread over all 36 `(d,k)` combinations) — more latent capacity or a denser latent graph buys essentially nothing. This points at the plain inner-product decoder as the bottleneck rather than the compression ratio.
- **`k`, not `d`, controls edge compression.** At `k=1`, `|A_z|` is ~37% of `|E|` (genuine compression); every `k≥2` makes A_z denser than the input graph (up to 7.5× at the NAS-best `k=20`).
- Since no point met the 0.90 floor, `configs/compression/cora.yaml` was written via the fallback (highest raw F1: d=16, k=20) — see `select_compression_optimal` in `src/gvls/compression/sweep.py`. Note this fallback pick is one of the *least* compressed points in the grid (edge ratio 7.37×); `d=8, k=1` reaches nearly the same F1 (0.825) at a fraction of the size on both axes and is arguably the more useful operating point for the QGNN use case. Full write-up in `README.md`.

### PubMed findings (full results: `results/compression/pubmed.csv`)

- **Capacity hurts, not helps.** Mean F1 falls monotonically as `k` grows (0.745→0.716 from `k=1` to `k=20`) and, at fixed `k=20`, falls monotonically as `d` grows (0.742→0.673 from `d=4` to `d=128`). PubMed's own Phase 2 NAS-best architecture is `d=128, k=20` — i.e. the config tuned for link-prediction AUC lands on the *worst* point in this compression grid (F1=0.673).
- **`bits_per_edge` is degenerate — ≈1.0 exactly at every grid point** (1.0000017–1.0015761). Per Phase 0's convention this corresponds to logit≈0 (maximum uncertainty). Likely explanation: PubMed's pair space (~194M) forces the sampled-estimate path (`dense_pair_limit`), and the sample is >99.97% negative pairs; combined with the extreme `pos_weight` (~4384×, from `(N²−E)/E`) that PubMed's scale requires during training, the loss may be leaving little pressure to push far-apart negative pairs to confidently-negative logits even though real edges are reasonably well separated (F1≈0.7–0.76 on the balanced eval set). Flagged as worth investigating, not yet root-caused.
- **`k` still controls edge compression the same way as Cora**: `k=1` gives `|A_z|` at ~42–45% of `|E|` across all `d`; `k≥2` again makes A_z denser than the input (up to 8.7× at `k=20`).
- Fallback pick (`select_compression_optimal`, floor not met): highest raw F1 is `d=16, k=2` (F1=0.761), written to `configs/compression/pubmed.yaml`. Unlike Cora, this fallback pick is *not* one of the least-compressed points — `k=2` is close to the compression-favorable end of the grid, and mean-F1-by-d also happens to peak near `d=16`, so this particular fallback is a reasonably good pick on both axes by coincidence rather than by the selection logic accounting for it.
- **T3.4 trigger fires more decisively than Cora**: F1 at `d=128, k=20` is 0.673 — not just below the 0.90 floor, but the single worst point in the entire 36-point grid, and part of a clear monotonic downward trend rather than a flat plateau.

### Headline comparison

For each dataset, at the compression-optimal `(d, k)` (Cora, PubMed: fallback picks, floor not met):

| Dataset | N | F | \|E\| | d | \|A_z\| | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 1433 | 5278 | 16 | 38891 | 0.0112 | 7.369 | 0.828 | 1.094 |
| CiteSeer | 3327 | 3703 | 4732 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| PubMed | 19717 | 500 | 44324 | 16 | 39107 | 0.0320 | 0.882 | 0.761 | 1.0000073 |

Note: Cora's `|E|=5278` here (vs. `5429` in Phase 0/1's link-prediction table) because `full_graph_split` removes self-loops the same way `split_edges` does, but the two counts were computed independently — both are correct for their respective splits.

---

## V-4: Decoder Fallback (conditional) 🔶 Triggered for Cora and PubMed, not yet implemented

Triggered only if `reconstruction_f1` at `(d=128, k=20)` is below 0.90 for a given dataset.

| Check | Pass condition | Result |
|---|---|---|
| Trigger evaluated | Recorded per-dataset whether the trigger fired, with the F1 numbers | ✅ Cora: **triggered** (F1=0.8235 at d=128,k=20, < 0.90). ✅ PubMed: **triggered**, more decisively (F1=0.673 at d=128,k=20 — the worst point in PubMed's entire grid, part of a monotonic decline, not a plateau). CiteSeer ⬜ pending its sweep |
| If triggered: decoder implemented | `LatentGraphDecoder` shape/gradient tests pass | ⬜ not yet built — two of three datasets now confirm the pattern (Cora: flat-and-low; PubMed: actively decreasing with capacity), which is a stronger signal to implement now rather than wait for CiteSeer |
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
