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

## V-3: Rate-Distortion Sweep ✅ Complete (Cora, CiteSeer, PubMed)

| Check | Pass condition | Result |
|---|---|---|
| Cora grid complete | 36/36 `(d, k)` combinations trained and logged to `results/compression/cora.csv` | ✅ 36/36, 200 epochs each |
| PubMed grid complete | 36/36 combinations logged to `results/compression/pubmed.csv` | ✅ 36/36, run on a remote A100 (PubMed's NAS-best `prior=graph_mrf` KL is O(N³)/epoch — too slow on CPU, see `specs/phase3/plan.md`) |
| CiteSeer grid complete | 36/36 combinations logged to `results/compression/citeseer.csv` | ✅ 36/36, 200 epochs each |
| Compression-optimal configs written | `configs/compression/{cora,citeseer,pubmed}.yaml` exist and are valid `gvls.yaml`-schema configs | ✅ all three written (all fallback picks, see below) |
| Fidelity floor met at some grid point | At least one `(d, k)` per dataset achieves `reconstruction_f1 ≥ 0.90` | ❌ **None of the three datasets meet this** — Cora best F1 0.828, CiteSeer best F1 0.819, PubMed best F1 0.761 |
| W&B logging | `compression-sweep-{dataset}` group contains 36 runs per dataset with all FR-3 metrics | ✅ all three (36 runs logged each) |

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

### CiteSeer findings (full results: `results/compression/citeseer.csv`)

- **`k` has essentially zero effect on F1 — a mechanistic result, not an empirical one.** CiteSeer's Phase 2 NAS-best config is `mp_rounds=0, prior=isotropic`. With `mp_rounds=0`, `z̃ = z` unconditionally, so `A_z` never reaches the reconstruction logits; with `prior=isotropic`, the KL term ignores `A_z` too. So `A_z` has **no path into the loss or the output** for this config — varying `k` only changes a value that's computed and discarded. F1 is bit-identical across all 6 `k` values at `d ∈ {4,8,128}`; the ~1e-4 residual differences at `d ∈ {16,32,64}` are too large for float32 rounding noise but have no identified causal path from `k` (no stochastic ops in `LatentGraphLearner`) — most likely floating-point non-determinism from parallel execution accumulating over 200 epochs.
- **`|A_z|/|E|` still varies with `k` exactly like Cora/PubMed** (`k=1` → ~49–50% of `|E|`; `k≥2` → denser than input, up to 10.5× at `k=20`) — but here it's a purely decorative axis given the config above.
- **F1 range (0.809–0.819) and ceiling are close to Cora's** — both plateau well short of 0.90, unlike PubMed's actively-declining curve.
- Fallback pick (floor not met): highest raw F1 is `d=16, k=3` (F1=0.8188), written to `configs/compression/citeseer.yaml`.
- **T3.4 trigger fires for CiteSeer too — three for three.**

### Cross-dataset synthesis

None of the three datasets reach the 0.90 fidelity floor anywhere in their 36-point grids. Two of three NAS-best configs (CiteSeer, PubMed) use `mp_rounds=0`, meaning GVLS's latent message-passing mechanism is inactive for the majority of these runs; for CiteSeer specifically, `A_z` is provably inert end-to-end. This is a convergent signal across all three datasets that a decoder/architecture change (T3.4), not further compression-ratio tuning, is the next lever to pull.

### Headline comparison

For each dataset, at the compression-optimal `(d, k)` (all three: fallback picks, floor not met):

| Dataset | N | F | \|E\| | d | \|A_z\| | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 1433 | 5278 | 16 | 38891 | 0.0112 | 7.369 | 0.828 | 1.094 |
| CiteSeer | 3327 | 3703 | 4552 | 16 | 6780 | 0.0043 | 1.489 | 0.819 | 1.063 |
| PubMed | 19717 | 500 | 44324 | 16 | 39107 | 0.0320 | 0.882 | 0.761 | 1.0000073 |

Note: the `|E|` values here (Cora 5278, CiteSeer 4552, PubMed 44324) differ slightly from Phase 0/1's link-prediction table (5429, 4732, 44338) because `full_graph_split` removes self-loops the same way `split_edges` does, but the two counts were computed independently — both are correct for their respective splits.

---

## V-4: Decoder Fallback (conditional) — **Superseded 2026-07-09, not implemented**

Triggered only if `reconstruction_f1` at `(d=128, k=20)` is below 0.90 for a given dataset.

| Check | Pass condition | Result |
|---|---|---|
| Trigger evaluated | Recorded per-dataset whether the trigger fired, with the F1 numbers | ✅ Cora: **triggered** (F1=0.8235 at d=128,k=20, < 0.90). ✅ PubMed: **triggered**, more decisively (F1=0.673 at d=128,k=20 — the worst point in PubMed's entire grid, part of a monotonic decline). ✅ CiteSeer: **triggered** (F1=0.8140 at d=128,k=20, < 0.90; also the dataset where `A_z` is provably inert given its NAS-best config — see CiteSeer findings above). **All three datasets trigger.** |
| If triggered: decoder implemented | `LatentGraphDecoder` shape/gradient tests pass | ⬜ **superseded, not built** — the project pivoted to node-count pooling (T3.6, see V-7) instead of a decoder tweak at fixed `M=N`. Revisit only if T3.6 fails to close the fidelity gap on its own. |
| If triggered: F1 comparison reported | Head-to-head F1 at matched `(d, k)`, baseline vs. A_z-conditioned decoder | ⬜ superseded, not applicable |

---

## V-5: Node Classification (secondary) ⬜

| Check | Pass condition | Result |
|---|---|---|
| Linear probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| MLP probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| CiteSeer / PubMed probes | Reported if time allows (secondary priority — may slip to Phase 4) | ⬜ |

---

## V-7: Node-Count Pooling (new, T3.6) ⬜

Reframing decision made 2026-07-09 — supersedes V-4's decoder-fallback direction. See `specs/roadmap.md`, `specs/phase3/plan.md`, and `mission.md`'s changelog for the full rationale.

| Check | Pass condition | Result |
|---|---|---|
| Pooling module built | `LatentGraphPooling` shape/gradient tests pass (`S` rows sum to 1, pooled `(μ, log_var)` shape `(M, d)`, gradients reach assignment logits and pooled params) | ⬜ |
| Unpool shape correct | `Â` has shape `(N, N)` regardless of `M` | ⬜ |
| Pooling sweep complete per dataset | `pool_ratio ∈ {0.5, 0.25, 0.125, 0.0625}` all trained and logged to `results/compression/{dataset}_pooling.csv`, `(d, k)` held fixed at each dataset's T3.3 compression-optimal config | ⬜ |
| Node-count ratio computed | `node_compression_ratio = M/N` reported per grid point | ⬜ |
| Assignment storage cost reported | `assignment_storage_bits(N, M)` reported per grid point, so total compressed size is honestly accounted (not treating `S` as free) | ⬜ |
| Fidelity vs. node-count tradeoff characterized | Reconstruction F1 reported as a function of `M/N`, independent of the `(d, k)` axes T3.3 already covers | ⬜ |
| W&B logging | `compression-pooling-sweep-{dataset}` group contains one run per `pool_ratio` per dataset | ⬜ |

---

## V-6: Code Quality ✅

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All tests pass (including new Phase 3 tests) | ✅ 115/115 |
| `ruff check src/` | Zero violations | ✅ |
| `test_compression_sweep.py` runtime | Completes in under 60 seconds (2×2 grid, 10 epochs) | ✅ full file (7 tests) runs in ~2s |

**Note:** V-6's test count (115/115) predates T3.6. Once `test_pooling.py` is added, this count and the `ruff check` result should be re-verified and updated.
