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
| Fidelity floor met at some grid point | At least one `(d, k)` per dataset achieves `reconstruction_f1 ≥ 0.90` | ❌ **None of the three datasets meet this** — Cora best F1 0.828, CiteSeer best F1 0.819, PubMed best F1 0.777 (updated 2026-07-13, was 0.761 — see PubMed findings below) |
| W&B logging | `compression-sweep-{dataset}` group contains 36 runs per dataset with all FR-3 metrics | ✅ all three (36 runs logged each) |

### Cora findings (full results: `results/compression/cora.csv`)

- **F1 is flat across the entire grid** (0.813–0.828, a 1.5-point spread over all 36 `(d,k)` combinations) — more latent capacity or a denser latent graph buys essentially nothing. This points at the plain inner-product decoder as the bottleneck rather than the compression ratio.
- **`k`, not `d`, controls edge compression.** At `k=1`, `|A_z|` is ~37% of `|E|` (genuine compression); every `k≥2` makes A_z denser than the input graph (up to 7.5× at the NAS-best `k=20`).
- Since no point met the 0.90 floor, `configs/compression/cora.yaml` was written via the fallback (highest raw F1: d=16, k=20) — see `select_compression_optimal` in `src/gvls/compression/sweep.py`. Note this fallback pick is one of the *least* compressed points in the grid (edge ratio 7.37×); `d=8, k=1` reaches nearly the same F1 (0.825) at a fraction of the size on both axes and is arguably the more useful operating point for the QGNN use case. Full write-up in `README.md`.

### PubMed findings (full results: `results/compression/pubmed.csv`)

**Rerun 2026-07-13, superseding the findings below — see V-8.** The original run (2026-07-08) showed a clean "capacity hurts" pattern, which turned out to be caused by the same ELBO KL-normalization bug diagnosed in V-8: PubMed's `graph_mrf` KL term was an un-normalized sum over `N=19717`, so `β·KL` dominated the loss and pulled `μ` toward the prior in a way that scaled with capacity. After the fix (`src/gvls/losses/elbo.py`), the grid was rerun and the pattern is gone:

- **Capacity no longer hurts.** Mean F1 by `d`: 0.7725 (`d=4`) → 0.7716 → 0.7713 → 0.7736 → 0.7744 → **0.7771** (`d=128`) — a shallow dip then a rise, essentially flat, the opposite of the original monotonic decline. `d=128, k=20` (PubMed's Phase 2 NAS-best architecture) is now F1=0.7768 — tied for the *best* end of the grid instead of the worst.
- **`k` now has ~zero effect on F1, for the same mechanistic reason as CiteSeer.** Mean F1 by `k` is 0.7734 at every one of the 6 tested `k` values, to 4 decimal places. PubMed's compression-optimal config has `mp_rounds=0`, so `A_z` only reaches training through the `graph_mrf` KL term (not the reconstruction logits directly). Before the fix, that KL term was artificially amplified, so `A_z`'s indirect influence produced large, real F1 differences as `k` varied; correctly scaled, that influence is now negligible — the same "`A_z`'s only path into the objective is too weak to matter" conclusion as CiteSeer, reached by a different mechanistic route (CiteSeer: no path at all; PubMed: a path that's now properly weak instead of artificially strong).
- **`bits_per_edge` is no longer suspiciously exact.** It was previously ≈1.0 at every grid point to within 1.6e-3 (flagged as "worth investigating, not yet root-caused") — now understood as a likely symptom of the same bug (KL domination forcing `μ` into a near-identical near-zero configuration regardless of `(d,k)`, so the resulting logits barely varied). Post-fix, `bits_per_edge` is a real, still-imperfect ~1.13–1.14 across the grid — calibration remains unresolved (matches the pattern seen in T3.6's pooling-sweep rerun), but it's no longer an artifact of this specific bug.
- **`k` still controls edge compression the same way as Cora**: `k=1` gives `|A_z|` at ~44–45% of `|E|` across all `d`; `k≥2` again makes A_z denser than the input (up to 8.9× at `k=20`).
- New selection (`select_compression_optimal`, floor still not met): highest raw F1 is now `d=128, k=1` (F1=0.7772), written to `configs/compression/pubmed.yaml` — a complete reversal from the old fallback pick's `d=16` (dimensionality ratio `d/F` jumped from 0.032 to 0.256, 8× larger, though edge ratio improved from 0.890 to 0.445). The best fidelity/size trade-off is arguably `d=4, k=1` (F1=0.7725, only 0.005 below the new best, at 1/32 the dimensionality ratio).
- **T3.4's trigger still fires** (no point reaches 0.90), but the reasoning has changed: this is now a flat plateau (max F1=0.777) like Cora and CiteSeer, not a declining curve where the NAS-best config was the single worst point in the grid.
- **Consequence for T3.6:** the pooling sweep (`results/compression/pubmed_pooling.csv`, confirmed working in V-8) was run against the *old* `configs/compression/pubmed.yaml` (`d=16, k=2`), which this rerun has superseded (`d=128, k=1`). **Rerun 2026-07-13 (commit `2ad521c`) — it re-collapsed**, at `pool_ratio=0.0625` (`M=1232`, F1 back to exactly 0.6667), with the non-collapsed points also down (0.717–0.725 vs. 0.745–0.753 under the old `d=16,k=2` config). Root-caused (not by the ELBO bug, already fixed) to `k=1` itself: `LatentGraphLearner` at `k=1` produces a near-spanning-tree `A_z` (edge count ≈ `M−1` at every pool size, confirmed directly from the CSV) rather than a meaningfully connected latent graph, and `select_compression_optimal`'s fallback picked it only because F1 is now flat across `k` (no tie-breaking toward a more reasonably-connected `k` in that branch). Response: `k=1` removed from the sweep grid (`configs/experiment/compression_sweep.yaml`, Design Decision 8, plan.md) rather than patching the fallback's tie-breaking — PubMed's `(d,k)` sweep and its downstream compression-optimal config, and this pooling sweep, all need a further rerun once `k=1` is out of the grid.

<details>
<summary>Original findings (2026-07-08, predate the ELBO fix) — kept as a historical record</summary>

- **Capacity hurts, not helps.** Mean F1 falls monotonically as `k` grows (0.745→0.716 from `k=1` to `k=20`) and, at fixed `k=20`, falls monotonically as `d` grows (0.742→0.673 from `d=4` to `d=128`). PubMed's own Phase 2 NAS-best architecture is `d=128, k=20` — i.e. the config tuned for link-prediction AUC lands on the *worst* point in this compression grid (F1=0.673).
- **`bits_per_edge` is degenerate — ≈1.0 exactly at every grid point** (1.0000017–1.0015761). Per Phase 0's convention this corresponds to logit≈0 (maximum uncertainty). Likely explanation: PubMed's pair space (~194M) forces the sampled-estimate path (`dense_pair_limit`), and the sample is >99.97% negative pairs; combined with the extreme `pos_weight` (~4384×, from `(N²−E)/E`) that PubMed's scale requires during training, the loss may be leaving little pressure to push far-apart negative pairs to confidently-negative logits even though real edges are reasonably well separated (F1≈0.7–0.76 on the balanced eval set). Flagged as worth investigating, not yet root-caused.
- **`k` still controls edge compression the same way as Cora**: `k=1` gives `|A_z|` at ~42–45% of `|E|` across all `d`; `k≥2` again makes A_z denser than the input (up to 8.7× at `k=20`).
- Fallback pick (`select_compression_optimal`, floor not met): highest raw F1 is `d=16, k=2` (F1=0.761), written to `configs/compression/pubmed.yaml`. Unlike Cora, this fallback pick is *not* one of the least-compressed points — `k=2` is close to the compression-favorable end of the grid, and mean-F1-by-d also happens to peak near `d=16`, so this particular fallback is a reasonably good pick on both axes by coincidence rather than by the selection logic accounting for it.
- **T3.4 trigger fires more decisively than Cora**: F1 at `d=128, k=20` is 0.673 — not just below the 0.90 floor, but the single worst point in the entire 36-point grid, and part of a clear monotonic downward trend rather than a flat plateau.

</details>

### CiteSeer findings (full results: `results/compression/citeseer.csv`)

- **`k` has essentially zero effect on F1 — a mechanistic result, not an empirical one.** CiteSeer's Phase 2 NAS-best config is `mp_rounds=0, prior=isotropic`. With `mp_rounds=0`, `z̃ = z` unconditionally, so `A_z` never reaches the reconstruction logits; with `prior=isotropic`, the KL term ignores `A_z` too. So `A_z` has **no path into the loss or the output** for this config — varying `k` only changes a value that's computed and discarded. F1 is bit-identical across all 6 `k` values at `d ∈ {4,8,128}`; the ~1e-4 residual differences at `d ∈ {16,32,64}` are too large for float32 rounding noise but have no identified causal path from `k` (no stochastic ops in `LatentGraphLearner`) — most likely floating-point non-determinism from parallel execution accumulating over 200 epochs.
- **`|A_z|/|E|` still varies with `k` exactly like Cora/PubMed** (`k=1` → ~49–50% of `|E|`; `k≥2` → denser than input, up to 10.5× at `k=20`) — but here it's a purely decorative axis given the config above.
- **F1 range (0.809–0.819) and ceiling are close to Cora's** — both plateau well short of 0.90, unlike PubMed's actively-declining curve.
- Fallback pick (floor not met): highest raw F1 is `d=16, k=3` (F1=0.8188), written to `configs/compression/citeseer.yaml`.
- **T3.4 trigger fires for CiteSeer too — three for three.**

### Cross-dataset synthesis

**Updated 2026-07-13 after the PubMed rerun.** None of the three datasets reach the 0.90 fidelity floor anywhere in their 36-point grids, and **all three now show the same flat-vs-(d,k) pattern** — PubMed's original declining curve was an artifact of the ELBO KL-normalization bug (V-8), not a real capacity/rate-distortion effect. Two of three NAS-best configs (CiteSeer, PubMed) use `mp_rounds=0`, meaning GVLS's latent message-passing mechanism is inactive for the majority of these runs. For CiteSeer, `A_z` is provably inert end-to-end (no path into the loss at all). For PubMed, `A_z`'s only path into training is through the `graph_mrf` KL term — and now that the KL is correctly scaled, that path is too weak to move F1 meaningfully either, which is why PubMed's `k`-sweep is now just as flat as CiteSeer's. So for two of the three datasets, flatness now has a shared underlying cause: `A_z` barely (or doesn't at all) influence the objective these configs actually train against. This remains a convergent signal that a decoder/architecture change (T3.4), not further compression-ratio tuning, is the next lever to pull for reaching the 0.90 floor.

### Headline comparison

For each dataset, at the compression-optimal `(d, k)` (all three: fallback picks, floor not met). **PubMed's row updated 2026-07-13** (was `d=16,k=2`, F1=0.761 — see PubMed findings above for the historical row):

| Dataset | N | F | \|E\| | d | \|A_z\| | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 1433 | 5278 | 16 | 38891 | 0.0112 | 7.369 | 0.828 | 1.094 |
| CiteSeer | 3327 | 3703 | 4552 | 16 | 6780 | 0.0043 | 1.489 | 0.819 | 1.063 |
| PubMed | 19717 | 500 | 44324 | 128 | 19715 | 0.2560 | 0.445 | 0.777 | 1.128 |

Note: the `|E|` values here (Cora 5278, CiteSeer 4552, PubMed 44324) differ slightly from Phase 0/1's link-prediction table (5429, 4732, 44338) because `full_graph_split` removes self-loops the same way `split_edges` does, but the two counts were computed independently — both are correct for their respective splits.

---

## V-4: Decoder Fallback (conditional) — **Superseded 2026-07-09, revived 2026-07-13** ✅ Head-to-head sweep complete — **not adopted**, mixed-to-negative result

Triggered only if `reconstruction_f1` at `(d=128, k=20)` is below 0.90 for a given dataset.

| Check | Pass condition | Result |
|---|---|---|
| Trigger evaluated | Recorded per-dataset whether the trigger fired, with the F1 numbers | ✅ Cora: **triggered** (F1=0.8235 at d=128,k=20, < 0.90). ✅ PubMed: **triggered** (2026-07-09 number, now stale — see note below). ✅ CiteSeer: **triggered** (F1=0.8140 at d=128,k=20, < 0.90; also the dataset where `A_z` is provably inert given its NAS-best config — see CiteSeer findings above). **All three datasets trigger, and still do after the 2026-07-13 PubMed rerun** (0.7768 < 0.90). |
| If triggered: decoder implemented | `LatentGraphDecoder` shape/gradient tests pass | ✅ **Revived 2026-07-13** (see plan.md Design Decision 8) — `src/gvls/models/decoder.py`, `tests/test_decoder.py` (5 tests: output shape, gradient flow to `z_tilde` and the decoder's own weight, sensitivity to `A_z`, zero-`A_z` reduces to the plain inner product). `train_gvls_full_graph`/`evaluate_compression` (`src/gvls/compression/sweep.py`) take a `decoder: "inner_product" \| "graph_conditioned"` parameter; `experiments/compression_sweep.py` exposes it as `experiment.decoder`. `tests/test_compression_sweep.py` adds an end-to-end smoke test through the sweep path plus an invalid-decoder-string test. Manually smoke-tested through the actual CLI (Cora, `d=8,k=2`, 3 epochs) — writes to `results/compression/cora_graph_decoder.csv` / `configs/compression/cora_graph_decoder.yaml`, baseline files untouched, as designed. |
| If triggered: F1 comparison reported | Head-to-head F1 at matched `(d, k)`, baseline vs. A_z-conditioned decoder | ✅ 2026-07-13 (commit `627f4e7`), all three datasets, full 30-point grid each — see results below. **Verdict: not adopted.** |

**Note on PubMed's original F1=0.673:** that number predates the ELBO KL-normalization fix (V-8) and is stale — PubMed's rerun (§V-3, 2026-07-13) gives F1=0.7768 at `d=128,k=20`, no longer the worst point in the grid. The trigger still fires either way (both numbers are below 0.90); kept here as the historical record of the original trigger evaluation. This is also, independently, part of why T3.4 was revived: with the bug fixed, the remaining gap to 0.90 is now a flat plateau across all three datasets rather than a declining curve, pointing more squarely at the decoder as the next lever (see plan.md Design Decision 8).

### Head-to-head results (2026-07-13): `inner_product` baseline vs. `graph_conditioned` decoder

Matched `(d,k)` comparison, `results/compression/{name}.csv` vs. `results/compression/{name}_graph_decoder.csv` (both use the post-`k=1`-removal grid, `k ∈ {2,3,5,10,20}` × `d ∈ {4,8,16,32,64,128}`, 30 points):

| Dataset | Mean ΔF1 | Range | Collapsed points (F1≈0.6667) |
|---|---|---|---|
| Cora | **−0.0089** | −0.033 to +0.001 | 0/30 |
| CiteSeer | **+0.0032** | −0.003 to +0.020 | 0/30 |
| PubMed | **−0.0538** | −0.106 to −0.009 | **12/30** |

**Cora: uniformly worse, and worse with more capacity.** Every point but two is negative; damage grows with `d` (−0.018 to −0.033 at `d=128`). Cora's NAS-best config already has `mp_rounds=1`, so `A_z` already had a real path into the reconstruction before this decoder existed — the extra, freshly-initialized, redundant message-passing round adds noise and competing parameters rather than new information.

**CiteSeer: modest, real improvement — the one case that validates the original hypothesis.** Mean +0.0032, strongest at `d=4` (+0.017 to +0.020), new ceiling 0.829 (up from 0.819). CiteSeer's `mp_rounds=0, prior=isotropic` config left `A_z` provably inert before this decoder — it's the only thing in this experiment that ever gave CiteSeer's `A_z` a path into training, and it measurably helped.

**PubMed: a new, severe collapse — 12/30 points hit the exact F1=0.6667 signature.** Every point at `d=4` and `d=8`, plus `d=16` at `k∈{2,3}`, collapsed to the same always-predict-edge signature seen twice before (V-7's assignment collapse, V-8's KL-normalization bug) — a *third*, distinct instance of the same symptom with a new cause. Non-collapsed PubMed points are also substantially worse than baseline (−0.006 to −0.09). Collapse severity tracks capacity inversely: `d=4` collapses at every `k`; by `d≥32` it mostly clears, though `d=32,k=2` remains badly degraded (0.695 vs. 0.774) without hitting the exact collapse signature. The ELBO KL-normalization bug (V-8) is already fixed and doesn't depend on decoder choice, so it's ruled out as the cause here. Likely mechanism (not directly verified): PubMed's extreme `pos_weight` (~4384×, the most severe class imbalance of the three datasets) combined with a brand-new, randomly-interacting transformation (`LatentGraphDecoder`'s weight starts at identity, `A_z` itself still early in learning) sitting directly in the reconstruction path — small `d` gives the model the least redundancy to absorb that instability without falling into the same collapse basin diagnosed twice already in this phase.

**Broader implication — this undercuts T3.4's original premise rather than confirming it.** Cora (`A_z` connected via `mp_rounds=1`) and CiteSeer (`A_z` provably disconnected) had nearly identical baseline F1 ceilings (~0.82) despite completely different `A_z`-connectivity situations. If `A_z`'s path into the decoder were really what capped fidelity at ~0.82, Cora (already connected) shouldn't have plateaued at the same level as CiteSeer (never connected) — this result points *away* from decoder/`A_z`-connectivity as the bottleneck, not toward it.

**Decision: `graph_conditioned` is not adopted as the default decoder for any dataset.** It's a net loss for Cora, a severe regression (including new collapses) for PubMed, and a small win for CiteSeer specifically because CiteSeer was the one dataset with zero baseline `A_z`-connectivity to begin with — not a generalizable result. The module (`src/gvls/models/decoder.py`) and the `experiment.decoder` sweep option are kept in the codebase (implemented, tested, and usable via `experiment.decoder=graph_conditioned`) as a documented negative result, not removed.

---

## V-5: Node Classification (secondary) ⬜

| Check | Pass condition | Result |
|---|---|---|
| Linear probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| MLP probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| CiteSeer / PubMed probes | Reported if time allows (secondary priority — may slip to Phase 4) | ⬜ |

---

## V-7: Node-Count Pooling (new, T3.6) 🔶 Module built and fixed; production sweep not yet rerun with the fix

Reframing decision made 2026-07-09 — supersedes V-4's decoder-fallback direction. See `specs/roadmap.md`, `specs/phase3/plan.md`, and `mission.md`'s changelog for the full rationale.

| Check | Pass condition | Result |
|---|---|---|
| Pooling module built | `LatentGraphPooling` shape/gradient tests pass (`S` rows sum to 1, pooled `(μ, log_var)` shape `(M, d)`, gradients reach assignment logits and pooled params) | ✅ `tests/test_pooling.py` |
| Unpool shape correct | `Â` has shape `(N, N)` regardless of `M` | ✅ `test_unpool_shape_independent_of_m` |
| Pooling sweep complete per dataset | `pool_ratio ∈ {0.5, 0.25, 0.125, 0.0625}` all trained and logged to `results/compression/{dataset}_pooling.csv`, `(d, k)` held fixed at each dataset's T3.3 compression-optimal config | 🔶 all three CSVs exist (committed 2026-07-09) but predate the collapse fix below — **stale, must be regenerated** |
| Node-count ratio computed | `node_compression_ratio = M/N` reported per grid point | ✅ present in the (stale) CSVs; formula unaffected by the fix |
| Assignment storage cost reported | `assignment_storage_bits(N, M)` reported per grid point, so total compressed size is honestly accounted (not treating `S` as free) | ✅ present in the (stale) CSVs; formula unaffected by the fix |
| Fidelity vs. node-count tradeoff characterized | Reconstruction F1 reported as a function of `M/N`, independent of the `(d, k)` axes T3.3 already covers | ❌ not yet — the stale CSVs show no tradeoff at all (see collapse finding below); needs a rerun |
| W&B logging | `compression-pooling-sweep-{dataset}` group contains one run per `pool_ratio` per dataset | ✅ (for the stale runs; will re-log on rerun) |

### Cold-start collapse: diagnosis and fix (2026-07-09)

The first production sweep (all three datasets, committed as `results/compression/{cora,citeseer,pubmed}_pooling.csv`) converged to a **degenerate always-predict-edge classifier on every single grid point** — `reconstruction_f1` was **exactly 0.6667** (the value produced by a trivial constant-positive predictor on a 1:1 balanced eval set: precision=0.5, recall=1.0) at all four `pool_ratio` values, for all three datasets, with `bits_per_edge ≈ 1.0` (logit ≈ 0, maximum uncertainty) throughout. This was not a real rate-distortion result.

**Root cause**, confirmed by direct inspection of a trained Cora model (`pool_ratio=0.5`): the unpooled reconstruction logits were nearly constant across the *entire* `N×N` matrix (`min=0.00207, max=0.00207`), with 100% of pairs predicted positive. Tracing further: the pooled `M×M` similarity matrix (`z̃_pooled z̃_pooledᵀ`) was itself nearly flat (`min=0.0532, max=0.0585`, i.e. no real structure), because the assignment `S` starts near-uniform at initialization (a plain linear layer over `M=1354` clusters gives near-uniform softmax output before training) — and because `S`'s rows sum to 1, unpooling an (almost) constant `M×M` matrix produces an *exactly* constant `N×N` reconstruction (`S · c𝟙𝟙ᵀ · Sᵀ = c𝟙𝟙ᵀ`). A constant output gives `S` no gradient to escape the collapse — a self-reinforcing dead end. This is a well-documented failure mode of naive soft-assignment pooling (it is why DiffPool, Ying et al. 2018, ships with exactly the two auxiliary losses added below).

**Fix, validated on Cora at `pool_ratio=0.5` (200 epochs each):**

| Config | Reconstruction F1 |
|---|---|
| No auxiliary losses (original) | 0.6667 (degenerate) |
| `entropy_weight=0.1` only | 0.6667 (unchanged — entropy alone insufficient) |
| `entropy_weight=0.1, aux_link_weight=1.0` | 0.686 |
| `entropy_weight=0.1, aux_link_weight=5.0` | **0.782** (selected default) |
| `entropy_weight=0.1, aux_link_weight=10.0` | 0.784 (diminishing returns beyond 5.0) |
| `entropy_weight=0.1, aux_link_weight=20–50` | 0.77–0.78 (no further improvement) |

`entropy_weight=0.1, aux_link_weight=5.0` are the sweep's defaults (`configs/experiment/pooling_sweep.yaml`). An intermediate bug was also caught and fixed during this validation: the first version of `assignment_link_loss` included the diagonal of `S·Sᵀ` (which is naturally close to 1 for a confident assignment) against `adj_true`'s diagonal (always 0, no self-loops) — directly fighting the entropy term. Fixed by excluding the diagonal, consistent with this codebase's no-self-loop convention elsewhere (e.g. `edge_compression_ratio`).

**Status:** the fix is implemented and unit-tested (`tests/test_pooling.py`, 20 tests covering `assignment_entropy` and `assignment_link_loss` directly). **The production sweep has not yet been rerun with the fix** — the committed `*_pooling.csv` files reflect the pre-fix degenerate collapse and must not be used for figures or conclusions until regenerated.

---

## V-8: ELBO KL-normalization bug (new, found 2026-07-11 while investigating T3.6's residual PubMed collapse) 🔶 Fixed and confirmed on T3.3/T3.6; surfaced a separate `k=1` structural issue (Design Decision 8) that still needs a rerun to clear

**Context.** After V-7's entropy/link-loss fix, the regenerated `results/compression/{dataset}_pooling.csv` files still showed PubMed stuck at the exact pre-fix collapse signature (`reconstruction_f1 = 0.6667`) at its two largest pool sizes (`pool_ratio=0.5` → `M=9858`, `pool_ratio=0.25` → `M=4929`), while `pool_ratio=0.125`/`0.0625` (`M=2465`/`1232`) escaped it (F1=0.719/0.732). Cora and CiteSeer never exhibited this. Investigating why the fix didn't fully transfer to PubMed's larger `M` values surfaced a second, independent bug.

**Root cause.** `kl_isotropic` and `kl_graph_mrf` (`src/gvls/losses/elbo.py`) both returned a raw, un-normalized `torch.sum()` over the `M` (or `N`, pre-pooling) nodes being scored, while `elbo()`'s reconstruction term uses `reduction="mean"` over all `N²` node-pair logits. This meant `β·KL`'s absolute magnitude scaled **linearly with the number of nodes/clusters it was computed over**, while `recon_loss`'s magnitude did not. At a fixed `N` (Phase 0–2, and T3.3's `M=N` compression baseline) this mismatch was invisible — it just meant `β` was implicitly doing double duty as a per-dataset-N scale correction. T3.6's `pool_ratio` sweep varies `M` over almost two orders of magnitude (169–9858 across the three datasets), which is what exposed it.

**Evidence** (full diagnostic scripts and reasoning in conversation; not committed, reproducible on request):
- *Analytical magnitude check*: computing `kl_isotropic`/`kl_graph_mrf` at each dataset's real `(M, d, β, prior)` grid points (synthetic init-scale `mu`/`log_var`) showed `raw_KL` scaling essentially linearly with `M` for every dataset. For Cora/CiteSeer (`prior=isotropic`, `β≈1e-5`), `β·KL` stayed negligible (≤0.003× the reconstruction-loss scale) at every `M` tested. For PubMed (`prior=graph_mrf`, `β=7.47e-4` — the largest `β` of the three, and independently ~18–20× more KL magnitude than `isotropic` at matched `mu`/`log_var`/`M`), `β·KL` reached **12–24× the reconstruction-loss scale** at `M=9858`/`4929`, versus 3–6× at `M=2465`/`1232` — tracking the exact boundary between collapsed and non-collapsed grid points in the production CSV.
- *Causal ablation*: training `PooledGVLS` on a structured synthetic graph (stochastic block model with class-correlated features, so there's real learnable homophily — an initial attempt on a purely random, structureless graph collapsed at *every* hyperparameter combination and was discarded as a flawed test) at fixed `M`, varying only `β`, showed the expected direction: PubMed's actual `β` (7.47e-4) versus a Cora-scale `β` (1.9e-5) produced `mean|μ|=0.82` vs `1.34` and F1=0.72 vs 0.75 — lower `β` let `μ` move further from the prior and improved fidelity. The effect was directionally consistent but modest at the `M` tested (this synthetic graph is 5× smaller than real PubMed, so absolute KL magnitude at matched `M` doesn't correspond to the same point on PubMed's collapse curve) — a partial, qualitative corroboration, not a full replication of PubMed's exact threshold.
- *Retrospective fit to an already-known anomaly*: T3.3's PubMed `(d,k)` sweep (§V-3) showed an unexplained **"capacity hurts, not helps"** pattern — F1 falling monotonically as `d` grows (0.742→0.673). `d` also scales `kl_graph_mrf`'s magnitude (`logdet_omega = d·logdet`, plus `d`-width `trace_term`/`mu_quad` sums). Same bug, smaller dynamic range (`d` only ranged 4–128, vs. pooling's `M` ranging up to 9858), producing a milder "capacity hurts" symptom there instead of outright collapse. This is a plausible, not fully isolated, explanation for that finding too.

**This is a compounding cause, not a replacement for V-7's fix.** The already-diagnosed assignment-collapse mechanism (`S` starting near-uniform, self-reinforcing once the unpooled reconstruction goes flat) is a separate pathway, and the entropy/link-loss auxiliary terms it's fixed by remain necessary. The KL-normalization bug independently pulls `μ` toward the prior at large `M`, which can produce the same flat-reconstruction symptom even if `S` itself is behaving reasonably — and the two failure modes likely compound each other in the pooled case (pooling itself, via a near-uniform `S`, already smooths/averages `μ` across nodes at initialization before any KL pressure is applied).

**Fix, implemented 2026-07-11.** Both `kl_isotropic` and `kl_graph_mrf` now divide their returned scalar by the node count (`N` or `M`) they're computed over, so `β·KL` reflects a per-node KL cost independent of graph/cluster size — matching `recon_loss`'s existing per-pair mean convention. Verified analytically: post-fix, `raw_KL` is flat (~2.2, prior `graph_mrf`) across PubMed's entire `M` grid instead of scaling 2736→22000, and `β·KL` is negligible (≤0.002×) relative to `recon_loss` at every grid point for all three datasets — the 12–24× domination is gone.

| Check | Pass condition | Result |
|---|---|---|
| Bug root-caused | KL-magnitude scaling with node count identified and evidenced (analytical + causal ablation + retrospective fit to T3.3's PubMed anomaly) | ✅ |
| Fix implemented | `kl_isotropic`/`kl_graph_mrf` normalized by node count | ✅ `src/gvls/losses/elbo.py` |
| Regression tests added | Node-count invariance verified directly (tiling a single node's distribution, and a block-diagonal MRF replica, must not change the returned per-node KL) | ✅ `test_kl_isotropic_invariant_to_node_count`, `test_kl_graph_mrf_invariant_to_node_count` in `tests/test_elbo.py` |
| Existing test suite still passes | No regressions from the normalization change | ✅ 143/143 (was 115 pre-T3.6, +13 in `test_elbo.py`, +remainder from T3.6's own test files) |
| T3.6 pooling sweep rerun with the fix | All three `*_pooling.csv` regenerated | ✅ 2026-07-12 — see "Confirmed results" below |
| T3.3 `(d,k)` baseline rerun with the fix | `results/compression/pubmed.csv` regenerated (Cora/CiteSeer's `.csv` files don't need a rerun — their `β` was already negligible, see the analytical check above) | ✅ 2026-07-13 (commit `eec133b`) — **directly confirms the retrospective "capacity hurts" hypothesis.** The declining-with-capacity pattern is gone: mean F1 by `d` now rises from 0.7725 (`d=4`) to 0.7771 (`d=128`) instead of falling from 0.742 to 0.673, and mean F1 by `k` is flat at 0.7734 for every tested `k` instead of falling from 0.745 to 0.716. `select_compression_optimal` picked `d=128, k=1` (F1=0.7772) instead of `d=16, k=2` (F1=0.761) — see updated §V-3 for the full before/after. |
| T3.6 PubMed pooling sweep, rerun against the new `d=128,k=1` config | `results/compression/pubmed_pooling.csv` regenerated against the post-T3.3-rerun `configs/compression/pubmed.yaml` | 🔶 2026-07-13 (commit `2ad521c`) — **rerun completed, but re-collapsed** at `pool_ratio=0.0625` (`M=1232`, F1 exactly 0.6667 again), with the non-collapsed points also worse (0.717–0.725 vs. 0.745–0.753 under the old `d=16,k=2` config). This is a *third*, distinct collapse instance — not the assignment-init issue (V-7, already fixed) or the KL-normalization bug (this section, already fixed). Root-caused to `k=1`'s latent graph structure: `LatentGraphLearner` at `k=1` produces `A_z` with edge count ≈ `M−1` at every pool size (a near-spanning-tree, confirmed directly from `num_latent_edges` in the CSV), which `select_compression_optimal`'s fallback selected only because F1 was statistically flat across `k` post-fix, with no tie-break toward a better-connected graph. **Response (2026-07-13, Design Decision 8): `k=1` removed from `configs/experiment/compression_sweep.yaml`'s grid**, rather than patching the fallback's tie-breaking logic — a decision, not a further bug fix. T3.3's `(d,k)` sweep, its compression-optimal selection, and this pooling sweep all need to be rerun once more with `k=1` out of the grid — not yet done. |

### Confirmed results (T3.6 pooling sweep rerun, 2026-07-12)

The rerun (`results/compression/{cora,citeseer,pubmed}_pooling.csv`, commit `0f5bfd0`) confirms the diagnosis precisely:

| Dataset | pool_ratio | M | F1 before fix | F1 after fix | Δ |
|---|---|---|---|---|---|
| **PubMed** | 0.5 | 9858 | 0.6667 (collapsed) | **0.7483** | **+0.082** |
| | 0.25 | 4929 | 0.6667 (collapsed) | **0.7498** | **+0.083** |
| | 0.125 | 2465 | 0.7195 | 0.7454 | +0.026 |
| | 0.0625 | 1232 | 0.7322 | 0.7534 | +0.021 |
| Cora | 0.5 / 0.25 / 0.125 / 0.0625 | 1354–169 | 0.685–0.723 | 0.685–0.726 | ±0.01 (run-to-run noise) |
| CiteSeer | 0.5 / 0.25 / 0.125 / 0.0625 | 1664–208 | 0.815–0.853 | 0.817–0.850 | ±0.003 (run-to-run noise) |

- **PubMed's two previously-collapsed grid points are fixed** — both moved from the exact degenerate signature (F1=0.6667) to real, non-degenerate results (~0.75), matching the fix's predicted mechanism.
- **Cora and CiteSeer moved by less than 1%** — consistent with the analytical check, which showed `β·KL` was always negligible for both (tiny `β`, `isotropic` prior). This is a clean negative control: the fix changed exactly the thing it was supposed to change, and nothing else.
- **New headline for PubMed: node-count compression is now nearly free.** PubMed's `M=N` baseline at this `(d=16,k=2)` config is F1=0.761 (§V-3). The pooling sweep now sits at 0.745–0.753 across the *entire* `pool_ratio` grid, essentially flat with respect to `M` — a gap of only 0.008–0.016 versus the `M=N` baseline, down from a gap of up to 0.094 (and two outright-broken points) before the fix. A 16× node-count reduction (`pool_ratio=0.0625`) now costs almost nothing in fidelity.
- **Calibration (`bits_per_edge`) is still bad, independent of this fix, and got numerically worse at PubMed's `pool_ratio=0.5`** (1.49 → 5.22): the old value was artificially low because a collapsed, near-zero-logit constant output sits close to the BCE decision boundary; now that the model makes real, confident predictions, F1 is much better but those predictions remain poorly calibrated (`bits_per_edge` 4.1–5.2 across PubMed's whole grid, versus a random-predictor reference of 1.0). This is the same calibration issue flagged when the (still-buggy) pooling results were first analyzed — unrelated to the KL-normalization bug and not addressed by this fix.
- **Cora's non-monotonic `pool_ratio` curve and its gap versus the manual validation sweep (production ≈0.699 vs. the ≈0.78 reported for the hyperparameter selection at `pool_ratio=0.5`) are both unchanged** — expected, since Cora's `β` was never large enough for this bug to be active.

**Broader impact, flagged not resolved here.** `elbo()` is called from `experiments/train_gvls.py` (Phase 0/1) and `src/gvls/nas/objective.py` (Phase 2 NAS) as well as Phase 3's compression/pooling sweeps. Phase 2's NAS search selected each dataset's `β` (and, for PubMed, `prior=graph_mrf` itself) against the old, un-normalized KL scale — PubMed's NAS-best `β=7.47e-4` was implicitly compensating for (or fighting against) an `O(N)` KL term rather than the `O(1)` term the fix now produces. This doesn't invalidate Phase 0–2's link-prediction results (`configs/best/{dataset}.yaml`, kept fixed at a single `N=N_full` throughout, so no comparison across different node counts was ever made there), but it does mean those `β` values are calibrated to a different loss landscape than they'll now train under if reused. Whether to re-run Phase 2 NAS under the corrected KL convention is an open question — added to `specs/roadmap.md`'s Open Questions.

---

## V-6: Code Quality ✅

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All tests pass (including new Phase 3 tests) | ✅ 150/150 (2026-07-13, includes T3.4's revived `test_decoder.py` and the two new `test_compression_sweep.py` decoder tests) |
| `ruff check src/` | Zero violations | ✅ |
| `test_compression_sweep.py` runtime | Completes in under 60 seconds (2×2 grid, 10 epochs) | ✅ full file (11 tests) runs in ~2s |
