# Phase 3 — Validation

## Exit Criteria

---

## V-1: Compression Metrics ⬜

| Check | Pass condition | Result |
|---|---|---|
| `reconstruction_f1` perfect predictor | Returns 1.0 | ⬜ |
| `reconstruction_f1` all-zero predictor | Returns 0.0 with real edges present | ⬜ |
| `dim_compression_ratio` correctness | `dim_compression_ratio(32, 1433)` ≈ 0.0223 | ⬜ |
| `edge_compression_ratio` correctness | Matches hand-counted non-zero pattern on a toy 5-node A_z | ⬜ |
| `bits_per_edge` sampled ≈ exact | Sampled estimator agrees with exact full-pair computation (±0.01 bits/edge) on a small graph | ⬜ |

---

## V-2: Full-Graph Split ⬜

| Check | Pass condition | Result |
|---|---|---|
| All real edges present | `train_edge_index` contains every real edge (both directions) | ⬜ |
| No val/test edges | `full_graph_split` produces no held-out edges | ⬜ |
| Determinism | Same seed → identical split across runs | ⬜ |

---

## V-3: Rate-Distortion Sweep ⬜

| Check | Pass condition | Result |
|---|---|---|
| Cora grid complete | 36/36 `(d, k)` combinations trained and logged to `results/compression/cora.csv` | ⬜ |
| PubMed grid complete | 36/36 combinations logged to `results/compression/pubmed.csv` | ⬜ |
| CiteSeer grid complete | 36/36 combinations logged to `results/compression/citeseer.csv` | ⬜ |
| Compression-optimal configs written | `configs/compression/{cora,citeseer,pubmed}.yaml` exist and are valid `gvls.yaml`-schema configs | ⬜ |
| Fidelity floor met at some grid point | At least one `(d, k)` per dataset achieves `reconstruction_f1 ≥ 0.90` | ⬜ |
| W&B logging | `compression-sweep-{dataset}` group contains 36 runs per dataset with all FR-3 metrics | ⬜ |

### Headline comparison (fill in once the sweep runs)

For each dataset, at the compression-optimal `(d, k)`:

| Dataset | N | F | \|E\| | d | \|A_z\| | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 1433 | 5429 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| CiteSeer | 3327 | 3703 | 4732 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| PubMed | 19717 | 500 | 44338 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

---

## V-4: Decoder Fallback (conditional) ⬜ / N/A

Triggered only if `reconstruction_f1` at `(d=128, k=20)` is below 0.90 for a given dataset.

| Check | Pass condition | Result |
|---|---|---|
| Trigger evaluated | Recorded per-dataset whether the trigger fired, with the F1 numbers | ⬜ |
| If triggered: decoder implemented | `LatentGraphDecoder` shape/gradient tests pass | ⬜ / N/A |
| If triggered: F1 comparison reported | Head-to-head F1 at matched `(d, k)`, baseline vs. A_z-conditioned decoder | ⬜ / N/A |

---

## V-5: Node Classification (secondary) ⬜

| Check | Pass condition | Result |
|---|---|---|
| Linear probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| MLP probe trained | Accuracy and macro-F1 reported for at least Cora | ⬜ |
| CiteSeer / PubMed probes | Reported if time allows (secondary priority — may slip to Phase 4) | ⬜ |

---

## V-6: Code Quality ⬜

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All tests pass (including new Phase 3 tests) | ⬜ |
| `ruff check src/` | Zero violations | ⬜ |
| `test_compression_sweep.py` runtime | Completes in under 60 seconds (2×2 grid, 10 epochs) | ⬜ |
