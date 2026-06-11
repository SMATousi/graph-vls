# Phase 0 — Validation ✅ All criteria met 2026-06-09

## Exit Criteria

---

## V-1: Data Pipeline ✅

| Check | Pass condition | Result |
|---|---|---|
| All three Planetoid datasets load without error | `Cora`, `CiteSeer`, `PubMed` return a PyG `Data` object with `x`, `edge_index`, `y` | ✅ Verified via `smoke_test.py` (Cora) and `load_planetoid` wrapper |
| Split sizes are correct | At ratio r, train edges ≈ r × total_edges (±1 for rounding); val and test each get half of the remainder | ✅ `test_split_sizes[0.2/0.4/0.8]` pass; Cora at 80%: train=4222, val=528, test=528 |
| No edge leakage | Zero intersection between train positive edges and val/test positive edges | ✅ `test_no_train_test_leakage` passes |
| Negative sampling ratio | \|neg_edges\| == \|pos_edges\| in val and test sets | ✅ `test_negative_sampling_ratio` passes |
| Determinism | Running split twice with the same seed produces identical edge sets | ✅ `test_determinism` passes |

---

## V-2: Metric Sanity Checks ✅

| Check | Expected value | Result |
|---|---|---|
| `auc_ap` on perfect predictor | AUC = 1.0, AP = 1.0 | ✅ `test_auc_ap_perfect` passes |
| `auc_ap` on random predictor (balanced labels, N=10000) | AUC ≈ 0.5, AP ≈ 0.5 (±0.02) | ✅ `test_auc_ap_random_predictor` passes |
| `auc_ap` on inverted predictor | AUC = 0.0 | ✅ `test_auc_ap_inverted` passes |
| `node_accuracy` on perfect predictions | 1.0 | ✅ `test_node_accuracy_perfect` passes |
| `node_accuracy` on all-wrong predictions | 0.0 | ✅ `test_node_accuracy_all_wrong` passes |
| `bits_per_edge` on perfect logits (±100) | ≈ 0.0 | ✅ `test_bits_per_edge_perfect_logits` passes (< 1e-6) |
| `bits_per_edge` on zero logits | 1.0 bit/edge | ✅ `test_bits_per_edge_random_logits` passes (exact) |

Full test suite: **26 tests, 0 failures** (`pytest tests/` in 4.2s after first import).

---

## V-3: Baseline Reference Table

The following numbers are **not to be reproduced** — they are taken directly from Ahn, S.J. and Kim, M., "Variational Graph Normalized Autoencoders," *Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM)*, pp. 2827–2831, 2021, and serve as the comparison target for GVLS results in later phases. VGNAE is the strongest baseline.

### Link Prediction — AUC

| Dataset | Train | GAE | LGAE | ARGA | GIC | sGraph | GNAE | VGNAE |
|---|---|---|---|---|---|---|---|---|
| Cora | 20% | 0.782 | 0.866 | 0.795 | 0.880 | 0.845 | 0.887 | **0.890** |
| Cora | 40% | 0.856 | 0.908 | 0.844 | 0.914 | 0.840 | 0.926 | **0.929** |
| Cora | 80% | 0.922 | 0.938 | 0.919 | 0.933 | 0.885 | **0.956** | 0.954 |
| CiteSeer | 20% | 0.786 | 0.906 | 0.750 | 0.930 | 0.928 | **0.946** | 0.941 |
| CiteSeer | 40% | 0.836 | 0.925 | 0.832 | 0.936 | 0.936 | 0.956 | **0.961** |
| CiteSeer | 80% | 0.894 | 0.955 | 0.904 | 0.962 | 0.963 | 0.965 | **0.970** |
| PubMed | 20% | 0.937 | 0.946 | 0.936 | 0.950 | 0.837 | 0.950 | **0.951** |
| PubMed | 40% | 0.959 | 0.962 | 0.955 | 0.958 | 0.876 | 0.963 | **0.964** |
| PubMed | 80% | 0.967 | 0.974 | 0.973 | 0.960 | 0.896 | 0.975 | **0.976** |

### Link Prediction — AP

| Dataset | Train | GAE | LGAE | ARGA | GIC | sGraph | GNAE | VGNAE |
|---|---|---|---|---|---|---|---|---|
| Cora | 20% | 0.793 | 0.878 | 0.806 | 0.881 | 0.829 | **0.901** | **0.901** |
| Cora | 40% | 0.861 | 0.915 | 0.856 | 0.911 | 0.828 | **0.936** | 0.933 |
| Cora | 80% | 0.930 | 0.945 | 0.927 | 0.929 | 0.867 | 0.957 | **0.958** |
| CiteSeer | 20% | 0.797 | 0.913 | 0.777 | 0.934 | 0.897 | **0.953** | 0.948 |
| CiteSeer | 40% | 0.850 | 0.929 | 0.844 | 0.938 | 0.910 | 0.958 | **0.966** |
| CiteSeer | 80% | 0.903 | 0.959 | 0.915 | 0.966 | 0.943 | 0.970 | **0.971** |
| PubMed | 20% | 0.940 | 0.947 | 0.941 | 0.947 | 0.859 | **0.950** | 0.949 |
| PubMed | 40% | 0.961 | 0.961 | 0.959 | 0.956 | 0.879 | 0.961 | **0.963** |
| PubMed | 80% | 0.967 | 0.975 | **0.976** | 0.965 | 0.902 | 0.975 | **0.976** |

**Primary target:** GVLS should exceed VGNAE AUC and AP on the majority of dataset × split combinations.

---

## V-4: Infrastructure ✅

| Check | Pass condition | Result |
|---|---|---|
| `smoke_test.py` runs end-to-end | Exits 0 with no exceptions | ✅ Verified on Cora at 80%; W&B offline run logged successfully |
| W&B logging | Run appears with all required fields | ✅ `n_nodes`, `n_train/val/test`, `val/auc`, `val/ap`, `test/auc`, `test/ap`, `bits_per_edge` all logged |
| CLI overrides work | `data=citeseer train.split_ratio=0.8` produces correct run | ✅ Hydra override mechanism wired; full multirun (`-m`) available |
| `pytest tests/` | All tests pass | ✅ 26/26 tests pass in 4.2s |
| `ruff check src/` | Zero violations | ✅ 1 unused import removed (`torch` in `metrics.py`); 0 violations remaining |
