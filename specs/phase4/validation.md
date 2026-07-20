# Phase 4 — Validation

**Status: T4.1 complete (2026-07-20).** Remaining tasks (T4.2–T4.7) not started.

## Exit Criteria

- [x] Dataset source confirmed (`plan.md` Design Decision 1) — not an assumption anymore
- [x] Jet dataset loads, builds correct k-NN graphs, deterministic split (FR-1)
- [ ] Fixed-`M` pooling confirmed working unmodified from T3.6's `PooledGVLS` (FR-2)
- [ ] Per-jet GVLS pretraining sweep over `M ∈ {4,6,8}` complete, compression-optimal `M` selected (FR-3)
- [ ] QGNN ansatz built, topology-equivariance to `A_z` verified directly (FR-4)
- [ ] Two-stage supervised training complete, best-val-accuracy checkpoint saved (FR-5)
- [ ] Test-set accuracy/AUC/macro-F1 reported, literature comparison identified or explicitly declared absent (FR-6)
- [ ] `README.md` updated with a new results section
- [ ] `pytest tests/` passes with all new Phase 4 tests included

---

## V-1: Jet Dataset & Graph Construction (FR-1) ✅ Complete 2026-07-20

**File:** `src/gvls/data/jets.py`. Tests: `tests/test_jets.py` (15 tests, all synthetic-jet unit tests — no network call in the suite itself, matching the existing precedent that `load_planetoid`/`load_tu_dataset` aren't exercised in `tests/` either; `load_qg_jets`'s actual `energyflow` download path was verified manually, see below).

| Check | Pass condition | Result |
|---|---|---|
| Dataset source confirmed | User has confirmed `energyflow.qg_jets` (or named an alternative) as the actual data source | ✅ User confirmed `energyflow.qg_jets` via `AskUserQuestion` (2026-07-20) |
| Jets load without error | A sample of jets returns valid `(x, edge_index, y)` triples | ✅ `load_qg_jets(num_jets=2000)` against real cached data: loads in ~1.3s (single 100k-jet `QG_jets.npz` file, already covers any subset up to 100k without further downloads), particle counts range 9–105, `x`/`edge_index`/`y` all well-formed |
| k-NN graph correctness | No self-loops, symmetric, edge count ≤ `k_graph` per node | Adjusted: union-symmetrized (not mutual-intersection) k-NN, the same tradeoff Phase 1 made for the latent-graph learner (`specs/phase1/plan.md`) to avoid emptying/disconnecting small graphs — this means per-node degree can slightly *exceed* `k_graph_cap` (a popular neighbor gets picked by more nodes than its own quota) rather than being strictly capped. No self-loops and full symmetry verified directly; degree bounded (tested ≤ `4×k_graph_cap`, empirically far tighter) rather than hard-capped at `k_graph_cap`. Periodic φ handled explicitly (`Δφ` wrapped to `[-π, π]`) and verified with a boundary-crossing test case. |
| Feature shape | `x.shape == (N, F)` with `F` matching Design Decision 6 | `F = NUM_FEATURES = 18`: `log(pT), y, φ` (3) + one-hot(pdgid) over 14 observed species + 1 "unknown" bucket (15) — higher than the plan's rough estimate of `F ≈ 15` because the dataset actually contains 14 distinct species (charge-separated), not ~11, plus the added unknown-species bucket for robustness at this system boundary |
| Split determinism | Same seed → identical train/val/test split | ✅ `split_jets` uses a seeded `torch.Generator`, mirroring `split_edges`'s convention; verified identical output across two calls with the same seed, different output across different seeds |
| Label balance | Quark/gluon ratio within tolerance on the subset | ✅ Exact 50/50 by construction — `load_qg_jets` samples `num_jets/2` from each label's pool rather than approximating a tolerance band (raw dataset is already ~49.9/50.1 so this required no aggressive oversampling; `raw_multiplier=1.3` cushion is ample) |

---

## V-2: Fixed-`M` Pooling for Jets (FR-2) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| `PooledGVLS` reused unmodified | No changes needed to `src/gvls/models/pooling.py` to support fixed absolute `M` per jet | ⬜ |
| No cross-jet leakage | Per-jet assignment `S`/`A_z`/reconstruction independent of other jets in the same minibatch | ⬜ |
| Gradient flow | Gradients reach encoder, pooling, and latent-graph-learner parameters after a minibatch of jets | ⬜ |

---

## V-3: Per-Jet Compression Sweep (FR-3) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| `M=4` run complete | Average per-jet F1, bits-per-edge recorded | ⬜ |
| `M=6` run complete | Same | ⬜ |
| `M=8` run complete | Same | ⬜ |
| Compression-optimal `M` selected | Smallest `M` within tolerance of best F1 | ⬜ |
| Results persisted | `results/compression/qg_jets_pooling.csv` written | ⬜ |

---

## V-4: QGNN Ansatz (FR-4) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| Qubit count correct | Circuit has exactly `M` qubits | ⬜ |
| Topology equivariance | `RZZ` gates appear exactly on `A_z`'s edges, on a toy graph with a known edge set | ⬜ |
| Zero-`A_z` reduction | No entangling gates emitted when `A_z` is all-zero | ⬜ |
| `TorchConnector` integration | Circuit callable as a `torch.nn.Module`, gradients flow to `θ`/`b_i` via `.backward()` | ⬜ |
| Simulator | Runs on Qiskit Aer noiseless statevector simulation | ⬜ |

---

## V-5: Two-Stage Supervised Training (FR-5) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| GVLS frozen correctly | No gradient updates to GVLS parameters during QGNN training | ⬜ |
| Training converges | Train/val loss decreases, no NaNs | ⬜ |
| W&B logging | `qgnn-jet-classification` group tag present | ⬜ |
| Checkpointing | Best-val-accuracy parameters saved | ⬜ |

---

## V-6: Evaluation and Literature Comparison (FR-6) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| Test metrics reported | Accuracy, AUC, macro-F1 on held-out test jets | ⬜ |
| Qubit/depth reported | `M` and `num_layers` stated alongside accuracy | ⬜ |
| Literature comparison | A specific, cited published QGNN (or closely related) result on this/a comparable dataset, or an explicit statement that none was found | ⬜ |

---

## V-7: Code Quality ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| `pytest tests/` | All new Phase 4 tests pass alongside the existing suite | ⬜ |
| `ruff check src/` | Zero violations | ⬜ |
