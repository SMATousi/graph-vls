# Phase 4 — Validation

**Status: not started (spec only, written 2026-07-14).** No code for this phase exists yet. This file will be filled in exactly as `specs/phase0/validation.md`–`specs/phase3/validation.md` were: one `V-n` section per task, with a Check/Pass-condition/Result table populated as work happens, plus write-ups for any bugs or surprises found along the way (this project's specs consistently treat those as first-class results, not footnotes — see e.g. `specs/phase3/validation.md` V-7/V-8).

## Exit Criteria

- [ ] Dataset source confirmed (`plan.md` Design Decision 1) — not an assumption anymore
- [ ] Jet dataset loads, builds correct k-NN graphs, deterministic split (FR-1)
- [ ] Fixed-`M` pooling confirmed working unmodified from T3.6's `PooledGVLS` (FR-2)
- [ ] Per-jet GVLS pretraining sweep over `M ∈ {4,6,8}` complete, compression-optimal `M` selected (FR-3)
- [ ] QGNN ansatz built, topology-equivariance to `A_z` verified directly (FR-4)
- [ ] Two-stage supervised training complete, best-val-accuracy checkpoint saved (FR-5)
- [ ] Test-set accuracy/AUC/macro-F1 reported, literature comparison identified or explicitly declared absent (FR-6)
- [ ] `README.md` updated with a new results section
- [ ] `pytest tests/` passes with all new Phase 4 tests included

---

## V-1: Jet Dataset & Graph Construction (FR-1) ⬜ Not started

| Check | Pass condition | Result |
|---|---|---|
| Dataset source confirmed | User has confirmed `energyflow.qg_jets` (or named an alternative) as the actual data source | ⬜ |
| Jets load without error | A sample of jets returns valid `(x, edge_index, y)` triples | ⬜ |
| k-NN graph correctness | No self-loops, symmetric, edge count ≤ `k_graph` per node | ⬜ |
| Feature shape | `x.shape == (N, F)` with `F` matching Design Decision 6 | ⬜ |
| Split determinism | Same seed → identical train/val/test split | ⬜ |
| Label balance | Quark/gluon ratio within tolerance on the subset | ⬜ |

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
