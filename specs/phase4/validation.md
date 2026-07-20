# Phase 4 — Validation

**Status: T4.1–T4.3 complete (2026-07-20).** Remaining tasks (T4.4–T4.7) not started.

## Exit Criteria

- [x] Dataset source confirmed (`plan.md` Design Decision 1) — not an assumption anymore
- [x] Jet dataset loads, builds correct k-NN graphs, deterministic split (FR-1)
- [x] Fixed-`M` pooling confirmed working unmodified from T3.6's `PooledGVLS` (FR-2)
- [x] Per-jet GVLS pretraining sweep over `M ∈ {4,6,8}` complete, compression-optimal `M` selected (FR-3)
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

## V-2: Fixed-`M` Pooling for Jets (FR-2) ✅ Complete 2026-07-20

**File:** `src/gvls/compression/jet_sweep.py` (`train_pooled_gvls_on_jets`, `jet_loss`, `build_pooled_gvls`, `jet_adjacency`, `jet_pos_weight`). Tests: `tests/test_jet_sweep.py` (7 tests).

| Check | Pass condition | Result |
|---|---|---|
| `PooledGVLS` reused unmodified | No changes needed to `src/gvls/models/pooling.py` to support fixed absolute `M` per jet | ✅ Zero changes to `pooling.py` — `LatentGraphPooling(latent_dim, num_clusters=M)` already takes an absolute `M`, exactly as Design Decision 3 predicted |
| No cross-jet leakage | Per-jet assignment `S`/`A_z`/reconstruction independent of other jets in the same minibatch | ✅ Structurally guaranteed (each jet gets its own `model(x, edge_index)` call on its own dense tensors — no batched tensor ever spans two jets) and verified empirically: processing jet A, then a very-different-range jet B, then jet A again gives bit-identical output for jet A both times (`test_same_jet_gives_identical_output_regardless_of_other_jets_processed`, `test_disjoint_feature_ranges_do_not_mix`) |
| Gradient flow | Gradients reach encoder, pooling, and latent-graph-learner parameters after a minibatch of jets | ✅ Verified directly (`test_gradient_flows_to_all_submodules_from_one_jet`) — note the default `graph_method="attention"` latent-graph-learner has *zero* learnable parameters of its own (confirmed against `fgp`'s `log_tau` and `nri`'s MLP), so that check uses `fgp` instead; encoder/pooling gradient checks use the default config. Also verified the accumulation itself is numerically exact: summing `(loss/B).backward()` per jet across a 3-jet batch produces gradients identical to one `.backward()` on the batched mean loss (`test_gradient_accumulation_matches_batched_mean`) |

---

## V-3: Per-Jet Compression Sweep (FR-3) ✅ Complete 2026-07-20

**Files:** `src/gvls/compression/jet_sweep.py` (`evaluate_pooled_gvls_on_jets`, `select_compression_optimal_m`, `run_jet_compression_sweep`), `experiments/pretrain_gvls_jets.py`, `configs/train/jet_pretrain.yaml`, `configs/experiment/jet_pooling_sweep.yaml`. Tests: 6 new cases in `tests/test_jet_sweep.py` (19 total in that file, all synthetic — no network dependency in the suite).

**Real run:** 10,000 jets (`load_qg_jets(num_jets=10_000, seed=42)`), split 70/15/15 (train=7000, val=1500 used as the held-out eval set here, test=1500 untouched, reserved for T4.5/T4.6). Starting config (not NAS-tuned for jets, per plan.md T4.3): `hidden_dim=32, latent_dim=8, k=3, graph_method=attention, prior=isotropic, mp_rounds=1, lr=0.01, beta=0.001, epochs=30, batch_size=32`. `entropy_weight=0.1, aux_link_weight=5.0, f1_negative_ratio=1.0` (T3.6's DiffPool auxiliary-loss defaults, carried over unchanged). W&B group tag `jet-compression-sweep` (offline).

| Check | Pass condition | Result |
|---|---|---|
| `M=4` run complete | Average per-jet F1, bits-per-edge recorded | ✅ avg_f1=0.7447, avg_bpe=0.9513, avg_node_ratio=0.1118 (over 1500 eval jets) |
| `M=6` run complete | Same | ✅ avg_f1=0.7348, avg_bpe=0.9344, avg_node_ratio=0.1676 |
| `M=8` run complete | Same | ✅ avg_f1=0.7367, avg_bpe=0.9394, avg_node_ratio=0.2235 |
| Compression-optimal `M` selected | Smallest `M` within tolerance of best F1 | ✅ `M=4` (best F1 among the three; F1 is flat/non-monotonic across `M`, the same pattern T3.3/T3.6 found on Cora/CiteSeer/PubMed — larger `M` does not reliably buy more fidelity here either). `select_compression_optimal_m`'s `tolerance=0.02` default was used |
| Results persisted | `results/compression/qg_jets_pooling.csv` written | ✅ |

**Finding:** F1 sits at 0.73–0.74 across the whole `M` grid — comfortably above the known trivial-classifier floor (F1=2/3, `specs/phase3/validation.md` V-7) but well below any 0.90-style fidelity floor, consistent with every citation-network compression sweep to date never meeting one either. This appears to be a recurring property of this GVLS/pooling architecture rather than something jet-specific. Not re-tuned in this task (T4.3 explicitly reuses a Phase 2/3-derived starting config, not NAS) — worth revisiting if T4.5's downstream QGNN classification accuracy turns out to be bottlenecked by reconstruction fidelity rather than by the quantum stage.

**Correctness note (found during implementation):** small jets whose particle count `N` satisfies `k_graph_cap ≥ N − 1` (Design Decision 5) produce a *complete* k-NN graph — zero non-edges. `evaluate_pooled_gvls_on_jets`'s first pass over synthetic 8–12-particle test jets hung indefinitely: `gvls.eval.compression.eval_pairs_with_labels` (unchanged, existing utility) does rejection sampling for negative pairs, which cannot terminate if none exist. Fixed by clamping `num_negatives` to `n*(n-1)//2 - num_input_edges` and skipping F1 entirely for jets with zero non-edges, rather than assuming enough negatives exist (the citation-network sweeps never hit this because Cora/CiteSeer/PubMed are always sparse at their full size). No occurrences in the real 10,000-jet run (only affects unusually small jets relative to `k_graph_cap`), but the fix is real and needed for correctness at any subset size/composition.

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
