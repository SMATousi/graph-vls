# Phase 4 — Validation

**Status: T4.1–T4.4 complete (2026-07-20); T4.5 code-complete and smoke-tested, but not actually run (2026-07-20, user instruction).** T4.6/T4.7 not started.

## Exit Criteria

- [x] Dataset source confirmed (`plan.md` Design Decision 1) — not an assumption anymore
- [x] Jet dataset loads, builds correct k-NN graphs, deterministic split (FR-1)
- [x] Fixed-`M` pooling confirmed working unmodified from T3.6's `PooledGVLS` (FR-2)
- [x] Per-jet GVLS pretraining sweep over `M ∈ {4,6,8}` complete, compression-optimal `M` selected (FR-3)
- [x] QGNN ansatz built, topology-equivariance to `A_z` verified directly (FR-4)
- [x] Two-stage supervised training code complete, smoke-tested; best-val-accuracy checkpointing implemented but **not yet exercised on a real run** (FR-5, see V-5)
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

## V-4: QGNN Ansatz (FR-4) ✅ Complete 2026-07-20

**File:** `src/gvls/models/qgnn.py` (`build_qgnn_circuit`, `QGNNCircuitParams`, `sum_z_observable`, `QGNNClassifier`). Tests: 12 in `tests/test_qgnn.py`.

**Dependency conflict found and fixed (corrects an earlier, wrong call in this same section):** installing `qiskit>=2.0` originally bumped `numpy` to 2.4.6 in the `graph-vls` conda env; `pip` warned this conflicts with `energyflow`'s `wasserstein` sub-dependency (which pins `numpy<2.0`, since it ships a compiled C extension built against NumPy 1.x's ABI). At the time, `import energyflow` still succeeded locally (macOS arm64), so this was wrongly assessed as harmless ("`wasserstein` is never imported by `qg_jets.load`" — actually false: `energyflow/__init__.py` unconditionally imports `emd`, which imports `wasserstein`; it merely didn't crash on that specific platform's wheel). It **did** crash on a remote Linux machine: `AttributeError: _ARRAY_API not found` / `ImportError: numpy.core.multiarray failed to import` the moment `load_qg_jets` ran. Fixed by pinning `qiskit>=1.4,<2.0` (qiskit 1.x only requires `numpy>=1.17`, unlike qiskit 2.x's hard `numpy>=2.0` requirement) and `qiskit-machine-learning>=0.8,<0.9` (the compatible line for qiskit 1.x) and `numpy<2` explicitly in `pyproject.toml`, resolving to `qiskit-1.4.6`, `qiskit-machine-learning-0.8.2`, `numpy-1.26.4`, `qiskit-aer-0.17.2` (unchanged). All 201 tests, including `test_qgnn.py`/`test_qgnn_training.py`, still pass against this downgraded stack — the `EstimatorQNN`/`TorchConnector` API surface T4.4/T4.5 use is unaffected between qiskit-machine-learning 0.8.2 and 0.9.0.

| Check | Pass condition | Result |
|---|---|---|
| Qubit count correct | Circuit has exactly `M` qubits | ✅ `test_circuit_has_exactly_m_qubits`, `m ∈ {2,4,6,8}` |
| Topology equivariance | `RZZ` gates appear exactly on `A_z`'s edges, on a toy graph with a known edge set | ✅ Adapted (see design note below): rather than gate *objects* being absent for non-edges, every possible qubit pair has an always-present `RZZ(theta · A_z[i,j])` gate, and `test_rzz_angle_nonzero_exactly_on_real_edges` verifies the *bound angle* is nonzero exactly on real edges and exactly `0` elsewhere for a toy 4-qubit graph with 2 known edges |
| Zero-`A_z` reduction | No entangling gates emitted when `A_z` is all-zero | ✅ Adapted: `test_zero_a_z_all_rzz_angles_are_zero` confirms every bound `RZZ` angle is exactly `0`; `test_zero_a_z_reduces_to_no_entangling_reference_circuit` goes further and verifies the *statevector* is exactly equivalent (`atol=1e-10`) to a hand-built reference circuit with no `RZZ` instructions at all — since `RZZ(0)` is exactly the identity, this is a real functional equivalence, not just a structural coincidence |
| `TorchConnector` integration | Circuit callable as a `torch.nn.Module`, gradients flow to `θ`/`b_i` via `.backward()` | ✅ `test_gradients_flow_to_weight_params`, and `test_gradients_flow_with_multiple_layers` additionally confirms *every* individual weight (both layers' `theta`/`b_i`, plus the readout rotation) gets a nonzero gradient, not just some — the direct check for the readout-rotation fix below |
| Simulator | Runs on Qiskit Aer noiseless statevector simulation | ✅ `qiskit_aer.primitives.EstimatorV2`, `default_precision=0.0` (see finding below); `test_forward_is_deterministic_given_fixed_weights` confirms repeated calls with identical inputs are bit-identical (no shot noise) |

**Design note (adapts FR-4's literal wording):** `plan.md`/FR-4 describe `build_qgnn_circuit(M, d, num_layers)` and imply a circuit whose `RZZ` gates are structurally absent for non-edges. Implemented instead as **one fixed, maximal-topology circuit** built once per `(M, num_layers)`, with `RZZ` gates on *every* possible qubit pair and `A_z[i,j]` bound as a per-call **input** parameter (0 for non-edges). Reason: `TorchConnector` owns its trainable-weight tensor as a fresh `nn.Parameter` (`torch.tensor(initial_weights)`, which is not autograd-linked to whatever was passed in) every time it's constructed. Since jets have different `A_z` topologies, a "rebuild the circuit structurally per jet" design would force rebuilding `TorchConnector` per jet too — and `theta`/`b_i` could then never be one persistent, Adam-optimized parameter across the training loop without manually relaying gradients between successive throwaway `TorchConnector` instances, which is close to reimplementing part of what `TorchConnector` already does and contradicts `plan.md`'s explicit "no custom backward pass needs to be written" design intent. Since `RZZ(0)` is exactly the identity gate, the fixed-topology design is functionally identical to structural gate omission — verified directly by `test_zero_a_z_reduces_to_no_entangling_reference_circuit`.

**Bug found and fixed: a purely diagonal ansatz is untrainable.** `RZZ` and `RZ` are both diagonal gates in the computational basis, and any `Z`-basis measurement commutes exactly with a diagonal unitary applied beforehand (`U^† Z U = Z` for diagonal `U`). Building the ansatz exactly as `plan.md`/FR-4 literally describe it — `RY` data encoding, then only `RZZ`+`RZ`, then measure `sum(Z_i)` — produced a circuit where `theta`'s and `b_i`'s gradients were `~1e-16` (numerically zero) regardless of their actual values, confirmed both by inspecting `TorchConnector`'s reported gradient and by directly varying `theta` and observing the QNN's output did not change at all. Fixed by appending one final trainable, non-diagonal `RY(gamma_i)` rotation per qubit after all `num_layers`, restoring a genuine, nonzero gradient to *every* layer's `theta`/`b_i` (confirmed on a toy 2-qubit circuit: `theta`'s gradient went from `~1e-16` to `~0.12`, and changing `theta` from `0.5` to `0.9` changed the output from `1.593` to `1.654`). `gamma_i` is included as an additional `m` trainable weights.

**Bug found and fixed: `EstimatorQNN`'s default precision silently introduces shot noise.** Even with `estimator=AerEstimatorV2()` configured for exact evaluation (`Options(default_precision=0.0, ...)`), `EstimatorQNN`'s own `default_precision` argument (0.015625 unless overridden) triggers shot-based sampling — confirmed empirically: repeated identical calls returned slightly different values (e.g. `1.581, 1.605, 1.578, 1.587, 1.612`) until `default_precision=0.0` was passed explicitly to `EstimatorQNN`'s constructor, after which repeated calls were bit-identical. FR-4 requires Aer's noiseless statevector simulator, so this fix is necessary, not cosmetic.

**Observable choice:** sum of single-qubit `Z` operators across all `M` qubits (`sum_z_observable`), over a single designated readout qubit — so every pooled latent node contributes to the classification signal rather than one arbitrarily chosen qubit, consistent with this project's stance that all `M` pooled nodes matter equally. Not empirically compared against the single-qubit alternative yet (only decidable once T4.5 trains a real classifier) — documented as the chosen default per FR-4's explicit permission to pick one and record the choice.

---

## V-5: Two-Stage Supervised Training (FR-5) 🟡 Code complete, smoke-tested, not run

**Status:** implemented and unit-tested against synthetic jets/tiny models only. **No real training was executed on this machine — the user explicitly asked for bash scripts to run this on a remote machine instead.** Everything below reflects what the code does and what the smoke tests verify, not results from an actual pretraining/training run.

**Files:** `src/gvls/qgnn_training.py` (`extract_latent_features`, `qgnn_jet_loss`, `train_qgnn_classifier`, `evaluate_qgnn_classifier`, `save_qgnn_checkpoint`/`load_qgnn_checkpoint`), `src/gvls/eval/metrics.py` (`classification_metrics`, new), `src/gvls/compression/jet_sweep.py` (`save_gvls_checkpoint`/`load_gvls_checkpoint`, new — T4.3 never persisted a checkpoint, so T4.5 needed this added). New experiment scripts: `experiments/pretrain_gvls_jets_final.py` (trains+saves the one production GVLS checkpoint T4.3's sweep never produced), `experiments/train_qgnn.py` (T4.5 proper), `experiments/evaluate_qgnn.py` (T4.6's metrics half). New bash wrappers: `scripts/run_pretrain_gvls_jets_final.sh`, `scripts/run_train_qgnn.sh`, `scripts/run_evaluate_qgnn.sh`, `scripts/run_full_qgnn_pipeline.sh` (chains all three), `scripts/_activate_env.sh` (shared conda-activation helper, portable — finds conda via `conda info --base` rather than a hardcoded path). Tests: `tests/test_qgnn_training.py` (12 tests), `classification_metrics` tests added to `tests/test_metrics.py` (5 tests).

| Check | Pass condition | Result |
|---|---|---|
| GVLS frozen correctly | No gradient updates to GVLS parameters during QGNN training | ✅ (by construction, smoke-tested) `extract_latent_features` only ever calls the GVLS model under `torch.no_grad()`; no optimizer is ever constructed over its parameters in `train_qgnn_classifier`. `test_extract_latent_features_does_not_change_model_params` confirms every parameter is bit-identical (and `.grad is None`) after extraction |
| Training converges | Train/val loss decreases, no NaNs | Not evaluated — this requires a real run (deferred to the user's remote machine). Smoke test (`test_train_qgnn_classifier_smoke`, 2 epochs on 6 synthetic jets) only confirms the loop completes and losses are finite, not that they trend downward over a real training run |
| W&B logging | `qgnn-jet-classification` group tag present | Implemented in `experiments/train_qgnn.py` (`wandb.init(..., group="qgnn-jet-classification")`); not exercised against a real run |
| Checkpointing | Best-val-accuracy parameters saved | ✅ mechanism verified: `train_qgnn_classifier` tracks the highest validation accuracy seen across epochs and returns that epoch's state dict; `test_train_qgnn_classifier_best_state_dict_is_loadable` confirms a saved/reloaded checkpoint reproduces the exact same validation accuracy |

**Full classification metrics (beyond FR-5/FR-6's minimum):** `classification_metrics` (`src/gvls/eval/metrics.py`) returns accuracy, AUC, average precision, macro-F1, precision, recall, and the confusion matrix from one call — used both for `train_qgnn_classifier`'s per-epoch validation tracking (not just accuracy) and for `evaluate_qgnn.py`'s test-set report. 5 tests in `tests/test_metrics.py` cover perfect/inverted/random predictions, tensor inputs, and threshold sensitivity.

**tqdm progress bars:** added to every jet-pipeline training loop — `train_pooled_gvls_on_jets` (epoch-level, postfixed with running mean loss), `run_jet_compression_sweep`'s outer `M`-grid loop, and `train_qgnn_classifier` (epoch-level, postfixed with train loss and val accuracy). Not added to the pre-existing Phase 0–3 citation-network training scripts (`train_gvls.py`, etc.) — out of scope for this task, flagged in case broader coverage was intended.

**Gap found and fixed (2026-07-21): stage-1 GVLS pretraining logged nothing to W&B.** `experiments/pretrain_gvls_jets_final.py` originally only called `wandb.init(config=...)` (the run's hyperparameters) — there was no `wandb.log(...)` call anywhere in it or in `train_pooled_gvls_on_jets`, so a run's W&B page would show a populated config panel but an empty metrics/charts tab; per-epoch loss was only ever visible locally via the tqdm postfix. Fixed by adding optional `eval_jets`, `eval_every`, and `on_epoch_end` parameters to `train_pooled_gvls_on_jets` (all default to the prior no-logging behavior, so T4.3's sweep is unaffected): each epoch it now builds a `{"epoch", "train_loss", **val_* keys}` dict (val_* computed via the existing `evaluate_pooled_gvls_on_jets` every `eval_every` epochs, and always on the final epoch) and, if given, calls `on_epoch_end(epoch, metrics)` — the training function stays logging-backend-agnostic; the wiring to `wandb.log` lives entirely in `experiments/pretrain_gvls_jets_final.py`. 4 new tests in `tests/test_jet_sweep.py` cover: the callback fires once per epoch; val_* keys appear only on eval epochs (and always on the last epoch, regardless of `eval_every`); static/config fields (`num_clusters`, `latent_dim`, `k`, `num_features`, `dim_compression_ratio`) are excluded from per-epoch logging since they don't change; and omitting `eval_jets` produces no `val_*` keys and doesn't crash.

**What still needs to happen before this is genuinely "done":** run `scripts/run_full_qgnn_pipeline.sh` (or the three `run_*.sh` scripts individually) on a real machine; confirm loss actually decreases, val F1 is reasonable, and accuracy is better than a 50/50 random baseline; fill in the "Result" cells above with real numbers; confirm the W&B run pages for stage 1 (`jet-gvls-final` group) actually show live-updating charts, not just a static config panel.

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
