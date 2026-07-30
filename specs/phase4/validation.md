# Phase 4 — Validation

**Status: T4.1–T4.4 complete (2026-07-20); T4.5 code-complete and smoke-tested, but not actually run (2026-07-20, user instruction).** T4.6/T4.7 not started. **T4.8 (batched QGNN training, 2026-07-27) gave only ~1.34x — insufficient alone. T4.9 (SPSA gradient estimator, 2026-07-27) followed up and gave ~15.5x combined — by far the largest lever found, and the first one that plausibly makes T4.5's real run tractable at the target dataset scale, though not yet confirmed at that real scale. See V-8, V-9.** **T4.10 (new, 2026-07-30): literature comparison target confirmed (Lorentz-EQGNN, user-supplied table) and pipeline realignment implemented (AdamW/lr=1e-3/batch=16 defaults, optimizer selector, train-accuracy + wall-clock reporting, bash-script overrides) and tested — the actual 800-jet comparability run has not been executed yet. See V-10.**

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
- [x] (T4.8, 2026-07-27) QGNN training batched (one `EstimatorQNN`/`TorchConnector` call per minibatch, `input_gradients=False`), gradient-parity-tested against the per-jet loop, with an actual before/after wall-clock recorded — measured ~1.34x, not the large speedup originally hoped for; see V-8
- [x] (T4.9, 2026-07-27) SPSA gradient estimator replacing parameter-shift as the default, with the evaluation-count claim measured directly (not assumed) and degeneracy-sensitive tests pinned to param_shift — combined ~15.5x speedup measured on synthetic jets; see V-9
- [x] (T4.10, 2026-07-30) Literature comparison target confirmed and pipeline realigned: `configs/train/qgnn_classifier.yaml` updated to `AdamW, lr=1e-3, batch_size=16`; an `800`-jet comparability subset reachable via `data.num_jets=800`/`NUM_JETS` (bash script); `evaluate_qgnn.py` reports train/test accuracy and wall-clock training/inference time — implemented and unit-tested; the real 800-jet run itself has not been executed — see V-10

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
| W&B logging | `qgnn-jet-classification` group tag present, metrics logged **per epoch** during training, not just at the end | ✅ Fixed 2026-07-27 (see below); `wandb.init(..., group="qgnn-jet-classification")` present, and per-epoch `wandb.log(metrics, step=epoch)` verified live via a real 3-epoch smoke run (3 distinct logged points, `train_loss` trending 1.74→1.61→1.47) |
| Checkpointing | Best-val-accuracy parameters saved | ✅ mechanism verified: `train_qgnn_classifier` tracks the highest validation accuracy seen across epochs and returns that epoch's state dict; `test_train_qgnn_classifier_best_state_dict_is_loadable` confirms a saved/reloaded checkpoint reproduces the exact same validation accuracy |
| Artifact logging | Checkpoints (not just scalar metrics) reach W&B | ✅ Fixed 2026-07-27 (see below); all three stages now `wandb.log_artifact(...)` their output |

**Full classification metrics (beyond FR-5/FR-6's minimum):** `classification_metrics` (`src/gvls/eval/metrics.py`) returns accuracy, AUC, average precision, macro-F1, precision, recall, and the confusion matrix from one call — used both for `train_qgnn_classifier`'s per-epoch validation tracking (not just accuracy) and for `evaluate_qgnn.py`'s test-set report. 5 tests in `tests/test_metrics.py` cover perfect/inverted/random predictions, tensor inputs, and threshold sensitivity.

**tqdm progress bars:** added to every jet-pipeline training loop — `train_pooled_gvls_on_jets` (epoch-level, postfixed with running mean loss), `run_jet_compression_sweep`'s outer `M`-grid loop, and `train_qgnn_classifier` (epoch-level, postfixed with train loss and val accuracy). Not added to the pre-existing Phase 0–3 citation-network training scripts (`train_gvls.py`, etc.) — out of scope for this task, flagged in case broader coverage was intended.

**Gap found and fixed (2026-07-21): stage-1 GVLS pretraining logged nothing to W&B.** `experiments/pretrain_gvls_jets_final.py` originally only called `wandb.init(config=...)` (the run's hyperparameters) — there was no `wandb.log(...)` call anywhere in it or in `train_pooled_gvls_on_jets`, so a run's W&B page would show a populated config panel but an empty metrics/charts tab; per-epoch loss was only ever visible locally via the tqdm postfix. Fixed by adding optional `eval_jets`, `eval_every`, and `on_epoch_end` parameters to `train_pooled_gvls_on_jets` (all default to the prior no-logging behavior, so T4.3's sweep is unaffected): each epoch it now builds a `{"epoch", "train_loss", **val_* keys}` dict (val_* computed via the existing `evaluate_pooled_gvls_on_jets` every `eval_every` epochs, and always on the final epoch) and, if given, calls `on_epoch_end(epoch, metrics)` — the training function stays logging-backend-agnostic; the wiring to `wandb.log` lives entirely in `experiments/pretrain_gvls_jets_final.py`. 4 new tests in `tests/test_jet_sweep.py` cover: the callback fires once per epoch; val_* keys appear only on eval epochs (and always on the last epoch, regardless of `eval_every`); static/config fields (`num_clusters`, `latent_dim`, `k`, `num_features`, `dim_compression_ratio`) are excluded from per-epoch logging since they don't change; and omitting `eval_jets` produces no `val_*` keys and doesn't crash.

**Same gap found and fixed for stage 2 (QGNN training) — 2026-07-27.** `experiments/train_qgnn.py` had the equivalent problem, in a different shape: `train_qgnn_classifier` had no callback hook at all, so `wandb.log` was only ever called *after* the whole training loop returned (`for row in result.history: wandb.log(row)`), dumping every epoch's metrics at once instead of streaming them live. Consequences: no live progress visible in the W&B UI while a run is in flight, and — more seriously — a run that crashed or was killed partway (a real risk at this stage's current per-jet, unbatched training cost, see T4.8) would report **zero** training metrics, since nothing was logged until completion. Fixed the same way as stage 1: added an `on_epoch_end: Callable[[int, dict], None] | None` parameter to `train_qgnn_classifier` (`src/gvls/qgnn_training.py`), called once per epoch with that epoch's `{"epoch", "train_loss", **val_metrics}` row (the same dict already appended to `result.history` — verified identical via `test_train_qgnn_classifier_on_epoch_end_called_once_per_epoch`, not just structurally similar); `experiments/train_qgnn.py` now wires `on_epoch_end=lambda epoch, metrics: wandb.log(metrics, step=epoch)` and no longer has a post-hoc `for row in result.history` loop. Verified live (not just unit-tested) via a real 3-epoch smoke run: W&B's run history showed 3 distinct logged points with `train_loss` trending 1.74→1.61→1.47, confirming per-epoch streaming rather than a bulk end-of-run dump. 1 new test in `tests/test_qgnn_training.py`.

**Gap found and fixed (2026-07-27): checkpoints and eval results never reached W&B, only scalar metrics did.** All three stages logged per-epoch/summary metrics via `wandb.log`, but nothing ever called `wandb.log_artifact` — the actual GVLS checkpoint, QGNN checkpoint, and test-set results JSON only ever existed as local files, so "the best model" was never actually retrievable from a run's W&B page. Fixed: `experiments/pretrain_gvls_jets_final.py` and `experiments/train_qgnn.py` now save their checkpoint *before* `wandb.finish()` and log it as a `wandb.Artifact(type="model")` (aliased `"latest"`/`"best"` respectively, with metadata — final/best-epoch val metrics); `experiments/evaluate_qgnn.py` previously had no W&B integration at all (metrics were only printed and written to a local JSON) — it now also calls `wandb.init`/`wandb.log` for test metrics and logs the results JSON as a `wandb.Artifact(type="evaluation")`; `configs/qgnn_evaluate_config.yaml` gained a `wandb:` block and `scripts/run_evaluate_qgnn.sh` now forwards `--online` like the other two wrappers instead of silently dropping it. Verified live (not just unit-tested): ran all three stages end-to-end at tiny scale and confirmed via `strings` on each stage's offline `.wandb` binary log that an artifact record (`gvls-jets-m4`, `qgnn-jets-m4`, `qgnn-jets-m4-test-metrics`) was actually staged for upload, not just that the code executed without error.

**Gap found and fixed (2026-07-27): `run_full_qgnn_pipeline.sh` couldn't safely take hyperparameter overrides.** The script forwarded one shared `"$@"` argument list to all three stages; passing a stage-specific key (e.g. `train.gradient_method`, only valid for stage 2) hard-errored the moment it reached a stage whose Hydra config doesn't have it, and with `set -euo pipefail` that aborted the whole pipeline — sometimes only after already running earlier stages to completion (confirmed directly: `train.epochs=5` is valid in both stage 1 and 2's configs, so it silently succeeds through both, then hard-errors at stage 3, which has no `train:` group at all). Fixed per explicit user preference for a zero-argument, edit-the-file workflow: every hyperparameter for all three stages is now a shell variable declared at the top of `run_full_qgnn_pipeline.sh`, mirroring `configs/*.yaml`'s defaults; each stage is invoked with its own explicitly-built argument list containing only the keys that stage's config actually recognizes, closing the cross-stage error off structurally rather than documenting around it. Also fixed a latent macOS-bash-3.2 bug hit while implementing this (`"${empty_array[@]}"` throws "unbound variable" under `set -u`, fixed only in bash 4.4+) by always building one non-empty combined array per stage. Verified end-to-end by actually running the rewritten script at small scale, including deliberately setting `gradient_method=param_shift` (the exact key that broke the old design) and confirming it reached only stage 2.

**What still needs to happen before this is genuinely "done":** run `scripts/run_full_qgnn_pipeline.sh` on a real machine at the target dataset scale (10,000–50,000 jets); confirm loss actually decreases, val F1 is reasonable, and accuracy is better than a 50/50 random baseline; fill in the "Result" cells above with real numbers; confirm the W&B run pages for both stages show live-updating charts *and* a retrievable model artifact, not just a static config panel. T4.9's SPSA gradient estimator (not T4.8's batching alone) is what's now expected to make this tractable at scale — see V-9 — but that expectation itself is still based on a small synthetic-jet benchmark, not a real run.

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

---

## V-8: Batched QGNN Training (T4.8) ✅ Implemented 2026-07-27 — modest speedup, honestly reported below

**Trigger:** T4.5's real run (deferred to a remote machine per V-5) was reported intractably slow. Root-caused to `train_qgnn_classifier`'s per-jet `EstimatorQNN`/`TorchConnector` calls — one Estimator job dispatch per jet — compounded by `input_gradients=True` differentiating all 19 circuit parameters (9 trainable weights + 10 runtime inputs) via parameter-shift when only the 9 weights are ever optimized. Full diagnosis and design in `plan.md` Design Decision 10; requirements in `requirements.md` FR-5 (amended) and NFR-2/NFR-3 (amended).

**Verified before speccing (not assumed):** read the installed `qiskit-machine-learning==0.8.2` source directly. `TorchConnector`'s `_TorchNNFunction.forward`/`backward` (`connectors/torch_connector.py`) already accept a batched `(B, num_inputs)` tensor with a single shared weight vector. `EstimatorQNN._forward`/`_backward` (`neural_networks/estimator_qnn.py`) already tile per-sample parameter values into one `estimator.run()`/`gradient.run()` call regardless of `num_samples`. Batching is natively supported by the pinned library version — it was simply never used, because the per-jet loop pattern was copied from the classical stack's T4.2 precedent (Design Decision 7) without checking whether that constraint (variable per-jet `N`) actually applied to the QGNN, which operates on fixed-size `(M,d)`/`(M,M)` inputs post-pooling and has no such constraint.

**Files:** `src/gvls/models/qgnn.py` (`QGNNClassifier.encode_input_batch`, dim-dispatching `forward`, `input_gradients=False`), `src/gvls/qgnn_training.py` (`collate_jet_features`, `qgnn_batch_loss`, batched `train_qgnn_classifier` inner loop, chunked-batch `evaluate_qgnn_classifier`). Tests: 5 new in `tests/test_qgnn.py`, 3 new in `tests/test_qgnn_training.py`.

| Check | Pass condition | Result |
|---|---|---|
| `QGNNClassifier.forward` accepts a batch | 3D `(B,M,d)`/`(B,M,M)` input routes through one `TorchConnector` call, not a Python loop over jets | ✅ dispatches on `z_tilde.dim()` |
| Batched forward matches per-jet loop | `test_forward_batch_matches_per_jet_loop`: batched logits equal per-jet-computed-and-stacked logits (float tolerance) | ✅ (also `test_batch_of_one_matches_single_jet_call` for the ragged-tail case) |
| Batched backward matches per-jet loop | `test_batched_backward_matches_accumulated_per_jet_gradients`: batched `loss.backward()` gradients on `theta`/`b_i`/`gamma_i` equal the sum of the pre-T4.8 per-jet-accumulated gradients — required before trusting the batched path over the known-correct one, given V-4's precedent of non-obvious bugs in this exact stack | ✅ `atol=1e-5` |
| `input_gradients=False` set | `EstimatorQNN` in `QGNNClassifier.__init__` no longer computes unused input-parameter gradients | ✅ |
| Ragged final minibatch handled | `collate_jet_features` with `len(features) % batch_size != 0` doesn't error | ✅ `test_collate_jet_features_handles_ragged_final_batch`, `test_train_qgnn_classifier_handles_ragged_final_minibatch` |
| Existing smoke test still passes | `test_train_qgnn_classifier_smoke` unaffected by the loop rewrite | ✅ |
| Full suite / lint | `pytest tests/` (212 tests) and `ruff check` on all touched files | ✅ |

**Real wall-clock measurement (not assumed) — the honest result is smaller than hoped.** Benchmarked the pre-T4.8 code (loaded directly from commit `31b5afe` via `git show`, not reconstructed from memory) against the current implementation on identical synthetic jets (`M=4, d=8, num_layers=1`, 300 train / 60 val jets, 3 epochs, CPU), decomposed into the two independent levers:

| Variant | Wall-clock | vs. original |
|---|---|---|
| (a) original: per-jet loop, `input_gradients=True` | 26.3–26.6s | 1.00x (baseline) |
| (b) per-jet loop, `input_gradients=False` only | 21.2–21.3s | 1.24–1.25x |
| (c) T4.8 (batched, `input_gradients=False`) | 19.7–19.8s | 1.34x |

Batching's *own* marginal contribution on top of `input_gradients=False` is only ~1.07–1.08x, tested at both `batch_size=16` and `batch_size=64` (no further improvement at the larger batch size — the win does not scale with batch size, i.e. it's saturated). **Most of the total 1.34x speedup comes from `input_gradients=False`, not from batching itself.** This contradicts the original diagnosis's expectation that per-jet Python/job-dispatch overhead was the dominant cost. The actual dominant cost appears to be intrinsic to `qiskit-machine-learning`'s parameter-shift gradient estimator (`ParamShiftEstimatorGradient`, invoked automatically per the "No gradient function provided, creating a gradient function" warning) constructing and evaluating one bound circuit per shifted parameter per sample — work our outer-level batching doesn't touch, since it happens *inside* the single `gradient.run()` call regardless of how many samples are packed into it.

**Implication:** T4.8 is implemented correctly (parity-tested) and gives a real, modest ~1.3x speedup, primarily from `input_gradients=False`. It is very unlikely to be enough on its own to make the target 10,000–50,000-jet × 50-epoch real run in `specs/phase4/plan.md` Design Decision 9 tractable — extrapolating the 1.34x from this benchmark's scale, a run that was intractable before is very likely still intractable after, just ~25% faster. Further levers (not implemented here, flagged for a follow-up decision): reducing `num_layers`/circuit depth further, reducing the jet subset size, switching to a cheaper gradient method (e.g. SPSA instead of exact parameter-shift, trading gradient variance for far fewer circuit evaluations per step), or reducing epoch count. This should be surfaced to the user before assuming T4.5's real run is now unblocked.

---

## V-9: SPSA Gradient Estimator (T4.9) ✅ Implemented 2026-07-27 — the large speedup T4.8 didn't deliver

**Trigger:** V-8's own honest finding — batching alone was insufficient (~1.07-1.08x), and the actual dominant cost was intrinsic to `ParamShiftEstimatorGradient`'s circuit-evaluation count, not per-jet dispatch overhead. A real fix required reducing that count, not just how it's dispatched. Full diagnosis and design in `plan.md` Design Decision 11; requirements in `requirements.md` NFR-2 (amended).

**Verified before/while implementing (not assumed):** read `qiskit_machine_learning.gradients.spsa.spsa_estimator_gradient.SPSAEstimatorGradient._run` directly. It perturbs every differentiated parameter jointly along one random ±1 direction and computes a single two-point finite difference per sample — `2 × spsa_batch_size` circuit evaluations, independent of parameter count. Also verified `AerEstimatorV2`'s own `Options.default_precision` defaults to `0.0` (exact) at the class level, distinct from `EstimatorQNN`'s separate `default_precision=0.015625` default (the thing V-4 had to override) — so sharing one `AerEstimatorV2()` instance between `EstimatorQNN` and `SPSAEstimatorGradient` keeps SPSA's gradient computation exact/noiseless automatically, without needing a second precision override.

**Files:** `src/gvls/models/qgnn.py` (`QGNNClassifier.__init__` gains `gradient_method`/`spsa_epsilon`/`spsa_batch_size`), `src/gvls/qgnn_training.py` (`train_qgnn_classifier`, `load_qgnn_checkpoint` plumb the same through), `experiments/train_qgnn.py`, `configs/train/qgnn_classifier.yaml`. Tests: 9 new in `tests/test_qgnn.py`; 2 existing tests (`test_gradients_flow_to_weight_params`, `test_gradients_flow_with_multiple_layers`) and T4.8's `test_batched_backward_matches_accumulated_per_jet_gradients` amended to pin `gradient_method="param_shift"` explicitly (see below for why).

| Check | Pass condition | Result |
|---|---|---|
| SPSA is the new default | `QGNNClassifier(...).gradient_method == "spsa"`, wired to `SPSAEstimatorGradient` | ✅ `test_spsa_is_the_default_gradient_method` |
| `param_shift` still selectable | `gradient_method="param_shift"` wires to `ParamShiftEstimatorGradient` | ✅ `test_param_shift_still_selectable` |
| Invalid method rejected | Unknown `gradient_method` raises `ValueError` | ✅ `test_invalid_gradient_method_raises` |
| Evaluation-count claim, measured not assumed | Direct pub-count via a monkeypatched `AerEstimatorV2.run` during one `.backward()` call | ✅ `test_spsa_evaluation_count_is_constant_independent_of_weight_count`: param-shift needs **28** evaluations at `m=4, num_layers=1` — not the naively-expected `2×9=18` (see next row) — SPSA needs a constant **2** |
| Why 28, not 18 | `theta` is shared across all `m(m-1)/2` RZZ gates per layer; the shift rule needs a shifted pair per *occurrence*, not per parameter | `theta` alone costs `2×6=12` of the 28; `bias`/`readout` (one gate each) cost the expected `2` each (`4×2 + 4×2 = 16`); `12+16=28` |
| Gap widens with `num_layers` | SPSA's cost stays flat; param-shift's grows | ✅ `test_spsa_evaluation_count_scales_with_num_layers_for_param_shift_only`: `48` vs. constant `2` at `num_layers=2` (measured, not derived) |
| Determinism given fixed seed (NFR-1) | Same seed -> same stochastic gradient estimate | ✅ `test_spsa_gradient_deterministic_given_fixed_seed` |
| Gradients still flow | SPSA backward produces a nonzero gradient | ✅ `test_spsa_gradients_flow_and_are_nonzero` |
| Degeneracy tests preserve their regression value | T4.4's diagonal-ansatz tests must not silently pass under SPSA regardless of whether the bug recurs | ✅ pinned to `gradient_method="param_shift"` explicitly — SPSA's jointly-perturbed estimate gives every differentiated parameter the same magnitude (`±diff`), so it cannot isolate an individual parameter's own zero gradient the way these tests require |
| T4.8's gradient-parity test preserved | Exact batched-vs-per-jet equality check must stay exact | ✅ pinned to `gradient_method="param_shift"` — SPSA's stochastic estimate makes "batched == accumulated per-jet" a fuzzier, RNG-draw-order-dependent claim rather than a clean invariant, even though empirically it still held under the default before this pin was added |
| Config plumbing | Hydra composes `gradient_method`/`spsa_epsilon`/`spsa_batch_size`; checkpoint records `gradient_method` for provenance | ✅ `python experiments/train_qgnn.py --cfg job --resolve` verified directly |
| Full suite / lint | `pytest tests/` and `ruff check` on all touched files | ✅ |

**Real wall-clock measurement (not assumed) — this is the large win the diagnosis originally hoped T4.8 would be.** Same benchmark methodology as V-8 (pre-T4.8 original loaded from commit `31b5afe` via `git show`, identical synthetic jets, `M=4, d=8, num_layers=1`, 300 train / 60 val jets, 3 epochs, `batch_size=16`, CPU):

| Variant | Wall-clock | vs. original |
|---|---|---|
| Original (pre-T4.8): per-jet loop, `param_shift`, `input_gradients=True` | ~26.1s | 1.00x (baseline) |
| T4.8 only: batched, `param_shift`, `input_gradients=False` | ~19.0s | 1.37x |
| T4.9 (this): batched, `spsa`, `input_gradients=False` | ~1.6–1.7s | **~15.5x** |

SPSA alone (batched param-shift -> batched SPSA, holding batching constant) contributes **~11.7x** on top of T4.8's own ~1.34-1.37x. This is by far the largest single lever found across T4.8/T4.9, consistent with the evaluation-count measurement above (14-24x fewer circuit evaluations per sample) modulo Python-level overhead that doesn't scale down as cleanly as the evaluation count itself.

**Implication:** unlike T4.8, this is a plausible real fix for T4.5's target-scale tractability, not just an honest-but-modest improvement. **Caveat, stated as plainly as V-8's was:** this is still a synthetic-jet CPU benchmark at a much smaller scale (300 jets, 3 epochs) than the target real run (10,000-50,000 jets, up to 50 epochs) — directionally strong evidence, not a substitute for actually running T4.5 at real scale. The stochastic-gradient tradeoff (SPSA's variance vs. param-shift's exactness) also hasn't been evaluated for its effect on classification accuracy/convergence, only on speed and gradient-flow sanity checks — if training doesn't converge well under SPSA's noisier gradient at the target scale, `spsa_batch_size` can be increased (still far cheaper than param-shift for any batch size well below `num_weight_params`) or `gradient_method="param_shift"` can be restored via one config override.

---

## V-10: Literature Comparison Target & Pipeline Realignment (T4.10) 🔶 Pipeline implemented and tested; comparability run not yet executed

**Trigger:** user supplied a benchmark table (`sota-table.png`, Table II "Quark-Gluon Dataset" / Table III "Electron-Photon Dataset", 2026-07-30) reporting QGNN/hybrid-QNN/classical-CNN accuracy on jet-classification tasks. Resolves T4.6's previously-unidentified literature-comparison target (Design Decision 4 / FR-6). Full design reasoning in `plan.md` Design Decision 12; requirement amendments in `requirements.md` FR-1/FR-5/FR-6.

**Comparison target:** `Lorentz-EQGNN`, Table II, Quark-Gluon dataset — `4` qubits (matches our `M=4`), `AdamW`, `lr=1e-3`, `batch_size=16`, `50` epochs, `Cross Entropy Loss`, `800`-jet training subset, `74.00% ± 0.26%` test accuracy. **Not yet bibliographically confirmed** — the source paper's title/venue/arXiv ID have not been verified, only the supplied table image; do not present as a confirmed citation in `README.md` until resolved (NFR-5).

**User decisions (`AskUserQuestion`, 2026-07-30):**

| Question | Decision |
|---|---|
| Alignment target | Match Lorentz-EQGNN exactly, not the table's broader conventions or a no-change citation-only approach |
| Dataset scope | Quark-Gluon only; Electron-Photon (Table III) explicitly deferred |
| Training subset size | Match literally: `800` jets, as a dedicated comparability run (additive to, not replacing, the 10,000–50,000-jet target) |
| Metrics format | Add train/test accuracy and wall-clock training/inference time reporting, alongside existing accuracy/AUC/macro-F1 |

**Implemented 2026-07-30.** Files: `src/gvls/qgnn_training.py` (`_build_optimizer`, `_OPTIMIZERS = {"adam": Adam, "adamw": AdamW}`, `train_qgnn_classifier` gains `optimizer: str = "adamw"`; `QGNNTrainingResult` gains `best_train_metrics`, computed once on the best-val-accuracy weights via the existing `evaluate_qgnn_classifier`); `configs/train/qgnn_classifier.yaml` (`optimizer: adamw`, `lr: 0.001`, `batch_size: 16`, `epochs: 50` unchanged); `experiments/train_qgnn.py` (times the `train_qgnn_classifier` call via `time.perf_counter()`, persists `optimizer`/`train_accuracy`/`training_time_s` into the QGNN checkpoint's saved config and W&B); `experiments/evaluate_qgnn.py` (reads those three back via `qgnn_config.get(...)` with `None` fallback for pre-T4.10 checkpoints, times its own inference pass, prints/persists/logs all of it, plus the NFR-5 hardware-non-parity note); `scripts/run_full_qgnn_pipeline.sh` (`QGNN_OPTIMIZER` variable added to Stage 2, `QGNN_LR`/`QGNN_BATCH_SIZE` defaults updated to `0.001`/`16` to mirror the new yaml defaults, `NUM_JETS` comment documents setting `800` for the literal comparability run). Tests: 4 new in `tests/test_qgnn_training.py`.

| Check | Pass condition | Result |
|---|---|---|
| Comparability config exists | `800`-jet subset reachable via the existing `data.num_jets` Hydra key / `NUM_JETS` bash variable — no new config file needed since this was already overridable | ✅ verified via `python experiments/train_qgnn.py --cfg job --resolve data.num_jets=800` |
| Optimizer/hyperparameters realigned | `configs/train/qgnn_classifier.yaml` uses `AdamW, lr=1e-3, batch_size=16, epochs=50` by default; `train_qgnn_classifier` gains an optimizer selector (`optimizer="adamw"` default, `"adam"` selectable, invalid names raise `ValueError`) | ✅ `test_train_qgnn_classifier_default_optimizer_is_adamw`, `test_train_qgnn_classifier_supports_adam`, `test_train_qgnn_classifier_invalid_optimizer_raises`; config verified via `--cfg job --resolve` both with and without an override back to `adam/0.05/32` |
| Loss-function equivalence documented | `BCEWithLogitsLoss` kept on the single-logit readout (no code change — already the default); the CE-equivalence reasoning is recorded in `plan.md` Design Decision 12, referenced from `evaluate_qgnn.py`'s module docstring | ✅ |
| Train/test accuracy reported | `evaluate_qgnn.py` reports both — `train_accuracy` read back from the checkpoint, `test_accuracy` from the fresh test-split evaluation | ✅ `test_train_qgnn_classifier_result_includes_best_train_metrics`; `evaluate_qgnn.py`'s `results` dict includes both keys |
| Wall-clock timing reported | Training time (`experiments/train_qgnn.py`) and inference time (`experiments/evaluate_qgnn.py`) in seconds, printed/logged/persisted, with the NFR-5 hardware-non-parity caveat printed alongside | ✅ code-complete; not yet exercised on a real (non-synthetic) run — see below |
| Real comparability run executed | `800`-jet run actually completed (not just smoke-tested) with real numbers reported against Table II | ⬜ not yet run |
| README updated | New results section includes a direct row added to (or placed alongside) `sota-table.png`'s Table II format | ⬜ not yet done — depends on the real run above |
| Full suite / lint | `pytest tests/` and `ruff check` on all touched files | ✅ 223/223 passing; `ruff check src/gvls/qgnn_training.py experiments/train_qgnn.py experiments/evaluate_qgnn.py tests/test_qgnn_training.py` clean (3 pre-existing import-order violations in unrelated files — `compression_sweep.py`, `smoke_test.py`, `train_gvls.py` — untouched by this task) |

**What still needs to happen before this is genuinely "done":** run `scripts/run_full_qgnn_pipeline.sh` with `NUM_JETS=800` on a real machine, confirm the reported `train_accuracy`/`test_accuracy`/`training_time_s`/`inference_time_s` are real (not synthetic-smoke-test) numbers, and add the resulting row to `README.md` alongside `sota-table.png`'s Table II. This mirrors V-5's own "code-complete, not run" status for the larger-scale pipeline — the two comparability configurations (10,000–50,000 jets vs. this 800-jet literature-comparability run) share the same underlying code path and both await an actual execution.
