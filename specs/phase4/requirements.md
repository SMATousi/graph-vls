# Phase 4 — Requirements

## Functional Requirements

### FR-1: Jet dataset loading and graph construction
- Loads the Pythia8 quark/gluon jet dataset (assumed `energyflow.datasets.qg_jets` — confirm before implementation, `plan.md` Design Decision 1) and exposes it as a sequence of per-jet graphs
- Each jet is converted to a PyG-`Data`-compatible object: `x` (per-particle feature matrix), `edge_index` (k-NN graph in `(η, φ)` space, `k_graph = min(particle_count − 1, 8)`), `y` (binary quark=0/gluon=1 label, or vice versa — pick one convention and document it)
- Per-particle features: `(log pT, y, φ, one_hot(pdgid))`, `F ≈ 15` (Design Decision 6); must handle jets with varying particle counts `N` without padding or truncation (each jet keeps its own real `N`)
- Subsets the full dataset to a documented, tractable size (target 10,000–50,000 jets, Design Decision 9), balanced quark/gluon within a configurable tolerance
- **Amended 2026-07-30 (T4.10, `plan.md` Design Decision 12):** a second, dedicated `num_jets=800` configuration exists for literal comparability with the Lorentz-EQGNN literature baseline — additive to, not a replacement for, the 10,000–50,000-jet target above. Uses the same 70/15/15 split convention (`560`/`120`/`120`) as an explicit assumption, since the source table states only "800 (subset)" with no train/val/test breakdown to match against (flag per NFR-5 until/unless a more specific protocol is found).
- Produces a deterministic train/val/test split given a seed (target 70/15/15)

### FR-2: Fixed-`M` pooling for variable-`N` jets
- Every jet is pooled to the **same absolute** `M` regardless of its own particle count `N` (`plan.md` Design Decision 3) — `M` is a fixed hyperparameter of a training run, not derived per-jet
- Achieved by calling `LatentGraphPooling(latent_dim, num_clusters=M)` / `PooledGVLS` (`src/gvls/models/pooling.py`) **unmodified** — no new pooling code is required; only the caller fixes `M` as a constant instead of computing it from a ratio
- Must support `M ∈ {4, 6, 8}` for the jet-level compression sweep (FR-3)

### FR-3: Per-jet GVLS pretraining and compression sweep
- Trains `PooledGVLS` unsupervised (ELBO only, `src/gvls/losses/elbo.py` reused unchanged, plus `assignment_entropy`/`assignment_link_loss` reused unchanged from T3.6) by iterating jets one at a time through the encoder/pooling/latent-graph/message-passing stack (`plan.md` Design Decision 7), accumulating losses over a minibatch of jets before each optimizer step
- Must not leak information across jets in a minibatch: each jet's assignment matrix `S`, pooled Gaussian, `A_z`, and reconstruction are computed independently of every other jet in the same batch
- Sweeps `M ∈ {4, 6, 8}`, other hyperparameters (`hidden_dim`, `latent_dim d`, `k`, `graph_method`, `prior`, `mp_rounds`, `lr`, `beta`, `lambda_`) fixed to a reasonable starting config (carried over from Phase 2/3's citation-network findings, not re-tuned via NAS in this phase)
- Computes and persists, per `M`: average per-jet `reconstruction_f1`, average per-jet `bits_per_edge` (both from `src/gvls/eval/compression.py`, reused unchanged), plus raw counts (`M`, `d`, `k`, average `N` and `|E|` across jets)
- Selects a compression-optimal `M`: the smallest `M` whose average F1 is within a documented tolerance of the largest tested `M`'s F1
- Results written to `results/compression/qg_jets_pooling.csv`; logged to W&B under group tag `jet-compression-sweep`

### FR-4: QGNN ansatz construction
- `build_qgnn_circuit(M, d, num_layers=1) -> QuantumCircuit`: constructs an `M`-qubit circuit whose entangling gates are a direct function of the input `A_z` (`plan.md` Design Decision 2) — for every edge `(i,j)` with `A_z[i,j] > 0`, an `RZZ(θ · A_z[i,j])` gate is applied between qubits `i` and `j`; every qubit additionally gets an `RZ(b_i)` bias rotation and a data-encoding `RY`/`RX` rotation seeded from `z̃`'s per-node features
- `θ` and `b_i` (one scalar `θ` shared across edges within a layer, one `b_i` per qubit; both trainable) are the circuit's only trainable weights — `A_z` and `z̃` are runtime inputs, not weights
- With no edges present (`A_z` all-zero), the circuit must reduce to independent single-qubit rotations (no `RZZ` gates emitted) — a direct correctness check on the topology-equivariance claim
- Wrapped as a Qiskit Machine Learning `EstimatorQNN`, then wrapped again in a `TorchConnector` exposing a standard `forward(z_tilde, A_z) -> Tensor` (a single logit, or one logit per readout qubit) callable from a `torch.nn.Module`
- Must run on Qiskit Aer's noiseless statevector simulator (no real-hardware or noise-model execution this phase)

### FR-5: Two-stage supervised training
- Loads the frozen GVLS checkpoint selected by FR-3, runs it once (no gradient) over every jet in the labeled split to produce `(z̃, A_z)` per jet
- Trains only the QGNN's circuit parameters (`θ`, `b_i`, the readout rotation `γ_i`) via Adam, using `TorchConnector`'s autograd bridge (parameter-shift rule under the hood), against a BCE loss on the quark/gluon label
- **Amended 2026-07-30 (T4.10, `plan.md` Design Decision 12):** for the literature-comparability run specifically, the optimizer is `AdamW` (`torch.optim.AdamW`, was `Adam`), `lr=1e-3` (was `0.05`), `batch_size=16` (was `32`) — matching Lorentz-EQGNN's reported protocol exactly. `epochs=50` is unchanged (already matched). `BCEWithLogitsLoss` is kept (not replaced by a literal `nn.CrossEntropyLoss`) — documented as the mathematically equivalent formulation for a single-logit binary readout, per Design Decision 12's reconciliation. These become new Hydra-configurable defaults in `configs/train/qgnn_classifier.yaml`; the original `Adam, lr=0.05, batch_size=32` configuration remains available via override for the project's own (non-comparability) experiments if still needed.
- **Batches jets via a true batched `EstimatorQNN`/`TorchConnector` call per minibatch (amended 2026-07-27, T4.8), not a per-jet loop with manually accumulated gradients.** The original wording ("no true batched quantum circuit execution required — each jet's circuit still runs individually") is superseded: T4.5's real run proved intractably slow, and `TorchConnector`/`EstimatorQNN` were confirmed (against the installed `qiskit-machine-learning==0.8.2` source, not assumed) to already natively support a `(B, num_inputs)` batched call submitted as one job, which the original per-jet design simply never used. See `plan.md` Design Decision 10.
- Tracks train/val loss and accuracy per epoch; logs to W&B under group tag `qgnn-jet-classification`; checkpoints the best-val-accuracy parameter set

### FR-6: Evaluation and literature comparison
- Computes accuracy, AUC, and macro-F1 on the held-out test split
- Reports the qubit count (`M`) and circuit depth (`num_layers`) actually used alongside classification metrics
- Identifies and cites at least one literature QGNN (or closely related quantum-ML jet-tagging) result on this or a comparable dataset for direct comparison — if none exists, this must be stated explicitly rather than substituting an unrelated benchmark
- **Target identified 2026-07-30**: `Lorentz-EQGNN` (`sota-table.png` Table II, Quark-Gluon dataset, 4 qubits, `74.00% ± 0.26%` test accuracy) — see `plan.md` Design Decision 12. The source paper is not yet bibliographically confirmed (title/venue/arXiv ID unverified, table image only); do not present it as a confirmed citation until resolved (NFR-5).
- **Amended 2026-07-30 (T4.10):** also computes and reports **train accuracy** (in addition to test accuracy — the table reports both) and **wall-clock training time** and **inference time** (seconds), formatted to allow a direct row-by-row comparison against `sota-table.png`'s Table II. Per the honesty flag in `plan.md` Design Decision 12, wall-clock numbers must be reported as measured on our own hardware, not implied to be hardware-matched against the literature table.
- Results and comparison written to `results/qgnn/` and summarized in `README.md`, following the existing results-section convention (numbers table + findings bullets)

### FR-7: Joint fine-tuning ablation (stretch, T4.7)
- Optionally unfreezes GVLS and backpropagates the QGNN's classification loss (in addition to, or instead of, the ELBO) through both the quantum circuit and the classical encoder/pooling/latent-graph stack
- Reports test accuracy against the frozen-feature baseline (FR-5/FR-6) for direct comparison
- Only attempted after FR-1–FR-6 are complete and validated

---

## Non-Functional Requirements

### NFR-1: Reproducibility
- Fixed seed for dataset subsetting/splitting (FR-1), GVLS pretraining (FR-3), and QGNN training (FR-5)
- Same config + seed must reproduce the same compression-sweep CSV row and the same QGNN test-set metrics within floating-point / parameter-shift-rule tolerance

### NFR-2: Scale and compute budget
- Per-jet iteration (`plan.md` Design Decision 7) means training cost scales with the number of jets × epochs × (one classical forward pass + one quantum circuit execution each) — if this proves too slow at the target 10,000–50,000-jet subset, reducing the subset size is preferred over prematurely rewriting the classical stack for true batching (document whichever tradeoff is actually taken). **This tradeoff is about the classical GVLS stack specifically** (constrained by variable per-jet `N`, Design Decision 7); it does not apply to the QGNN stage, whose inputs are fixed-size post-pooling — see the QGNN-specific note below.
- Qiskit Aer statevector simulation cost scales as `O(2^M)` in qubit count `M`; since `M ≤ 8` here, this is not expected to be a bottleneck — flag immediately if it becomes one, since it would suggest the ansatz or simulator choice needs revisiting, not just the subset size
- Circuit depth (`num_layers`) must stay shallow enough (1–2 layers to start) that gradient estimation via parameter-shift doesn't dominate wall-clock time — parameter-shift requires 2 circuit evaluations per trainable parameter per sample, which grows with both `M` and `num_layers`
- **(Added 2026-07-27, T4.8) QGNN training must batch circuit evaluations across a minibatch, not loop per jet.** Unlike the classical stack, the QGNN's inputs (`z̃`, `A_z`) are already fixed-size (`M×d`, `M×M`) across every jet post-pooling, so there is no per-jet-`N` obstacle to batching here — the per-jet loop pattern in the original FR-5 wording was carried over from the classical stack's constraint without re-examining whether it applied. A per-jet loop pays Qiskit/`EstimatorQNN`'s Python-level job-dispatch overhead once per jet instead of once per minibatch. `input_gradients` on the `EstimatorQNN` must also be set to `False` — `z̃`/`A_z` are frozen extraction outputs that never need `∂L/∂input`, so computing them via parameter-shift is pure waste. **Measured impact was smaller than expected: ~1.34x combined, mostly from `input_gradients=False` (~1.24-1.25x) rather than batching itself (~1.07-1.08x)** — insufficient on its own at the target dataset scale, which motivated T4.9 below.
- **(Added 2026-07-27, T4.9) The default gradient estimator must not scale linearly with weight-parameter count.** `ParamShiftEstimatorGradient`'s cost is `2` circuit evaluations *per differentiated parameter, per gate occurrence* — for this ansatz, `theta` (shared across `m(m-1)/2` RZZ gates per layer) alone costs `2×edges` evaluations, giving `28` total at `m=4, num_layers=1` (not the naively-expected `2×9=18`) and `48` at `num_layers=2`. `qiskit_machine_learning.gradients.SPSAEstimatorGradient` costs a constant `2×spsa_batch_size` per sample regardless of parameter/gate count — `14x`/`24x` fewer evaluations at `num_layers=1`/`2` respectively (measured, `tests/test_qgnn.py`). `gradient_method="spsa"` is the default; `"param_shift"` remains selectable (`QGNNClassifier`, `train_qgnn_classifier`, `configs/train/qgnn_classifier.yaml`) since SPSA's stochastic, jointly-perturbed gradient estimate cannot substitute for exact per-parameter gradients in tests that check individual-parameter degeneracy (T4.4) or exact batching parity (T4.8) — those tests pin to `"param_shift"` explicitly. **Measured combined speedup (T4.8+T4.9) over the pre-T4.8 original: ~15.5x** on synthetic jets — not yet confirmed at the target real dataset scale.

### NFR-3: Test coverage
- Every new module (`jets.py`, `jet_sweep.py`, `qgnn.py`, `train_qgnn.py`) has at least one shape/correctness test
- `test_qgnn.py` must verify the core topology-equivariance claim directly: a toy `A_z` with a known edge set produces `RZZ` gates on exactly those qubit pairs and no others
- `test_jet_sweep.py` and `test_train_qgnn.py` use tiny smoke-test configurations (few jets, few epochs, small `M`) so the suite completes quickly
- **(Added 2026-07-27, T4.8)** Batched QGNN forward/backward must be verified against the pre-existing per-jet loop, not trusted from reading the `qiskit-machine-learning` source alone — this codebase has already found two non-obvious correctness bugs in this exact pinned stack (T4.4, `validation.md` V-4). A gradient-parity test (batched `loss.backward()` vs. summed per-jet gradients) is required before the batched path replaces the per-jet one, mirroring T4.2's `test_gradient_accumulation_matches_batched_mean` precedent

### NFR-4: Code style
- `ruff check src/` passes with zero warnings after each task
- New dataset code lives under `src/gvls/data/`; new quantum-model code lives under `src/gvls/models/qgnn.py`; new sweep code lives under `src/gvls/compression/`, mirroring T3.3/T3.6's existing module layout

### NFR-5: Honesty about unresolved items
- Do not present an assumed detail (dataset source, feature set, literature comparison number) as confirmed fact in `README.md` or `validation.md` until it has actually been checked — carry forward `plan.md`'s explicit "assumption, not yet confirmed" flags until they're resolved, consistent with how this project's other specs (e.g. `specs/phase3/plan.md`'s dataset-source and decoder-trigger flags) have handled open items
- **Added 2026-07-30 (T4.10):** the Lorentz-EQGNN literature table's source paper is not yet bibliographically confirmed (title/venue/arXiv ID unverified, table image only) — do not cite it as a confirmed reference until resolved. Wall-clock training/inference time comparisons against that table are directional only, not apples-to-apples, since the table's hardware is unknown; report our own measured numbers plainly without implying hardware parity.

---

## New Dependencies

- `qiskit` and `qiskit-machine-learning` (QGNN circuit construction, `EstimatorQNN`, `TorchConnector`) — new, not previously used anywhere in this codebase
- `energyflow` (assumed source for `qg_jets`; confirm before adding — Design Decision 1) — new
- No changes expected to existing dependencies (PyTorch, PyTorch Geometric, Hydra, W&B all carry over unchanged, per Design Decision 1's framework-integration rationale)
