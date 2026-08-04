# Phase 4 — Plan

## Objective

Build the classical→quantum pipeline `mission.md` and `reports/midterm_report.md` §6 have anticipated since the project's start: train GVLS to compress a graph into (z̃, A_z), then feed that compressed representation into a **Quantum Graph Neural Network (QGNN)** that performs a real downstream task. This phase is the first time that downstream task is a QGNN rather than a classical probe, and the first time the benchmark is **particle-jet graph classification** rather than a citation network.

**Benchmark:** Pythia8-generated quark-vs-gluon jet classification — binary classification of whether a jet (a spray of collimated particles from a high-energy collision) originated from a quark or a gluon. Assumed data source: EnergyFlow's `qg_jets` dataset (Komiske, Metodiev & Thaler; ~2M jets, Pythia8-generated, particle features `(pT, y, φ, pdgid)` per jet) — **this is an assumption, not yet confirmed against what the user has locally; flagged in Design Decision 1, confirm before T4.1 starts.** [tentatively, we might want to expand the scope of benchmarking to involve datasets such as PubMed or other Citation datasets too.]

This phase is a genuine architectural departure from Phases 0–3, which all operated on **one** large transductive graph (Cora/CiteSeer/PubMed: a single N, fixed for the whole phase). Jets are small (tens of particles), there are hundreds of thousands of them, and the task is graph-level (inductive) classification, not node-level link prediction. Several Phase 0–3 conventions do not transfer as-is — see Design Decisions.

---

## Design Decisions (resolved 2026-07-14, before writing this spec)

Four architectural questions were put to the user directly (`AskUserQuestion`) given how consequential — and hard to reverse — they are; the rest are ordinary spec-writing calls, made here and flagged for the user to override if wrong.

1. **Quantum framework: Qiskit + Qiskit Machine Learning** (user choice, over PennyLane and TensorFlow Quantum). Concretely: build the QGNN as a Qiskit `QuantumCircuit`, wrap it as an `EstimatorQNN` (Qiskit Machine Learning), and embed that inside the existing PyTorch training loop via `TorchConnector` — so Hydra configs, the W&B logging convention, and the optimizer loop from Phases 0–3 all carry over unchanged; only the model's forward pass gains a quantum sub-module. Gradients through the quantum circuit come from Qiskit Machine Learning's built-in parameter-shift-rule differentiation, exposed to PyTorch's autograd via `TorchConnector` transparently — no custom backward pass needs to be written.
2. **QGNN ansatz: graph-topology equivariant, not a generic hardware-efficient circuit** (user choice). Concretely, a **Verdon-style Quantum Graph Neural Network layer** (Verdon, Broughton, McClean et al., "Quantum Graph Neural Networks," arXiv:1909.12264 — the paper the "QGNN" name in this project's own mission/report already refers to): one qubit per pooled latent node (`M` qubits total), an entangling Hamiltonian `H = Σ_{(i,j) ∈ A_z} A_z[i,j] · ZZ_{ij} + Σ_i b_i · Z_i`, Trotterized into gates (`RZZ(θ · A_z[i,j])` per learned latent edge, `RZ(b_i)` per node), with `θ` and `b_i` trainable circuit parameters. This makes the circuit's entangling structure a direct, literal encoding of the learned latent graph `A_z` — the same "graph-structured latent space" story as the rest of the project, carried into the quantum circuit itself, rather than treating (z̃, A_z) as just another flat feature vector for a generic VQC.
3. **Fixed-`M` pooling, not ratio-based** (user choice). A quantum circuit has a fixed qubit count; jets have wildly varying particle counts (a handful to 100+), so `M` must be an **absolute** number, identical for every jet, not `M = round(pool_ratio · N)` as in T3.6's citation-network sweep (where all graphs shared one `N`). This requires **no change** to `PooledGVLS`/`LatentGraphPooling` (`src/gvls/models/pooling.py`) — `LatentGraphPooling(latent_dim, num_clusters)` already takes an absolute `num_clusters`, and `PooledGVLS`'s forward pass has no dependency on how `M` was chosen. Only `experiments/pooling_sweep.py`'s ratio-based `M = max(2, round(pool_ratio * n_nodes))` line is jet-inapplicable; Phase 4 calls `PooledGVLS`/`LatentGraphPooling` directly with the same constant `M` for every jet, sidestepping `pooling_sweep.py` entirely rather than modifying it (T3.6's citation-network use of ratio-based pooling remains valid and unchanged for its own purposes).
4. **No classical baseline in this phase** (user choice). Validate the pipeline by comparing against **literature-reported QGNN accuracy on this dataset**, not a from-scratch classical control — the user will decide whether classical baselines (uncompressed GNN, classical head on frozen z̃, mirroring T3.5's pattern) are needed after seeing how the literature comparison lands. **The specific paper(s) and numbers to compare against are not yet identified** — this is real, unresolved research legwork (T4.6), not a placeholder to fill in mechanically. Do not fabricate or guess a comparison number before that search is done.
5. **Jet → graph construction: k-NN in (η, φ) space, not a complete graph.** Jets don't come with an explicit "true" adjacency the way citation networks do. Building a complete graph (every particle connected to every other) would make GVLS's whole "compress the input graph's structure" framing close to vacuous — a complete graph has no topology to compress. Instead, build a k-nearest-neighbor graph over each jet's particles in angular (η, φ) space (rapidity/azimuth), matching the standard construction used in classical jet-tagging GNNs (e.g. ParticleNet's EdgeConv graph). `k_graph` (the input-graph k-NN parameter — distinct from `LatentGraphLearner`'s `k`) defaults to **min(particle_count − 1, 8)**, chosen to keep the input graph meaningfully sparse (mirroring the T3.3 finding that small `k` is what gives genuine compression) without leaving very small jets (fewer than 8 particles are common) disconnected.
6. **Per-jet node features: kinematic + particle-ID, not raw detector output.** Default to the four fields `qg_jets` actually provides per particle — `(log pT, y, φ, pdgid)` — with `pdgid` mapped to a small fixed-size one-hot (the dataset uses a bounded set of ~11 particle species: photon, e±, μ±, π±, K±, K_L, p, n̄/n). This gives `F ≈ 4 + 11 = 15` input features per particle node. **This is a default, not a locked-in decision** — if GVLS's compression fidelity on jets turns out to be poor with these features, revisit before blaming the pooling/quantum stages.
7. **Batching: per-jet forward pass, not a fully-batched dense implementation.** `GVLSEncoder`'s GCN layers (`src/gvls/models/encoder.py`) already tolerate an arbitrary single-graph `edge_index` and would batch fine via PyG's disjoint-union convention. But `LatentGraphPooling`, `LatentGraphLearner`, and `LatentMessagePassing` (Phases 1/3, reused unchanged per Design Decision 3) all operate on one dense `(N, d)` / `(N, N)` graph at a time — a jet's assignment softmax must not mix nodes from a different jet in the same minibatch into the same cluster, which PyG's flat disjoint-union batching would silently allow unless every one of those modules were rewritten to be batch-aware (masked block-diagonal softmax, block-diagonal `A_z`, etc.). Given jets are small (tens of particles → single-digit-microsecond dense ops), Phase 4 processes **one jet per forward call** and accumulates gradients over a minibatch of jets before each `optimizer.step()` — reusing `GVLS`/`PooledGVLS`/`LatentGraphPooling`/`LatentGraphLearner` completely unmodified, at the cost of some training throughput. Revisit only if this throughput actually blocks the phase (see NFR-2).
8. **Two-stage training, not joint end-to-end fine-tuning.** GVLS is pretrained fully unsupervised (ELBO only, no jet labels) on the full pretraining split, then **frozen**; (z̃, A_z) are extracted once per jet; the QGNN classifier is trained supervised on top, using quark/gluon labels — directly mirroring T3.5's frozen-features pattern for node classification. This keeps the (already novel) quantum training loop isolated from the classical ELBO training loop, so a failure in one is easy to attribute. Joint fine-tuning (backprop the classification loss through the quantum circuit *and* into the classical encoder/pooling/latent-graph stack) is a stretch goal (T4.7) — attempt only once the frozen-feature pipeline is validated end-to-end.
9. **Dataset scope: a subset, not the full ~2M-jet dataset.** Qiskit Aer's statevector simulator is exponential in qubit count but `M` here is small (single digits), so simulating one circuit is fast — the actual bottleneck is the **number of jets** to pretrain GVLS and train the QGNN on, both of which require one classical forward pass *and* one quantum circuit execution per jet per epoch. Start with a subset on the order of **10,000–50,000 jets** (balanced quark/gluon, standard train/val/test split), sized to keep a full training run tractable on a laptop/single machine; scale up only if accuracy is compute-bound rather than data-bound. Exact subset size is a tuning knob, not fixed here — record whatever is actually used in `validation.md` once T4.1 runs.

10. **Batch QGNN training instead of per-jet iteration (new, 2026-07-27, motivated by an actual slow real run, not anticipated at original spec time).** T4.5's design (Decision 8 above, and NFR-2) iterates jets one at a time through `QGNNClassifier` for the same reason the classical stack does (Decision 7) — but that reason doesn't actually apply here. The classical stack stays per-jet because `LatentGraphPooling`/`LatentGraphLearner`/`LatentMessagePassing` operate on one dense `(N,N)`-shaped graph and jets have different `N`; the QGNN operates purely downstream of pooling, on the fixed-size `(M,d)`/`(M,M)` `(z̃, A_z)` pair that's identical in shape across every jet by construction (that's the whole point of fixed-`M` pooling, Decision 3). There is no jagged-shape obstacle to batching the QGNN specifically.

    Diagnosis (see `specs/phase4/requirements.md` FR-5 and NFR-2 for the resulting requirement): `train_qgnn_classifier`'s inner loop calls the `TorchConnector`-wrapped circuit once per jet — one `EstimatorQNN`/Aer `Estimator.run()` job dispatch per jet, `batch_size` times more Python-level overhead per minibatch than necessary. Verified directly against the installed `qiskit-machine-learning==0.8.2` source (not assumed): `TorchConnector`'s `_TorchNNFunction.forward`/`backward` (`connectors/torch_connector.py`) already accept a 2D `(B, num_inputs)` tensor with a single shared 1D weight vector — the 1D-vs-2D branch only special-cases `len(shape)==1` — and `EstimatorQNN._forward`/`_backward` (`neural_networks/estimator_qnn.py`) already tile per-sample parameter values into one `estimator.run()` / `gradient.run()` call regardless of `num_samples`. So batching is natively supported by the pinned library version; it was simply never used because the per-jet loop pattern was copied from the classical stack's T4.2 precedent without re-examining whether the same constraint applied.

    This is a **constant-factor** fix, not an algorithmic one: parameter-shift still needs `O(batch_size × num_params)` circuit evaluations per minibatch — batching collapses `batch_size` separate job dispatches into one, it does not reduce the number of underlying circuit evaluations. A complementary, independent, near-zero-risk change is bundled in: `EstimatorQNN(..., input_gradients=True)` (`src/gvls/models/qgnn.py`) currently makes parameter-shift differentiate all 19 parameters (9 trainable weights + 10 runtime inputs: `A_z` edges + re-uploaded `z̃` features) per call, even though `z̃`/`A_z` are frozen extraction outputs that never need `∂L/∂input` — only the 9 weights are ever optimized. Setting `input_gradients=False` roughly halves the shifted-circuit count per sample, independent of and multiplicative with the batching win.

    Risk: this codebase has already found two non-obvious correctness bugs in this exact pinned `qiskit`/`qiskit-machine-learning` stack (T4.4's diagonal-ansatz degeneracy and `default_precision` shot-noise default, `specs/phase4/validation.md` V-4) — reading the library source is not sufficient grounds to trust batched forward/backward without a direct numerical-parity test against the known-correct per-jet loop (see T4.8 below).

11. **SPSA gradient estimator replaces parameter-shift as the default (T4.9, new 2026-07-27, user-directed after T4.8's measured speedup proved insufficient).** T4.8's own measurement (`validation.md` V-8) showed batching only removed per-jet job-dispatch overhead — the dominant cost was `ParamShiftEstimatorGradient` itself needing 2 circuit evaluations per differentiated parameter *per occurrence in the circuit* (not simply per parameter: `theta`, shared across all `m(m-1)/2` RZZ gates in a layer, alone costs `2×edges` evaluations — measured `28` total for the `m=4, num_layers=1` ansatz, not the naively expected `18`). `qiskit_machine_learning.gradients.SPSAEstimatorGradient` (verified directly from its source, `spsa_estimator_gradient.py`) perturbs every differentiated parameter jointly along one random ±1 direction and computes a single two-point finite difference — exactly `2 × spsa_batch_size` circuit evaluations per sample, *independent of parameter or gate count*. Measured: `2` evaluations vs. param-shift's `28` at `num_layers=1` (`14x`), `48` at `num_layers=2` (`24x`) — the gap widens as the ansatz grows since SPSA's cost never changes.

    **Tradeoff, not free:** SPSA's gradient is a stochastic single-direction estimate, not an analytic one, and every differentiated parameter in one call gets the *same-magnitude* estimate (`±diff`, sign only), since they're all divided from one shared finite difference. This makes SPSA structurally unable to isolate "does this one specific parameter have a genuinely zero gradient" — exactly the question T4.4's diagonal-ansatz degeneracy bug was about. `gradient_method="param_shift"` remains fully supported (a `QGNNClassifier` constructor argument, plumbed through `train_qgnn_classifier`/`configs/train/qgnn_classifier.yaml`) and is what the degeneracy-sensitive tests (`test_gradients_flow_to_weight_params`, `test_gradients_flow_with_multiple_layers`) and T4.8's exact gradient-parity tests pin to explicitly, rather than silently losing their regression value under the new default.

    **Real measured wall-clock (not assumed):** benchmarked against the pre-T4.8 original code (loaded from git commit `31b5afe`, not reconstructed from memory) on identical synthetic jets (`M=4, d=8, num_layers=1`, 300 train / 60 val jets, 3 epochs, CPU): the original took ~26.1s; batched+SPSA takes ~1.7s — a **~15.5x combined speedup**, of which SPSA alone (batched param-shift → batched SPSA) contributes ~11.7x on top of T4.8's own ~1.34x. This is the first change in the whole slow-training investigation that plausibly makes the target 10,000–50,000-jet × 50-epoch real run in Design Decision 9 tractable, though it hasn't been run at that real scale yet — see `validation.md` V-9.

12. **Literature comparison target confirmed: Lorentz-EQGNN, and the training pipeline realigned to match it (T4.10, new 2026-07-30, user-directed).** T4.6's literature-comparison search (Design Decision 4's "not yet identified" flag) is resolved by a benchmark table the user supplied directly (`sota-table.png`, Table II "Quark-Gluon Dataset" and Table III "Electron-Photon Dataset" — **the source paper itself is not yet bibliographically confirmed: no title/venue/arXiv ID has been verified, only the table image; do not cite this as a confirmed reference in `README.md`/`results/qgnn/` until that's resolved, per NFR-5**). Table II reports seven models on the Quark-Gluon dataset (the dataset this project already has, `energyflow.qg_jets`, T4.1); `Lorentz-EQGNN` is the strongest quantum/hybrid row and — uniquely among the table's rows — uses **4 qubits**, matching T4.3's already-selected compression-optimal `M=4` exactly, by coincidence rather than by design. Its full protocol: `AdamW` optimizer, `lr=1e-3`, `batch_size=16`, `50` epochs, `Cross Entropy Loss`, an `800`-jet training subset, reaching `74.00% ± 0.26%` test accuracy.

    User decisions (`AskUserQuestion`, 2026-07-30): (a) realign the pipeline to match Lorentz-EQGNN specifically, not the table's broader/looser conventions, and not a no-change citation-only approach; (b) Quark-Gluon only — Electron-Photon (Table III) explicitly deferred, no data source confirmed for it, revisit after Quark-Gluon results land; (c) match the `800`-jet training subset **literally**, as a dedicated comparability run distinct from Design Decision 9's larger 10,000–50,000-jet run (both are kept — the large run for the project's own rate-distortion/statistical-power purposes, the 800-jet run specifically for this comparison); (d) `evaluate_qgnn.py` gains train/test accuracy and wall-clock training/inference time reporting, matching the table's columns, in addition to the existing accuracy/AUC/macro-F1 metrics.

    **Reconciling `Cross Entropy Loss` with this project's single-logit binary readout.** `QGNNClassifier`'s observable (`sum_z_observable`, V-4) produces one scalar logit per jet, trained via `BCEWithLogitsLoss` (T4.5/T4.8's `qgnn_batch_loss`). The literature table's `Cross Entropy Loss` almost certainly means `nn.CrossEntropyLoss` over a 2-class softmax head, which is mathematically equivalent to binary cross-entropy on a single logit for a 2-class problem (softmax-CE with logits `(0, x)` reduces exactly to sigmoid-BCE with logit `x`) — not a functional difference, just a different parameterization. Building a literal 2-output softmax head would require a second observable/readout qubit, which is unnecessary for equivalence and would confound the "matches Lorentz-EQGNN's qubit count" comparison. **Decision: keep `BCEWithLogitsLoss` on the existing single-logit readout, documented explicitly in `results/qgnn/` and `README.md` as the equivalent of the table's `Cross Entropy Loss` rather than a literal `nn.CrossEntropyLoss` call** — a real reconciliation judgment call, recorded here so it isn't mistaken for an unexamined mismatch later.

    **What this does *not* change:** the QGNN ansatz itself (Design Decisions 1–2, T4.4), the SPSA gradient estimator default (Design Decision 11, T4.9 — Lorentz-EQGNN's own classical-optimizer choice doesn't speak to our quantum gradient-estimation method, a separate concern), or T4.3's GVLS pretraining sweep (`M=4` was already selected before this table was seen; T4.10 does not re-run T4.3, it only changes T4.5's stage-2 QGNN training and T4.1's jet-subset size for this specific comparability run).

    **Honesty flag (NFR-5):** wall-clock training/inference time comparisons against the table are directional only, not apples-to-apples — the table's hardware is unknown and almost certainly differs from whatever machine this project's run executes on. Report our own measured numbers plainly; do not imply hardware parity.

13. **Data protocol corrected to match arXiv:2411.01641 exactly, after reading the source paper directly (T4.10 Part B, 2026-08-01, user-directed).** Design Decision 12 above matched Lorentz-EQGNN's *hyperparameters* from the table image alone; it did not match the paper's *data protocol*, which turned out to differ on every axis: the paper draws one fixed `12,500`-jet pool (jets with `>=10` particles, **no forced class balance** — its own split yields `4,982/658/583` quark jets, not exact 50/50), splits it `80/10/10` **positionally** into `10,000/1,250/1,250` train/val/test, and its `800`-jet data-scarcity row shrinks **only the training set** — validation and test stay fixed at `1,250/1,250`. The prior comparability config instead drew an `800`-jet **class-balanced** pool and re-split it `70/15/15` (`560/120/120`) — every one of those four properties was wrong. `load_qg_jets_lorentz_protocol` + `subsample_train`(`_balanced`) (`src/gvls/data/jets.py`, `configs/data/qg_jets_lorentz.yaml`) implement the corrected protocol; the pre-existing balanced/ratio-split behavior (`qg_jets.yaml`, used by Design Decision 9's separate 10,000–50,000-jet target run) is unaffected. See `validation.md` V-10 Part B.

14. **Repeated-training-subset methodology: 5 trials on different balanced 800-jet draws, not 5-fold CV and not 5 identical-data seed reruns (T4.10 Part C, 2026-08-01–02, user-directed).** The literature's `± 0.26%` std implies 5-fold cross-validation — a genuinely different variance source (re-partitions the data itself) than either (a) reusing one fixed 800-jet training subset across all 5 trials, which would vary only the QGNN's own training seed, or (b) true k-fold CV, which would also re-partition validation/test and likely require retraining the frozen GVLS encoder per fold. After discussing the tradeoff directly with the user, the adopted middle ground is: each trial draws a *different* class-balanced (`400`/`400`) `800`-jet training subset from the same fixed `10,000`-jet pool (`train_subset_seed`, decoupled from the outer `seed` that fixes val/test), while GVLS is pretrained once and reused frozen across all 5 trials (it is deterministic and produces no gradient-bearing variance for the QGNN stage to sample over — confirmed, not assumed, since `extract_latent_features` runs under `torch.no_grad()`). This is explicitly documented as an approximation of 5-fold CV, not a claim of matching it. Also fixed in this window: `train_pooled_gvls_on_jets` never selected a best-validation checkpoint (always returned the last epoch unconditionally, unlike `train_qgnn_classifier`'s existing best-val-accuracy tracking) — now mirrors that behavior when `eval_jets` is given. See `validation.md` V-10 Part C.

15. **First real comparability run scored far below the literature target (59.12% vs. 74.00%); diagnosed as GVLS data starvation, not threshold miscalibration (T4.10 followup, 2026-08-02, user-directed).** The first real 800-jet run (both GVLS pretraining and QGNN training capped at 800 jets, since Stage 1 inherited Stage 2's data-args block) scored `59.12% ± 4.10%` test accuracy — a `~14.9`-point gap. Diagnostic signal: AUC/AP (threshold-independent) were stable across seeds while accuracy/macro-F1/recall (threshold-dependent) were not — consistent with a miscalibrated decision boundary. Two fixes were tried together: (a) a validation-selected decision threshold (`select_best_threshold`, `src/gvls/eval/metrics.py`, searched exactly over achievable operating points, never tuned against the test set); (b) decoupling GVLS's own pretraining set size from the QGNN's `800`-jet comparability cap — Lorentz-EQGNN's `800`-jet constraint is about *its* classifier's training budget; it has no unsupervised pretraining stage of its own, so nothing about comparability requires GVLS to also be starved to `800` jets. **Result: (b) was the actual fix.** `test_accuracy` rose to `66.88% ± 0.30%` (gap narrowed to `~7.1` points), but the validation-selected threshold landed almost exactly on the un-tuned `0.5` default (`threshold_mean=0.5077`) and its accuracy was statistically indistinguishable from the fixed-`0.5` number — the earlier instability disappeared because GVLS's features improved, not because the threshold was recalibrated. The threshold machinery is kept (cheap, harmless) but is not the causal story. This is the current best confirmed configuration (`num_layers=1`, `readout_mode=sum`, `epochs=50`). See `validation.md` V-11 Step 1.

16. **Classical-baseline diagnostic found the QGNN underperforming a plain logistic regression on identical frozen features, and found the actual cause: a re-uploading bottleneck (T4.10 followup, 2026-08-02–03, user-directed).** To decide whether the remaining `~7.1`-point gap was a GVLS feature-quality ceiling or a QGNN-training deficiency, `LogisticRegression`/`MLPClassifier` were trained on the exact same frozen `(z̃, A_z)` features, same 5-trial protocol (`src/gvls/eval/classical_baseline.py`). Both beat the QGNN (`69.07%`/`69.49%` vs. `66.88%`) by `2.2`–`2.6` accuracy points, and landed within `0.4` points of each other (features are close to linearly separable — little unused nonlinear structure). This placed the bottleneck squarely on the QGNN side. Reading `src/gvls/models/qgnn.py` directly found the mechanism: `encode_input`'s re-uploading is `dim = layer % d`; at the then-default `num_layers=1, d=8`, every circuit only ever encoded `z̃`'s dimension `0` — the other `7` dimensions were discarded before the circuit ever saw them, while the classical baselines used the full `32`-dimensional `z̃`. See `validation.md` V-11 Step 2.

17. **Three QGNN-architecture interventions tried in response to Design Decision 16, all failed to beat the 66.88% baseline, escalating cost up to ~31.6x (T4.10 followup, 2026-08-03–04, user-directed).** In order: (a) `num_layers = GVLS_LATENT_DIM` (full re-uploading, directly targeting Design Decision 16's finding) — regressed to `65.07%` at `epochs=50` (`849`s/trial, `3`x); both train *and* test accuracy dropped together, an optimization-difficulty signature (trainable weights `9→44`, and SPSA's joint-perturbation gradient estimate is known to degrade as parameter count grows), not an overfitting one. (b) Same config, `epochs=150` (more optimization steps, user-directed, before concluding the deeper circuit is fundamentally harder to train) — recovered to `66.08% ± 1.17%`, still a statistical wash against the `66.88%` baseline, at `2539`s/trial (`8.8`x) and `~4`x the run-to-run std. (c) `readout_mode="learned"` (per-qubit `Z` measurement + a trainable classical `Linear(m,1)` head, `src/gvls/models/qgnn.py`'s `per_qubit_z_observables`/`readout_head`, kept `num_layers=8`/`epochs=150`, user-directed as a lower-risk complementary fix instead of pushing `num_layers` further, since the head's own gradient is exact autograd and adds zero parameters to SPSA's spread) — `65.90% ± 2.02%` (no improvement), with train accuracy actually *dropping* to `67.08% ± 2.34%` (from `69.30%`) and `9069`s/trial (`31.6`x). **Correction made and measured, not left standing:** (c) was initially described as "free" in circuit-evaluation terms; measured directly instead (mirroring Design Decision 11's methodology) and found `EstimatorQNN`'s gradient estimators submit one pub per observable rather than batching a multi-observable readout, multiplying evaluation count by `m` (SPSA `2→8`, param-shift `28→112` at `m=4`) — small in absolute terms, but not free. **Recommendation (not yet executed): revert to `num_layers=1`, `readout_mode=sum`, `epochs=50`** — the best, cheapest, most consistent confirmed result. The GVLS-side auxiliary-supervision idea (a light supervised classification loss during GVLS's otherwise-unsupervised pretraining) was explicitly deferred by the user when this followup was scoped to the QGNN side only, and remains the most-evidenced untried lever given (16)'s finding that the ceiling sits in what the QGNN extracts, and this decision's finding that pushing the QGNN's own architecture/training under SPSA hasn't closed it. See `validation.md` V-11 Step 3.

---

## Scope

### In scope
- **T4.1** — Jet dataset loader: download/parse the Pythia8 quark/gluon dataset, build a per-jet k-NN graph (Design Decision 5), extract node features (Design Decision 6), produce a labeled train/val/test split
- **T4.2** — Inductive per-jet adaptation of the existing GVLS/`PooledGVLS` stack (Design Decisions 3, 7): a training loop that iterates jets one at a time, with fixed `M`, accumulating gradients over a minibatch
- **T4.3** — GVLS pretraining on jets: unsupervised ELBO training (reusing `src/gvls/losses/elbo.py` unchanged) across a small sweep over `M ∈ {4, 6, 8}`, reusing Phase 3's compression metrics (`reconstruction_f1`, `bits_per_edge`) computed per-jet and averaged, to pick the smallest `M` with acceptable fidelity — this phase's version of T3.3's rate-distortion sweep, at jet scale
- **T4.4** — QGNN ansatz (Design Decisions 1, 2): Qiskit circuit construction from `(z̃, A_z)`, wrapped as an `EstimatorQNN`, embedded in a `torch.nn.Module` via `TorchConnector`
- **T4.5** — Two-stage supervised training (Design Decision 8): freeze the pretrained GVLS, extract (z̃, A_z) for every jet once, train the QGNN classifier on quark/gluon labels; Hydra config + W&B logging following the existing convention (new group tag, e.g. `qgnn-jet-classification`)
- **T4.6** — Evaluation: accuracy / AUC / macro-F1 on held-out test jets; literature search to identify a comparable published QGNN result on this dataset and report against it (Design Decision 4) — the comparison target itself is a deliverable of this task, not an input to it
- **T4.8 (new, 2026-07-27)** — Batch T4.5's QGNN training loop (Design Decision 10): replace the per-jet `EstimatorQNN`/`TorchConnector` call with one batched call per minibatch, plus `input_gradients=False`, to fix real observed training slowness
- **T4.9 (new, 2026-07-27)** — Replace parameter-shift with SPSA as the default gradient estimator (Design Decision 11): a genuine reduction in circuit-evaluation count (not just dispatch overhead), measured ~15.5x combined speedup over the pre-T4.8 original
- **T4.10 (new, 2026-07-30)** — Realign the pipeline with the Lorentz-EQGNN literature baseline (Design Decision 12): `AdamW`, `lr=1e-3`, `batch_size=16`, an `800`-jet comparability subset, and train/test-accuracy + wall-clock reporting in `evaluate_qgnn.py`, so the eventual T4.6 comparison is a direct row-by-row match against the literature table rather than a loosely-comparable citation
- **T4.11 (new, 2026-08-01)** — Correct the comparability run's data protocol to match arXiv:2411.01641 exactly after reading it directly (Design Decision 13): fixed `12,500`-jet pool, no forced class balance, `80/10/10` positional split, `800`-jet row shrinks only training. Also implements the 5-trial repeated-balanced-training-subset methodology (Design Decision 14) and fixes GVLS pretraining's missing best-val-F1 checkpoint selection
- **T4.12 (new, 2026-08-02)** — Diagnose and fix the first real run's 59.12% result (Design Decision 15): validation-selected decision threshold (implemented, later found not to be the causal fix) + decoupling GVLS's pretraining set size from the QGNN's 800-jet cap (the actual fix, 59.12%→66.88%)
- **T4.13 (new, 2026-08-02–04)** — Classical-baseline diagnostic (Design Decision 16) identifies a QGNN re-uploading bottleneck; three architecture interventions tried in response (Design Decision 17) — full re-uploading, more epochs, a learned classical readout — none beat the 66.88% baseline; revert recommended, not yet executed

### Stretch / explicitly deferred
- **T4.7 (stretch)** — Joint end-to-end fine-tuning of GVLS + QGNN together (Design Decision 8), compared against the frozen-feature baseline
- ~~Classical baselines (uncompressed classical GNN on the full jet graph; classical head on frozen z̃ with no quantum circuit) — deferred per Design Decision 4, pick up only if the literature comparison is inconclusive~~ **Done (T4.13, 2026-08-02–03, Design Decision 16)** — a classical head on frozen z̃ (`LogisticRegression`/`MLPClassifier`) was trained and found to beat the QGNN by 2.2–2.6 accuracy points on identical features; not an uncompressed classical GNN on the raw jet graph, which remains untried
- Real quantum hardware execution or noise-model simulation — Qiskit Aer noiseless statevector simulation only this phase; hardware/noise is Phase 5+ ablation material
- Any new latent-graph-inference method beyond Phase 1's existing attention/FGP/NRI — reuse whichever method Phase 2/3 already validated as a starting point
- Multi-class or full-detector jet tagging — this dataset and task are binary quark-vs-gluon only
- **Electron-Photon dataset (new, 2026-07-30)** — `sota-table.png`'s Table III reports the same literature methods on an Electron-Photon jet dataset; user explicitly deferred pursuing it (`AskUserQuestion`, 2026-07-30) until Quark-Gluon results land. No data source is confirmed for it.
- **GVLS auxiliary supervision (new, 2026-08-04)** — a light supervised classification loss during GVLS's otherwise-unsupervised pretraining (semi-supervised GVLS), to test whether the classical-baseline ceiling (Design Decision 16, ~69–70%) can itself be raised. User explicitly deferred this when the T4.10 followup was scoped to QGNN-side interventions only (`AskUserQuestion`, 2026-08-04) — no design work done yet. Remains the most-evidenced untried lever given Design Decision 17's three QGNN-side attempts all failed to close the gap.

---

## File Map

```
src/gvls/
  data/
    jets.py                    # T4.1 (new) — qg_jets download/parse, kNN graph
                                #              construction, feature engineering,
                                #              train/val/test split
  compression/
    jet_sweep.py                # T4.3 (new) — per-jet rate-distortion sweep over M,
                                #              mirrors compression/sweep.py's structure
                                #              but iterates jets instead of one big graph
  models/
    qgnn.py                      # T4.4 (new) — Qiskit circuit builder from (z̃, A_z),
                                #              EstimatorQNN + TorchConnector wrapper
configs/
  data/
    qg_jets.yaml                # T4.1 (new) — dataset path/subset-size/split config
  train/
    jet_pretrain.yaml           # T4.3 (new) — GVLS-on-jets pretraining config
    qgnn_classifier.yaml        # T4.5 (new) — QGNN supervised training config
  experiment/
    jet_pooling_sweep.yaml      # T4.3 (new) — M grid definition ({4, 6, 8})
experiments/
  pretrain_gvls_jets.py         # T4.3 (new) — Hydra CLI wrapper for jet_sweep.py
  train_qgnn.py                  # T4.5 (new) — Hydra CLI wrapper for the two-stage
                                #              (frozen GVLS → QGNN) training loop
  evaluate_qgnn.py                # T4.6 (new) — test-set metrics + literature comparison
tests/
  test_jets.py                   # T4.1 (new) — graph construction, feature shapes,
                                #              split determinism
  test_jet_sweep.py               # T4.3 (new) — smoke test, tiny M grid, few jets
  test_qgnn.py                     # T4.4 (new) — circuit shape/qubit-count, gradient
                                #              flow through TorchConnector, A_z-edge
                                #              → RZZ-gate correctness on a toy graph
  test_train_qgnn.py               # T4.5 (new) — end-to-end smoke test, tiny jet
                                #              subset, few epochs
```

**New top-level dependencies:** `qiskit`, `qiskit-machine-learning`, `energyflow` (for `qg_jets`, if that ends up being the confirmed data source — see Design Decision 1's open flag).

---

## Tasks

### T4.1 — Jet dataset & graph construction

**File:** `src/gvls/data/jets.py`

- Load the Pythia8 quark/gluon dataset (assumed: `energyflow.datasets.qg_jets`; **confirm this against what the user actually has/wants before implementing** — see Design Decision 1)
- Per jet: build a k-NN graph over particles in `(η, φ)` space (`k_graph = min(particle_count − 1, 8)`, Design Decision 5); assemble node feature matrix `(log pT, y, φ, one_hot(pdgid))` (Design Decision 6)
- Produce a `JetGraph` container (PyG `Data`-compatible: `x`, `edge_index`, `y` for the quark/gluon label) per jet
- Subset to a tractable size (Design Decision 9) with a balanced quark/gluon label ratio; standard train/val/test split (e.g. 70/15/15), deterministic given a seed

Tests (`tests/test_jets.py`):
- k-NN graph construction on a synthetic jet: correct edge count, no self-loops, symmetric
- Feature matrix shape `(num_particles, F)` matches Design Decision 6's `F`
- Determinism: same seed → identical split
- Label balance within a configurable tolerance of 50/50 on the subset

---

### T4.2 — Inductive per-jet GVLS adaptation

**Files:** training-loop code only (likely folded into T4.3's `jet_sweep.py`, not a separate module — see note below)

- A training loop that iterates jets one at a time through `GVLSEncoder` → `LatentGraphPooling` (fixed `M`) → `LatentGraphLearner` → `LatentMessagePassing` → unpooled reconstruction logits, **reusing `PooledGVLS` (`src/gvls/models/pooling.py`) completely unmodified** (Design Decision 3)
- Per-jet losses (`elbo(...) + entropy_weight · assignment_entropy(S) + aux_link_weight · assignment_link_loss(S, A)`, all reused unchanged from T3.6) are summed/averaged over a minibatch of jets before `optimizer.step()` (Design Decision 7)
- No new model code is expected here — this task is "does the existing T3.6 stack work correctly when called once per jet in a loop," which is really a test-and-validate task folded into T4.3's implementation, not a standalone module. Kept as a separate task ID because it's a separate *risk* (correctness of per-jet iteration + gradient accumulation), not because it produces separate code.

Tests: covered by `test_jet_sweep.py` (T4.3) — a smoke test that confirms gradients reach the encoder, pooling, and latent-graph-learner parameters after a minibatch of jets, and that no cross-jet leakage occurs (e.g. two jets with disjoint feature ranges produce assignments that don't reference each other's nodes).

---

### T4.3 — GVLS pretraining sweep over jet-level `M`

**File:** `src/gvls/compression/jet_sweep.py`, `experiments/pretrain_gvls_jets.py`

- Mirrors `src/gvls/compression/sweep.py`'s structure (T3.3) but iterates jets: for each `M ∈ {4, 6, 8}`, pretrain `PooledGVLS` unsupervised (ELBO only) over the pretraining split (T4.2's per-jet loop), then compute **per-jet** `reconstruction_f1` and `bits_per_edge`, averaged over a held-out subset of jets
- `(hidden_dim, latent_dim d, k, graph_method, prior, mp_rounds, lr, beta, lambda_)` start from whichever config Phase 2/3 already validated as a reasonable default (not re-run through NAS for jets in this phase — that's Phase 5 ablation material if jet performance demands it)
- Select the compression-optimal `M`: smallest `M` whose average per-jet F1 is within a small tolerance of the largest tested `M`'s F1 (mirrors T3.3's rate-distortion logic, adapted since there's no single fixed 0.90 floor precedent yet for jets)
- Write one row per `M` to `results/compression/qg_jets_pooling.csv` (same schema convention as `results/compression/{dataset}_pooling.csv`, plus a `dataset=qg_jets` column)
- Log each `M` value's run to W&B under group tag `jet-compression-sweep`

Tests (`tests/test_jet_sweep.py`):
- Smoke test: tiny `M` grid (`{4, 6}`), a handful of synthetic jets, few epochs, completes without error and writes rows to the results CSV
- Gradient-flow and no-cross-jet-leakage checks (T4.2's validation, folded in here)

---

### T4.4 — QGNN ansatz (Qiskit)

**File:** `src/gvls/models/qgnn.py`

- `build_qgnn_circuit(M, num_layers=1) -> QuantumCircuit`: `M` qubits; each layer applies (a) a data-encoding sub-layer of single-qubit rotations (`RY(feature)` per qubit, one rotation per z̃ feature via data re-uploading if `d > 1` features need encoding per qubit — see open question below) and (b) the Verdon-style entangling sub-layer: `RZZ(θ · A_z[i,j])` for every edge `(i,j)` present in `A_z` (Design Decision 2), plus `RZ(b_i)` per qubit; `θ`, `b_i` (and any per-layer copies, if `num_layers > 1`) are the circuit's trainable parameters, `A_z[i,j]` and the encoded z̃ features are runtime *inputs*, not trainable weights
- Wrap as a Qiskit Machine Learning `EstimatorQNN` (observable: `Z` on a designated readout qubit, or a sum of `Z_i` across all qubits — pick whichever gives a better-conditioned gradient empirically, record the choice)
- Wrap the `EstimatorQNN` in a `TorchConnector` so it behaves as a standard `torch.nn.Module` with a `forward(z_tilde, A_z) -> logit` signature, matching the rest of the codebase's PyTorch-first convention
- **Open question, to resolve during implementation, not fixed here:** `z̃` is `d`-dimensional per node (`d` from GVLS's compression sweep, likely `d ∈ {4, 8, 16}` per the citation-network precedent) but each qubit only has one natural single-qubit rotation axis per encoding pass — either (a) use only 1–2 of z̃'s `d` dimensions per qubit (a further, deliberate information bottleneck beyond `M`), or (b) use data re-uploading (multiple encoding+entangling layers, cycling through z̃'s dimensions across layers). Document whichever is chosen and why once T4.4 is actually implemented.

Tests (`tests/test_qgnn.py`):
- Circuit has exactly `M` qubits for a given `M`
- On a toy `A_z` with a known edge set, the constructed circuit's `RZZ` gates appear on exactly those qubit pairs (topology correctness — the core claim of Design Decision 2)
- Gradients flow from the `TorchConnector`-wrapped module's output back to the circuit's trainable parameters (`θ`, `b_i`) via a `.backward()` call
- Zero-`A_z` sanity check: with no edges, the circuit reduces to independent single-qubit rotations (no entangling gates fire)

---

### T4.5 — Two-stage supervised QGNN training

**File:** `experiments/train_qgnn.py`

1. Load the frozen, pretrained GVLS from T4.3 at the selected `M`; run it once over every jet in the labeled train/val/test split to extract `(z̃, A_z)` per jet (no further gradient updates to GVLS — Design Decision 8)
2. For each jet: build its QGNN circuit (T4.4) from `(z̃, A_z)`, get the readout logit, compute BCE loss against the quark/gluon label
3. Train the QGNN's circuit parameters (`θ`, `b_i`) with Adam (via `TorchConnector`'s PyTorch-compatible autograd), minibatched with gradient accumulation across jets (Design Decision 7)
4. Track train/val loss and accuracy per epoch; log to W&B under group tag `qgnn-jet-classification`
5. Save the best (by val accuracy) circuit parameters

Tests (`tests/test_train_qgnn.py`):
- End-to-end smoke test: tiny jet subset (≤20 jets), `M=4`, 2–3 epochs, completes without error, loss is finite and does not NaN

---

### T4.6 — Evaluation and literature comparison

**File:** `experiments/evaluate_qgnn.py`

- Report accuracy, AUC, macro-F1 on the held-out test split
- **Literature search task (not yet done):** identify at least one published result reporting QGNN (or, failing that, closely related quantum-ML jet-tagging) accuracy on this same or a comparable quark/gluon jet dataset, and report GVLS+QGNN's numbers alongside it in `results/qgnn/` and `README.md`. If no directly comparable published QGNN number exists, say so explicitly rather than comparing against an unrelated benchmark and implying equivalence.
- Report the qubit count (`M`) and circuit depth actually used, since — per the midterm report's own framing — the qubit/gate budget achieved is as much a headline result here as classification accuracy

---

### T4.7 — Joint fine-tuning ablation (stretch)

Only attempted once T4.1–T4.6 produce a working, evaluated frozen-feature pipeline.

- Unfreeze GVLS; backpropagate the QGNN's classification loss through the quantum circuit (via `TorchConnector`) and into the classical encoder/pooling/latent-graph stack, alongside (or instead of) the unsupervised ELBO term
- Compare test accuracy against the frozen-feature baseline (T4.5/T4.6) to determine whether end-to-end fine-tuning is worth the added training complexity

---

### T4.8 — Batched QGNN training (new, 2026-07-27, performance fix for T4.5)

**Files:** `src/gvls/models/qgnn.py` (`QGNNClassifier.encode_input_batch`, batch-dispatching `forward`), `src/gvls/qgnn_training.py` (`collate_jet_features`, `qgnn_batch_loss`, batched `train_qgnn_classifier` inner loop, batched `evaluate_qgnn_classifier`)

Triggered by T4.5's real run (deferred to a remote machine, `validation.md` V-5) proving intractably slow. See Design Decision 10 for the full diagnosis and rationale; summarized here as concrete changes:

1. **`QGNNClassifier.encode_input_batch(z_tildes, a_zs)`** — `(B,M,d)`, `(B,M,M)` in, `(B, num_input_params)` out; stacks the existing per-jet `encode_input` across the batch dimension (pure tensor construction, no quantum calls, so a Python loop here is fine).
2. **`QGNNClassifier.forward`** — dispatches on `z_tilde.dim()`: `dim()==3` routes through `encode_input_batch` and a single `self.connector(...)` call for the whole batch; `dim()==2` keeps today's single-jet path unchanged (needed by any remaining single-jet callers/tests).
3. **`collate_jet_features(features: list[JetFeatures])`** — stacks a minibatch's `z_tilde`/`a_z`/`label` into `(B,M,d)`, `(B,M,M)`, `(B,)` tensors. Valid because every jet's pooled shape is identical by construction (fixed-`M` pooling, Design Decision 3) — no padding/masking logic needed, unlike the classical stack's per-jet constraint (Design Decision 7).
4. **`qgnn_batch_loss`** — one `BCEWithLogitsLoss` call (mean-reduced) over a collated minibatch, replacing the current per-jet loop + manual gradient accumulation in `train_qgnn_classifier` (`qgnn_training.py:133-143`) with a single `loss.backward()` per minibatch.
5. **`evaluate_qgnn_classifier`** — same batching treatment for the eval pass (chunked if the val/test split is large enough that one `EstimatorQNN` call over the whole split is impractical).
6. **`EstimatorQNN(..., input_gradients=False)`** in `QGNNClassifier.__init__` (`qgnn.py:196`) — bundled in as a complementary, independent fix (Design Decision 10): `z̃`/`A_z` are frozen extraction outputs that never need `∂L/∂input`, so computing input gradients via parameter-shift is pure waste.

Tests (`tests/test_qgnn.py`, `tests/test_qgnn_training.py`):
- `test_forward_batch_matches_per_jet_loop` — for a fixed batch of jets and fixed weights, the batched `forward` call's logits match calling `forward` per jet and stacking (float tolerance).
- `test_batched_backward_matches_accumulated_per_jet_gradients` — batched `loss.backward()` produces `theta`/`b_i`/`gamma_i` gradients matching the sum of per-jet gradients from the pre-T4.8 loop — the direct analogue of T4.2's `test_gradient_accumulation_matches_batched_mean`, load-bearing here specifically because this codebase has already found two non-obvious correctness bugs in this exact pinned `qiskit`/`qiskit-machine-learning` stack (V-4) and reading the library source is not sufficient grounds for trust on its own.
- A ragged-final-minibatch case (`len(train_features) % batch_size != 0`) for `collate_jet_features`.
- `test_train_qgnn_classifier_smoke` (existing) must keep passing unchanged.

**Explicitly not claimed:** this does not reduce the underlying `O(num_jets × num_params)` parameter-shift circuit-evaluation count — it removes `batch_size`-fold per-jet job-dispatch overhead only. Record the actual before/after wall-clock once run, rather than assuming the fix worked.

---

### T4.9 — SPSA gradient estimator (new, 2026-07-27, follow-up to T4.8)

**Files:** `src/gvls/models/qgnn.py` (`QGNNClassifier.__init__` gains `gradient_method`/`spsa_epsilon`/`spsa_batch_size`), `src/gvls/qgnn_training.py` (`train_qgnn_classifier`, `load_qgnn_checkpoint` plumb the same through), `experiments/train_qgnn.py`, `configs/train/qgnn_classifier.yaml`

Triggered by T4.8's own measurement showing batching's contribution was small (~1.07-1.08x) — the dominant cost was intrinsic to `ParamShiftEstimatorGradient`, not per-jet dispatch overhead, so a real fix required reducing the circuit-evaluation count itself, not just how it's dispatched. See Design Decision 11 for the full diagnosis; summarized here as concrete changes:

1. **`QGNNClassifier.__init__`** gains `gradient_method: str = "spsa"` (`"param_shift"` remains selectable), `spsa_epsilon: float = 1e-6`, `spsa_batch_size: int = 1` — constructs `qiskit_machine_learning.gradients.SPSAEstimatorGradient` or `ParamShiftEstimatorGradient` accordingly and passes it to `EstimatorQNN(gradient=...)` (passing an explicit `gradient` skips `EstimatorQNN`'s own default-construction branch and its warning entirely — verified against the source, not assumed).
2. **Shared `AerEstimatorV2()` instance** passed to both `EstimatorQNN` and the gradient estimator — `AerEstimatorV2`'s own `default_precision` is `0.0` (exact) by default (a class-level `Options` default, distinct from `EstimatorQNN`'s own separate `default_precision=0.015625` default that V-4 had to override), so `SPSAEstimatorGradient`'s internal `estimator.run(...)` calls (which don't pass a `precision=` override themselves) are exact/noiseless automatically through this shared instance — verified directly, not assumed, since it would have been easy to re-introduce V-4's shot-noise bug in a new code path.
3. **Determinism preserved:** SPSA's internal RNG is seeded from the same `seed` argument `QGNNClassifier` already uses for weight initialization, satisfying NFR-1 — "deterministic" here means "the same stochastic gradient estimate for the same seed and inputs," not "the true analytic gradient."
4. **`train_qgnn_classifier`/`experiments/train_qgnn.py`/`configs/train/qgnn_classifier.yaml`** plumb `gradient_method`/`spsa_epsilon`/`spsa_batch_size` through as ordinary Hydra-configurable training hyperparameters, mirroring how `graph_method`/`prior` are exposed elsewhere in the codebase. `save_qgnn_checkpoint`'s config now includes `gradient_method` for provenance (functionally inert at load time, since `gradient_method` only affects `.backward()`, never `.forward()`/inference — restored anyway, defaulting to `"spsa"` for older checkpoints).

**Correctness-vs-testing tension, resolved explicitly:** SPSA's single-shared-perturbation-direction estimate means every differentiated parameter in one call gets the same-magnitude gradient (`±diff`), so it cannot distinguish "this parameter has a genuinely zero analytic gradient" from "this parameter only mattered because it was perturbed jointly with others that do" — exactly the property T4.4's diagonal-ansatz-degeneracy tests need to check. Rather than silently letting those tests lose their regression value under the new default, `test_gradients_flow_to_weight_params`, `test_gradients_flow_with_multiple_layers`, and T4.8's exact gradient-parity test (`test_batched_backward_matches_accumulated_per_jet_gradients`) now pass `gradient_method="param_shift"` explicitly.

Tests (`tests/test_qgnn.py`):
- `test_spsa_is_the_default_gradient_method` / `test_param_shift_still_selectable` / `test_invalid_gradient_method_raises`.
- `test_spsa_evaluation_count_is_constant_independent_of_weight_count` — monkeypatches the shared `AerEstimatorV2.run` to directly count pubs evaluated during one `.backward()` call. Measured (not hand-derived): param-shift needs `28` evaluations for the `m=4, num_layers=1` ansatz, not the naively-expected `2×9=18` — `theta` is shared across `m(m-1)/2` RZZ gates per layer and the shift rule needs a shifted pair per *occurrence*, not per parameter (`12` of the `28` are `theta`'s alone). SPSA needs a constant `2`.
- `test_spsa_evaluation_count_scales_with_num_layers_for_param_shift_only` — confirms the gap widens with `num_layers` (`48` vs. `2` at `num_layers=2`) while SPSA's count stays flat.
- `test_spsa_gradient_deterministic_given_fixed_seed`, `test_spsa_gradients_flow_and_are_nonzero`.

**Real measured wall-clock (not assumed):** see Design Decision 11 — ~15.5x combined speedup (T4.8 + T4.9) over the pre-T4.8 original, of which SPSA alone contributes ~11.7x on top of T4.8's batching. Not yet run at the target real dataset scale (10,000–50,000 jets); the synthetic-jet benchmark is directionally strong evidence but not a substitute for an actual run.

---

### T4.10 — Realign the pipeline with the Lorentz-EQGNN literature baseline (new, 2026-07-30)

**Files:** `configs/data/qg_jets.yaml` (or a new named override, e.g. `configs/data/qg_jets_lit_compare.yaml`) — `num_jets=800`; `configs/train/qgnn_classifier.yaml` — `optimizer=AdamW`, `lr=1e-3`, `batch_size=16` (`epochs=50` already matches); `src/gvls/qgnn_training.py` (`train_qgnn_classifier`'s optimizer construction, currently hardcoded `torch.optim.Adam`, `qgnn_training.py:184`) needs an `optimizer` selector, not just an `lr` override, to switch to `AdamW`; `experiments/evaluate_qgnn.py` — add train accuracy, wall-clock training time, and wall-clock inference time to its reported/logged metrics.

Triggered by a benchmark table the user supplied directly (Design Decision 12): `Lorentz-EQGNN` (4 qubits, `AdamW`, `lr=1e-3`, `batch_size=16`, `50` epochs, `Cross Entropy Loss`, `800`-jet subset, `74.00% ± 0.26%` test accuracy on Quark-Gluon) is the closest comparable published result — same qubit count as our compression-optimal `M=4` — so the pipeline is realigned to reproduce its protocol as closely as our single-logit-readout architecture allows, per the reconciliation in Design Decision 12.

1. **`800`-jet comparability config** — additive to (not replacing) Design Decision 9's 10,000–50,000-jet target; same 70/15/15 split convention (`560`/`120`/`120`) unless the literature table's own validation protocol turns out to be discoverable and different (currently unknown — the table only states "800 (subset)" with no train/val/test breakdown; document this as an assumption per NFR-5 until/unless resolved).
2. **`configs/train/qgnn_classifier.yaml` realigned**: `AdamW` (new optimizer option in `train_qgnn_classifier`), `lr=1e-3` (was `0.05`), `batch_size=16` (was `32`). `epochs=50` unchanged. `gradient_method=spsa` (T4.9's default) is **not** changed by this task — Lorentz-EQGNN's own quantum-gradient method (if any) isn't stated in the table, and T4.9's SPSA-vs-param-shift choice is an orthogonal concern (Design Decision 11).
3. **Loss function reconciliation**: keep `BCEWithLogitsLoss` on the single-logit readout (no code change needed — already the default), documented in `results/qgnn/` and `README.md` as equivalent to the table's `Cross Entropy Loss` (see Design Decision 12's derivation).
4. **`evaluate_qgnn.py` metric additions**: train accuracy (currently only test-split metrics are computed — needs a pass over the train split too), wall-clock training time (already measurable by timing `train_qgnn_classifier`'s call in `experiments/train_qgnn.py`, needs to be persisted/logged, not just observed), wall-clock inference time (time the test-split forward pass in `evaluate_qgnn.py`). Report alongside the existing accuracy/AUC/macro-F1, formatted so a row can be added directly under `sota-table.png`'s Table II style in `README.md`.

Tests: extend `tests/test_qgnn_training.py` with an `AdamW`-optimizer-selection smoke test (mirrors the existing `Adam`-path smoke test); extend `evaluate_qgnn.py`'s test coverage with a train-accuracy-computation check; no new correctness-critical logic is introduced (the metric additions are read-only measurements, and `AdamW` is a drop-in `torch.optim` swap), so this task's test burden is lighter than T4.8/T4.9's.

**Explicitly not claimed:** this task does not itself run the 800-jet comparability training — it specs and (once implemented) wires up the configuration to allow that run. Recording real numbers against Table II is T4.6's job, using this task's pipeline.

---

## Deliverable

- A working `src/gvls/data/jets.py` loader producing labeled, graph-structured jet data at a documented subset size
- `results/compression/qg_jets_pooling.csv`: per-jet compression fidelity vs. fixed `M ∈ {4,6,8}`, with a chosen compression-optimal `M`
- `src/gvls/models/qgnn.py`: a tested, working Qiskit QGNN ansatz whose entangling structure is a direct function of the learned `A_z`
- A trained QGNN classifier (T4.5) with accuracy/AUC/macro-F1 reported on held-out test jets
- A literature comparison point (T4.6) — **target identified 2026-07-30**: `Lorentz-EQGNN` (`sota-table.png` Table II, 4 qubits, 74.00% test accuracy on Quark-Gluon), pending bibliographic confirmation of the source paper and pending the actual comparability run (T4.10's pipeline realignment, then T4.6's real execution)
- `README.md` updated with a new "Quantum Graph Neural Network — Quark/Gluon Jet Classification" results section, following this repo's existing convention (numbers, findings bullets, a plot if one is informative)
- `specs/phase4/validation.md` populated with the actual results and any bugs/surprises found along the way, mirroring Phases 0–3's validation-doc convention
- (T4.8) Batched QGNN training with the gradient-parity tests passing and an actual before/after wall-clock comparison recorded — done; the measured ~1.34x speedup on its own (mostly from `input_gradients=False`, not batching itself, `validation.md` V-8) was insufficient, which motivated T4.9
- (T4.9) SPSA gradient estimator replacing parameter-shift as the default — measured ~15.5x combined speedup (T4.8+T4.9) over the pre-T4.8 original on synthetic jets, the first change in this investigation that plausibly makes the target-scale real run tractable; not yet confirmed at that real scale (`validation.md` V-9)
- (T4.10) Pipeline realigned with the Lorentz-EQGNN literature baseline (`AdamW`, `lr=1e-3`, `batch_size=16`, `800`-jet comparability subset, train/test-accuracy + wall-clock reporting) — specced 2026-07-30, not yet implemented or run (`validation.md` V-10)
