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

---

## Scope

### In scope
- **T4.1** — Jet dataset loader: download/parse the Pythia8 quark/gluon dataset, build a per-jet k-NN graph (Design Decision 5), extract node features (Design Decision 6), produce a labeled train/val/test split
- **T4.2** — Inductive per-jet adaptation of the existing GVLS/`PooledGVLS` stack (Design Decisions 3, 7): a training loop that iterates jets one at a time, with fixed `M`, accumulating gradients over a minibatch
- **T4.3** — GVLS pretraining on jets: unsupervised ELBO training (reusing `src/gvls/losses/elbo.py` unchanged) across a small sweep over `M ∈ {4, 6, 8}`, reusing Phase 3's compression metrics (`reconstruction_f1`, `bits_per_edge`) computed per-jet and averaged, to pick the smallest `M` with acceptable fidelity — this phase's version of T3.3's rate-distortion sweep, at jet scale
- **T4.4** — QGNN ansatz (Design Decisions 1, 2): Qiskit circuit construction from `(z̃, A_z)`, wrapped as an `EstimatorQNN`, embedded in a `torch.nn.Module` via `TorchConnector`
- **T4.5** — Two-stage supervised training (Design Decision 8): freeze the pretrained GVLS, extract (z̃, A_z) for every jet once, train the QGNN classifier on quark/gluon labels; Hydra config + W&B logging following the existing convention (new group tag, e.g. `qgnn-jet-classification`)
- **T4.6** — Evaluation: accuracy / AUC / macro-F1 on held-out test jets; literature search to identify a comparable published QGNN result on this dataset and report against it (Design Decision 4) — the comparison target itself is a deliverable of this task, not an input to it

### Stretch / explicitly deferred
- **T4.7 (stretch)** — Joint end-to-end fine-tuning of GVLS + QGNN together (Design Decision 8), compared against the frozen-feature baseline
- Classical baselines (uncompressed classical GNN on the full jet graph; classical head on frozen z̃ with no quantum circuit) — deferred per Design Decision 4, pick up only if the literature comparison is inconclusive
- Real quantum hardware execution or noise-model simulation — Qiskit Aer noiseless statevector simulation only this phase; hardware/noise is Phase 5+ ablation material
- Any new latent-graph-inference method beyond Phase 1's existing attention/FGP/NRI — reuse whichever method Phase 2/3 already validated as a starting point
- Multi-class or full-detector jet tagging — this dataset and task are binary quark-vs-gluon only

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

## Deliverable

- A working `src/gvls/data/jets.py` loader producing labeled, graph-structured jet data at a documented subset size
- `results/compression/qg_jets_pooling.csv`: per-jet compression fidelity vs. fixed `M ∈ {4,6,8}`, with a chosen compression-optimal `M`
- `src/gvls/models/qgnn.py`: a tested, working Qiskit QGNN ansatz whose entangling structure is a direct function of the learned `A_z`
- A trained QGNN classifier (T4.5) with accuracy/AUC/macro-F1 reported on held-out test jets
- A literature comparison point (T4.6) — either a genuine published QGNN benchmark number for this dataset, or an explicit, honest statement that none was found
- `README.md` updated with a new "Quantum Graph Neural Network — Quark/Gluon Jet Classification" results section, following this repo's existing convention (numbers, findings bullets, a plot if one is informative)
- `specs/phase4/validation.md` populated with the actual results and any bugs/surprises found along the way, mirroring Phases 0–3's validation-doc convention
