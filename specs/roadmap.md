# Roadmap

The project is organized into five phases. Each phase has a concrete deliverable that gates the next.

---

## Phase 0 — Foundation ✅ Completed 2026-06-09

**Goal:** Working data and evaluation infrastructure; baselines sourced from literature, not re-implemented.

Baseline numbers are taken directly from Ahn & Kim, "Variational Graph Normalized Autoencoders," CIKM 2021, which reports GAE, LGAE, ARGA, GIC, sGraph, GNAE, and VGNAE on Cora, CiteSeer, and PubMed under 20/40/80% training ratios. VGNAE is the current best. See `specs/phase0/` for the full plan, requirements, and validation criteria.

### Tasks
- [x] Set up project structure: `src/`, `experiments/`, `data/`, `tests/`
- [x] Data loading pipeline for Cora, CiteSeer, PubMed (PyG `Planetoid`) and TU datasets (`TUDataset`)
- [x] Implement evaluation metrics: AUC/AP for link prediction, accuracy for node classification, bits-per-edge for compression
- [x] Experiment config system (Hydra) and logging (Weights & Biases)

**Exit criterion met:** 26 unit tests pass; `smoke_test.py` runs end-to-end on Cora (AUC=0.498, AP=0.493, bits_per_edge=1.000 for dummy predictor) with W&B offline logging confirmed.

---

## Phase 1 — Core Architecture ✅ Completed 2026-06-10

**Goal:** A working GVLS model that trains end-to-end.

### Tasks
- [x] T1.1 — GNN encoder → (μ, log σ², z) with reparameterization trick
- [x] T1.2 — Latent graph inference: attention, FGP, NRI with top-k union sparsification
- [x] T1.3 — Latent message passing with residual connection over inferred A_z
- [x] T1.4 — ELBO loss (isotropic and graph-MRF KL, pos_weight for class imbalance), Hydra training script, W&B logging

**Exit criterion met:** GVLS trains on Cora (80% split, 200 epochs) without NaN losses; ELBO decreases; non-trivial latent graph (density=0.007); best val_auc=0.739, test_auc=0.742. 69/69 tests pass.

**Key implementation notes** (see `specs/phase1/plan.md` for details):
- Mutual top-k intersection replaced by union symmetrization (intersection empties graph at N=2708)
- Message passing uses residual connection and no activation (ReLU breaks inner-product decoder)
- BCE uses pos_weight=(N²−E)/E; beta=0.001 to prevent KL posterior collapse

---

## Phase 2 — Architecture Search (NAS) (Weeks 9–11)

**Goal:** Find the best GVLS configuration for each benchmark dataset using Optuna.

### Tasks
- [ ] T2.1 — Search space helpers and NAS Hydra config
- [ ] T2.2 — Optuna objective function (one trial = train GVLS, return best val AUC)
- [ ] T2.3 — NAS entry point with TPE sampler, MedianPruner, SQLite storage, W&B summary
- [ ] T2.4 — Run 50-trial search on Cora; researcher runs CiteSeer and PubMed manually

**Search space:** latent_dim ∈ {16,32,64,128}, hidden_dim ∈ {32,64,128,256}, mp_rounds ∈ {0,1,2}, graph_method ∈ {attention, fgp}, k ∈ {5,10,20,50}, prior ∈ {isotropic, graph_mrf}, lr ∈ [1e-4, 5e-2], beta ∈ [1e-5, 0.1].

**Exit criterion:** 50 trials complete on Cora, at least one achieves val AUC > 0.7, `configs/best/cora.yaml` written and retrainable.

---

## Phase 3 — Downstream Tasks and Decoders (Weeks 12–16)

**Goal:** Evaluate the best GVLS configurations (from Phase 2) on all three target tasks.

### Link Prediction
- [ ] Inner-product decoder on z̃: σ(z̃_i · z̃_j) for edge probability
- [ ] Evaluate AUC/AP on standard train/val/test splits across all ratios (20/40/80%)

### Node Classification
- [ ] Linear probe and 2-layer MLP head on z̃
- [ ] Semi-supervised setting: 20 labels/class

### Graph Compression
- [ ] Encoder compresses input graph to (z̃, A_z)
- [ ] Decoder reconstructs adjacency A from (z̃, A_z)
- [ ] Metrics: reconstruction F1, bits-per-edge, rate-distortion curve across latent dimensions d

### Graph-Level Tasks
- [ ] Global pooling (mean/sum/attention) over z̃ → graph embedding
- [ ] Graph classification head; evaluate on MUTAG, PROTEINS, IMDB-B

**Exit criterion:** GVLS matches or exceeds VGAE on link prediction AUC on at least two datasets using the Phase 2 best configs.

---

## Phase 4 — Ablations, Analysis, and Paper (Weeks 17–22)

**Goal:** Produce results suitable for a research submission.

### Ablations
- [ ] Flat vs. graph-structured prior (isotropic vs. graph-MRF)
- [ ] Latent graph inference method (attention vs. FGP vs. NRI)
- [ ] Number of latent message-passing rounds (0, 1, 2)
- [ ] Effect of β (KL weight) on downstream task performance
- [ ] Input topology vs. learned latent topology (overlap analysis)

### Baselines to Compare Against
| Model | What it tests |
|---|---|
| VGAE (Kipf & Welling 2016) | Flat variational latent space |
| ARGVA | Adversarial variant, still flat |
| GraphVAE (Simonovsky & Komodakis 2018) | Graph generation, flat latent |
| S-VGAE / community VGAE variants | Structured priors but not graph latent |

### Analysis and Visualization
- [ ] Visualize inferred latent graph A_z (node color = class label); check alignment with ground truth communities
- [ ] Latent space t-SNE/UMAP colored by node class
- [ ] Rate-distortion curves: plot compression quality vs. d for VGAE and GVLS side-by-side

### Paper Deliverables
- [ ] Method section with ELBO derivation
- [ ] Experiment tables (link prediction, node classification, compression)
- [ ] Ablation table
- [ ] Latent graph visualization figure

---

## Open Questions (to resolve during Phases 2–3)

- Should the latent graph inference module receive the input graph as a conditioning signal?
- Is a Gaussian MRF prior tractable at scale, or do we need an amortized / sampled approximation?
- How to handle dynamic-sized graphs (variable N) cleanly with batched training?
- Does the NAS-found architecture for Cora transfer well to CiteSeer and PubMed, or is per-dataset tuning essential?
