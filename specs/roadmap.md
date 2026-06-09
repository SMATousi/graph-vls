# Roadmap

The project is organized into four phases. Each phase has a concrete deliverable that gates the next.

---

## Phase 0 — Foundation (Weeks 1–2)

**Goal:** Reproducible baselines and a clean project skeleton.

### Tasks
- [ ] Set up project structure: `src/`, `experiments/`, `data/`, `tests/`
- [ ] Data loading pipeline for Cora, CiteSeer, PubMed (PyG `Planetoid`) and TU datasets (`TUDataset`)
- [ ] Implement and validate a standard VGAE baseline (encoder: 2-layer GCN → mean/logvar; decoder: inner product)
- [ ] Implement evaluation metrics: AUC/AP for link prediction, accuracy for node classification, bits-per-edge for compression
- [ ] Experiment config system (Hydra) and logging (Weights & Biases)

**Exit criterion:** VGAE baseline reproduces published numbers on Cora link prediction (AUC ≥ 0.91).

---

## Phase 1 — Core Architecture (Weeks 3–8)

**Goal:** A working GVLS model that trains end-to-end.

### 1a — Variational Encoder
- [ ] GNN encoder maps node features → (μ, log σ²) per node
- [ ] Reparameterization trick produces latent samples z ∈ ℝ^(N×d)
- [ ] Verify KL divergence with isotropic prior as a baseline

### 1b — Latent Graph Inference Module
- [ ] Implement a differentiable latent graph learner that takes latent embeddings z and outputs a soft adjacency A_z ∈ ℝ^(N×N)
- [ ] Candidate approaches (implement in order, compare):
  - **Attention-based**: A_z[i,j] = softmax(e_i · e_j / √d) — simple, differentiable
  - **FGP (Feature Graph Prior)**: cosine similarity with a learned temperature
  - **NRI-style**: encode node pairs, infer edge type distribution
- [ ] Sparsification: threshold or top-k to keep A_z tractable

### 1c — Latent Message Passing
- [ ] Run 1–2 rounds of GNN message passing over (z, A_z) to produce refined latent states z̃
- [ ] z̃ is used both for decoding and for computing the graph-aware KL

### 1d — Graph-Aware Prior and ELBO
- [ ] Define a graph-structured prior p(z | A_z) — initial choice: Gaussian MRF with precision matrix Ω = I + λ·L_z, where L_z is the Laplacian of A_z
- [ ] Derive and implement the closed-form (or estimated) KL(q(z|x, A) ‖ p(z|A_z))
- [ ] Full ELBO = reconstruction loss − β·KL; expose β as a hyperparameter

**Exit criterion:** GVLS trains to convergence on Cora without NaN losses and produces a non-trivial inferred latent graph (A_z differs from the input adjacency).

---

## Phase 2 — Downstream Tasks and Decoders (Weeks 9–13)

**Goal:** Evaluate GVLS on all three target tasks.

### Link Prediction
- [ ] Inner-product decoder on z̃: σ(z̃_i · z̃_j) for edge probability
- [ ] Evaluate AUC/AP on standard train/val/test splits

### Node Classification
- [ ] Linear probe and 2-layer MLP head on z̃
- [ ] Semi-supervised setting: 20 labels/class

### Graph Compression
- [ ] Encoder compresses input graph to (z̃, A_z)
- [ ] Decoder reconstructs adjacency A from (z̃, A_z)
- [ ] Metrics: reconstruction F1, bits-per-edge (lossless approximation via entropy coding), rate-distortion curve across latent dimensions d

### Graph-Level Tasks
- [ ] Global pooling (mean/sum/attention) over z̃ → graph embedding
- [ ] Graph classification head; evaluate on MUTAG, PROTEINS, IMDB-B

**Exit criterion:** GVLS matches or exceeds VGAE on link prediction AUC on at least two datasets.

---

## Phase 3 — Ablations, Analysis, and Paper (Weeks 14–20)

**Goal:** Produce results suitable for a research submission.

### Ablations
- [ ] Flat vs. graph-structured prior (ablate the MRF prior vs. isotropic)
- [ ] Latent graph inference method (attention vs. FGP vs. NRI)
- [ ] Number of latent message-passing rounds (0, 1, 2)
- [ ] Effect of β (KL weight) on downstream task performance
- [ ] Input topology vs. learned latent topology (how much do they overlap?)

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

## Open Questions (to resolve during Phase 1)

- Should A_z be symmetric by construction, or learned asymmetrically?
- Should the latent graph inference module receive the input graph as a conditioning signal?
- Is a Gaussian MRF prior tractable at scale, or do we need an amortized / sampled approximation?
- How to handle dynamic-sized graphs (variable N) cleanly with batched training?
