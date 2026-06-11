# Phase 1 — Requirements

## Functional Requirements

### FR-1: GNN Encoder
- Must accept node features X ∈ ℝ^(N×F) and `edge_index` as input
- Must output per-node mean μ ∈ ℝ^(N×d) and log-variance log_σ² ∈ ℝ^(N×d)
- Architecture: 2-layer GCN with a shared first layer, then two parallel linear heads for μ and log_σ²
- Reparameterization: z = μ + σ·ε, ε ~ N(0, I); must be used at train time, μ at eval time
- Latent dimension `d` must be configurable

### FR-2: Latent Graph Inference
- Must accept z ∈ ℝ^(N×d) and output a soft adjacency A_z ∈ ℝ^(N×N) with values in [0, 1]
- Must be fully differentiable end-to-end (gradients flow through A_z into the encoder)
- Must be symmetric by construction: A_z = (A_raw + A_raw^T) / 2
- Three methods implemented in order; method selected via config:
  1. **attention**: A_z[i,j] = sigmoid(z_i · z_j / √d)
  2. **fgp** (Feature Graph Prior): A_z[i,j] = sigmoid(cos_sim(z_i, z_j) / τ), τ learned
  3. **nri**: encode node pairs (z_i ‖ z_j) through a 2-layer MLP, output Bernoulli logit per pair
- Sparsification: keep top-k edges per row (k configurable; default k=10); diagonal zeroed out
- Must log mean graph density (fraction of non-zero entries in A_z) to W&B each epoch

### FR-3: Latent Message Passing
- Must accept z ∈ ℝ^(N×d) and A_z ∈ ℝ^(N×N) and return refined z̃ ∈ ℝ^(N×d)
- Must perform L rounds of message passing over the soft adjacency (L configurable; default L=1)
- Message passing: z̃ = ReLU(D^{-1} A_z Z W), where D is the degree matrix of A_z and W ∈ ℝ^(d×d)
- z̃ is the output used for both decoding and KL computation

### FR-4: ELBO Loss
- Reconstruction term: BCE between inner-product decoder output σ(z̃_i · z̃_j) and true adjacency; use `bits_per_edge` from Phase 0 for logging, raw BCE sum for optimization
- KL term: two modes selected via config:
  - **isotropic**: KL(q ‖ N(0,I)) = -0.5 · Σ (1 + log_σ² − μ² − σ²)
  - **graph_mrf**: KL(q ‖ Gaussian MRF) using precision Ω = I + λ·L_z, where L_z is the Laplacian of A_z; use the closed-form KL for multivariate Gaussians with diagonal q
- Full ELBO: L = recon_loss − β · KL; β and λ configurable (defaults β=1.0, λ=1.0)
- Must detect NaN loss and raise a RuntimeError with a descriptive message before the optimizer step

### FR-5: Training Script
- `experiments/train_gvls.py`: end-to-end training loop for GVLS on a single dataset
- Must log per-epoch: train ELBO, KL, recon loss, val AUC/AP, latent graph density
- Must checkpoint the best model (by val AUC) to `checkpoints/`
- Must be fully Hydra-configured; new config group `configs/model/gvls.yaml`

---

## Non-Functional Requirements

### NFR-1: Shape and gradient tests
- Every module (encoder, latent graph, message passing, ELBO) must have at least one test verifying output shapes and that gradients reach the encoder parameters

### NFR-2: Numerical stability
- No NaN or Inf in forward pass on standard Cora input at default hyperparameters
- log_σ² must be clamped to [−10, 10] before exponentiation

### NFR-3: Code style
- `ruff check src/` passes with zero warnings after each task
- New model code lives under `src/gvls/models/`; loss code under `src/gvls/losses/`

### NFR-4: Reproducibility
- Training run with fixed seed must produce the same loss trajectory across runs (set `torch.manual_seed` and `torch.use_deterministic_algorithms(True)` at start of training script)

---

## New Dependencies

No new top-level dependencies required; all Phase 1 components use PyTorch and PyG primitives already installed.

`torch_geometric.nn.GCNConv` is used for the encoder and latent message passing layers.
