# Phase 1 — Plan

## Objective

Build the core GVLS model end-to-end: a variational GNN encoder that produces per-node Gaussian distributions, a differentiable latent graph inference module, latent message passing over the inferred graph, and a graph-aware ELBO loss. The phase ends when GVLS trains to convergence on Cora with no NaN losses and produces a latent graph A_z that is observably different from the input adjacency.

---

## Scope

### In scope
- GNN variational encoder (μ, log σ² heads)
- Three latent graph inference methods (attention, FGP, NRI) — implemented in order, tested independently
- Latent message passing over soft adjacency
- Isotropic KL baseline and graph-aware Gaussian MRF KL
- Full ELBO training loop with W&B logging and model checkpointing

### Out of scope
- Node classification or graph classification heads (Phase 2)
- Ablation sweeps across all methods (Phase 3)
- Graph compression decoder (Phase 2)
- Any baseline model re-implementation

---

## File Map

```
src/gvls/
  models/
    encoder.py          # T1.1 — GCN encoder → (μ, log_σ², z)
    latent_graph.py     # T1.2 — attention / FGP / NRI inference → A_z
    gvls.py             # T1.3 — full model wiring encoder + latent graph + message passing
  losses/
    elbo.py             # T1.4 — isotropic KL, graph-MRF KL, full ELBO
configs/
  model/
    gvls.yaml           # T1.4 — latent_dim, mp_rounds, graph_method, prior, β, λ, k
experiments/
  train_gvls.py         # T1.4 — training loop
tests/
  test_encoder.py       # T1.1
  test_latent_graph.py  # T1.2
  test_gvls.py          # T1.3
  test_elbo.py          # T1.4
```

---

## Tasks

### T1.1 — Variational Encoder

**File:** `src/gvls/models/encoder.py`

Implement `GVLSEncoder(in_channels, hidden_channels, latent_dim)`:
- Layer 1: `GCNConv(in_channels, hidden_channels)` + ReLU — shared
- Layer 2a: `GCNConv(hidden_channels, latent_dim)` → μ
- Layer 2b: `GCNConv(hidden_channels, latent_dim)` → log_σ²
- log_σ² clamped to [−10, 10]
- `reparameterize(mu, log_var)` returns z = μ + σ·ε at train time, μ at eval time
- `forward(x, edge_index)` returns `(mu, log_var, z)`

Tests (`tests/test_encoder.py`):
- Output shapes are `(N, latent_dim)` for all three outputs
- Gradients reach `layer1.weight` after backward on a scalar loss
- In eval mode, `z == mu` (no sampling noise)
- log_σ² is clamped: input log_var of ±100 maps to ±10

---

### T1.2 — Latent Graph Inference

**File:** `src/gvls/models/latent_graph.py`

Implement `LatentGraphLearner(latent_dim, method, k)`:
- All methods return A_z ∈ ℝ^(N×N) with values in [0,1], diagonal zeroed, top-k sparsified, symmetrized
- `method='attention'`: dot-product similarity scaled by √d, passed through sigmoid
- `method='fgp'`: cosine similarity / τ (τ initialized to 1, learned), through sigmoid
- `method='nri'`: MLP on concatenated pairs (z_i ‖ z_j) → scalar logit → sigmoid; MLP is (2d → d → 1)
- Top-k sparsification: zero out all entries except the k largest per row, then re-symmetrize
- `forward(z)` returns A_z

Tests (`tests/test_latent_graph.py`):
- Output shape is `(N, N)` for all three methods
- Values are in [0, 1]
- Diagonal is zero
- A_z is symmetric
- At most k non-zero entries per row after sparsification
- Gradients reach learner parameters for all three methods

---

### T1.3 — Full Model and Latent Message Passing

**File:** `src/gvls/models/gvls.py`

Implement `GVLS(encoder, latent_graph_learner, latent_dim, mp_rounds)`:
- `forward(x, edge_index)`:
  1. `mu, log_var, z = encoder(x, edge_index)`
  2. `A_z = latent_graph_learner(z)`
  3. L rounds of message passing: z̃ = ReLU(D^{-1} A_z Z̃ W), W ∈ ℝ^(d×d), initialized as identity; z̃₀ = z
  4. Return `(mu, log_var, z, A_z, z_tilde)`
- Degree normalization: D_ii = max(Σ_j A_z[i,j], ε) to avoid division by zero (ε=1e-8)
- `mp_rounds=0` is valid: z̃ = z (skip message passing entirely)

Tests (`tests/test_gvls.py`):
- Full forward pass on a small synthetic graph produces correct output shapes
- `mp_rounds=0` gives z̃ == z
- Gradients reach encoder layer 1 after backward through z̃

---

### T1.4 — ELBO Loss and Training Script

**File:** `src/gvls/losses/elbo.py`

Implement:
- `kl_isotropic(mu, log_var) → scalar`: -0.5 · Σ (1 + log_var − μ² − exp(log_var))
- `kl_graph_mrf(mu, log_var, A_z, lambda_) → scalar`: closed-form KL against Gaussian MRF prior with precision Ω = I + λ·L_z
  - L_z = D_z − A_z (graph Laplacian of A_z)
  - For diagonal q: KL = 0.5 · [tr(Ω·Σ_q) + μ^T Ω μ − N·d + log det(Ω) − Σ log_var]
  - log det(Ω) computed via `torch.linalg.slogdet`; detach A_z for this term to avoid second-order gradients through the Laplacian determinant
- `elbo(recon_logits, adj_true, mu, log_var, A_z, beta, lambda_, prior) → scalar`: recon_loss − β·KL
  - `recon_logits`: inner-product z̃_i · z̃_j for all node pairs; `adj_true`: dense binary adjacency
  - `recon_loss`: mean BCE over all node pairs (positive and negative equally weighted)
  - `prior` ∈ {'isotropic', 'graph_mrf'}

NaN guard: after computing ELBO, if `torch.isnan(loss)`, raise `RuntimeError("NaN loss detected — check hyperparameters")`

**File:** `configs/model/gvls.yaml`
```yaml
latent_dim: 32
hidden_dim: 64
mp_rounds: 1
graph_method: attention   # attention | fgp | nri
prior: isotropic          # isotropic | graph_mrf
k: 10                     # top-k sparsification
beta: 1.0
lambda_: 1.0
```

**File:** `experiments/train_gvls.py`
- Load dataset and split edges (reuse Phase 0 utilities)
- Instantiate `GVLS` from config
- Adam optimizer, lr and epochs from `configs/train/default.yaml`
- Each epoch: forward pass, compute ELBO, backward, optimizer step
- Log to W&B: `train/elbo`, `train/kl`, `train/recon`, `val/auc`, `val/ap`, `latent/density`
- Save best checkpoint (by `val/auc`) to `checkpoints/best.pt`

Tests (`tests/test_elbo.py`):
- `kl_isotropic` returns 0 when mu=0, log_var=0
- `kl_isotropic` is positive for non-zero mu
- `kl_graph_mrf` is finite on valid inputs (positive semi-definite A_z)
- NaN guard fires when loss is manually set to NaN

---

## Deliverable

`experiments/train_gvls.py` runs to completion on Cora (default config, 80% split) with:
- No NaN or Inf losses at any epoch
- ELBO decreasing over the first 50 epochs
- A_z density logged; value must differ from the density of the input adjacency by at least 0.05
- Val AUC > 0.6 after 200 epochs (sanity threshold — not the paper target)
