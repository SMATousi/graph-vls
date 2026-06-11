# Graph Variational Latent Space (GVLS)

A variational autoencoder where the latent space is **graph-structured** rather than flat Euclidean. Instead of mapping each node to an independent Gaussian, GVLS infers a sparse graph over the latent embeddings and refines them via message passing — learning relational structure at the latent level.

![GVLS Pipeline](GVLS-Pipeline.png)

## How it works

1. **GCN Encoder** — two-layer GCN reads node features and the input graph, producing per-node mean μ and log-variance log σ² in a latent space. Samples z via reparameterization.
2. **Latent Graph Learner** — builds a sparse adjacency A_z over the latent vectors using pairwise similarity (attention, FGP cosine, or NRI), keeping the top-k neighbors per node.
3. **Latent Message Passing** — one round of diffusion on A_z refines z into z̃, letting nodes aggregate information from their latent neighbors.
4. **Inner-Product Decoder** — reconstructs the adjacency as Â = σ(z̃ z̃ᵀ).
5. **ELBO Loss** — reconstruction BCE + β·KL, with optional graph-MRF prior that encodes the latent graph structure into the regularization term.

## Results

| Dataset | Val AUC | Test AUC | Best config |
|---------|---------|----------|-------------|
| Cora    | 0.944   | 0.917    | FGP / isotropic / latent_dim=128 |

## Usage

```bash
# Train with default config (Cora)
python experiments/train_gvls.py

# Train with NAS-found best config
python experiments/train_gvls.py model=best/cora

# Run hyperparameter search
python experiments/nas.py data=cora
```

## Project structure

```
src/gvls/
  data/        # dataset loaders and edge splitting
  models/      # encoder, latent graph learner, full GVLS model
  losses/      # ELBO with isotropic and graph-MRF KL
  eval/        # AUC-ROC and average precision metrics
  nas/         # Optuna search space and objective
experiments/
  train_gvls.py   # training entry point (Hydra + W&B)
  nas.py          # NAS entry point
configs/
  model/best/     # NAS-found best configs per dataset
```
