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

### Link Prediction — AUC

Baselines from Ahn & Kim, *"Variational Graph Normalized Autoencoders"*, CIKM 2021.

| Dataset | Train | GAE | LGAE | ARGA | GIC | sGraph | GNAE | VGNAE | **GVLS** |
|---------|-------|-----|------|------|-----|--------|------|-------|----------|
| Cora | 20% | 0.782 | 0.866 | 0.795 | 0.880 | 0.845 | 0.887 | **0.890** | — |
| Cora | 40% | 0.856 | 0.908 | 0.844 | 0.914 | 0.840 | 0.926 | **0.929** | — |
| Cora | 80% | 0.922 | 0.938 | 0.919 | 0.933 | 0.885 | **0.956** | 0.954 | **0.917** |
| CiteSeer | 20% | 0.786 | 0.906 | 0.750 | 0.930 | 0.928 | **0.946** | 0.941 | — |
| CiteSeer | 40% | 0.836 | 0.925 | 0.832 | 0.936 | 0.936 | 0.956 | **0.961** | — |
| CiteSeer | 80% | 0.894 | 0.955 | 0.904 | 0.962 | 0.963 | 0.965 | **0.970** | — |
| PubMed | 20% | 0.937 | 0.946 | 0.936 | 0.950 | 0.837 | 0.950 | **0.951** | — |
| PubMed | 40% | 0.959 | 0.962 | 0.955 | 0.958 | 0.876 | 0.963 | **0.964** | — |
| PubMed | 80% | 0.967 | 0.974 | 0.973 | 0.960 | 0.896 | 0.975 | **0.976** | — |

### Link Prediction — AP

| Dataset | Train | GAE | LGAE | ARGA | GIC | sGraph | GNAE | VGNAE | **GVLS** |
|---------|-------|-----|------|------|-----|--------|------|-------|----------|
| Cora | 20% | 0.793 | 0.878 | 0.806 | 0.881 | 0.829 | **0.901** | **0.901** | — |
| Cora | 40% | 0.861 | 0.915 | 0.856 | 0.911 | 0.828 | **0.936** | 0.933 | — |
| Cora | 80% | 0.930 | 0.945 | 0.927 | 0.929 | 0.867 | 0.957 | **0.958** | **0.909** |
| CiteSeer | 20% | 0.797 | 0.913 | 0.777 | 0.934 | 0.897 | **0.953** | 0.948 | — |
| CiteSeer | 40% | 0.850 | 0.929 | 0.844 | 0.938 | 0.910 | 0.958 | **0.966** | — |
| CiteSeer | 80% | 0.903 | 0.959 | 0.915 | 0.966 | 0.943 | 0.970 | **0.971** | — |
| PubMed | 20% | 0.940 | 0.947 | 0.941 | 0.947 | 0.859 | **0.950** | 0.949 | — |
| PubMed | 40% | 0.961 | 0.961 | 0.959 | 0.956 | 0.879 | 0.961 | **0.963** | — |
| PubMed | 80% | 0.967 | 0.975 | **0.976** | 0.965 | 0.902 | 0.975 | **0.976** | — |

> GVLS Cora 80% uses the NAS best config: FGP / isotropic / latent_dim=128 / hidden_dim=256 / β=1.9e-5. CiteSeer and PubMed results pending.

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
