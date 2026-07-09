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
| Cora | 20% | 0.782 | 0.866 | 0.795 | 0.880 | 0.845 | 0.887 | **0.890** | 0.870 |
| Cora | 40% | 0.856 | 0.908 | 0.844 | 0.914 | 0.840 | 0.926 | **0.929** | 0.887 |
| Cora | 80% | 0.922 | 0.938 | 0.919 | 0.933 | 0.885 | **0.956** | 0.954 | 0.917 |
| CiteSeer | 20% | 0.786 | 0.906 | 0.750 | 0.930 | 0.928 | **0.946** | 0.941 | 0.941 |
| CiteSeer | 40% | 0.836 | 0.925 | 0.832 | 0.936 | 0.936 | 0.956 | **0.961** | 0.942 |
| CiteSeer | 80% | 0.894 | 0.955 | 0.904 | 0.962 | 0.963 | 0.965 | **0.970** | 0.929 |
| PubMed | 20% | 0.937 | 0.946 | 0.936 | 0.950 | 0.837 | 0.950 | **0.951** | 0.835 |
| PubMed | 40% | 0.959 | 0.962 | 0.955 | 0.958 | 0.876 | 0.963 | **0.964** | 0.884 |
| PubMed | 80% | 0.967 | 0.974 | 0.973 | 0.960 | 0.896 | 0.975 | **0.976** | 0.934 |

### Link Prediction — AP

| Dataset | Train | GAE | LGAE | ARGA | GIC | sGraph | GNAE | VGNAE | **GVLS** |
|---------|-------|-----|------|------|-----|--------|------|-------|----------|
| Cora | 20% | 0.793 | 0.878 | 0.806 | 0.881 | 0.829 | **0.901** | **0.901** | 0.876 |
| Cora | 40% | 0.861 | 0.915 | 0.856 | 0.911 | 0.828 | **0.936** | 0.933 | 0.898 |
| Cora | 80% | 0.930 | 0.945 | 0.927 | 0.929 | 0.867 | 0.957 | **0.958** | 0.909 |
| CiteSeer | 20% | 0.797 | 0.913 | 0.777 | 0.934 | 0.897 | **0.953** | 0.948 | 0.946 |
| CiteSeer | 40% | 0.850 | 0.929 | 0.844 | 0.938 | 0.910 | 0.958 | **0.966** | 0.948 |
| CiteSeer | 80% | 0.903 | 0.959 | 0.915 | 0.966 | 0.943 | 0.970 | **0.971** | 0.939 |
| PubMed | 20% | 0.940 | 0.947 | 0.941 | 0.947 | 0.859 | **0.950** | 0.949 | 0.843 |
| PubMed | 40% | 0.961 | 0.961 | 0.959 | 0.956 | 0.879 | 0.961 | **0.963** | 0.884 |
| PubMed | 80% | 0.967 | 0.975 | **0.976** | 0.965 | 0.902 | 0.975 | **0.976** | 0.923 |

> GVLS uses per-dataset NAS best configs.

### Graph Compression

Phase 3 asks a different question than link prediction: how small can (z̃, A_z) be made relative to the input graph (X, A) while still reconstructing it? A dedicated rate-distortion sweep trains a fresh model at every `latent_dim × k` grid point on the **full graph** (all edges, no held-out split — see `specs/phase3/plan.md`), independent of the AUC-optimal `k` that Phase 2's NAS chose.

#### Cora

Full results: [`results/compression/cora.csv`](results/compression/cora.csv) (36 points, 200 epochs each). Input graph: N=2708 nodes, F=1433 features, |E|=5278 edges.

| d | k | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|-----|------------|-----|-----------|
| 4 | 1 | 0.0028 | 0.368 | 0.815 | 1.088 |
| 8 | 1 | 0.0056 | 0.377 | 0.825 | 1.077 |
| 16 | 1 | 0.0112 | 0.372 | 0.822 | 1.091 |
| 32 | 1 | 0.0223 | 0.367 | 0.813 | 1.146 |
| 64 | 1 | 0.0447 | 0.368 | 0.815 | 1.224 |
| 128 | 1 | 0.0893 | 0.368 | 0.820 | 1.412 |
| 16 | 20 | 0.0112 | 7.369 | **0.828** | 1.094 |
| 128 | 20 | 0.0893 | 7.490 | 0.823 | 1.381 |

**Findings:**
- **`k` controls compression, not `d`.** At `k=1`, A_z has ~37% as many edges as the input graph — genuine structural compression — while every `k≥2` makes A_z *denser* than the input (up to 7.5× at `k=20`, the NAS-best value chosen for link-prediction AUC, not compression). This confirms the concern flagged in `specs/phase3/plan.md`: an AUC-optimal `k` and a compression-optimal `k` are different numbers.
- **Reconstruction F1 is flat across the entire grid** — 0.813 to 0.828 (a 1.5-point spread) across all 36 combinations of `d ∈ {4,…,128}` and `k ∈ {1,…,20}`. More latent capacity buys essentially nothing. This points at the plain inner-product decoder (`σ(z̃_i · z̃_j)`) as the bottleneck, not the compression ratio itself.
- **No grid point met the 0.90 fidelity floor** — even the largest tested capacity (`d=128, k=20`) only reaches F1=0.823. Per `specs/phase3/plan.md`'s T3.4 trigger condition, this **fires the conditional decoder fallback** (an explicit A_z-conditioned decoder) for Cora.
- The best trade-off is arguably **d=8, k=1**: F1=0.825 (near the grid's best) at d/F=0.56% and only 37.7% of the input's edge count — matching the best raw-F1 point (d=16, k=20, F1=0.828) almost exactly, at a fraction of the size on both axes.

#### PubMed

Full results: [`results/compression/pubmed.csv`](results/compression/pubmed.csv) (36 points, run on a remote A100 given PubMed's NAS-best `prior=graph_mrf` O(N³) KL term). Input graph: N=19717 nodes, F=500 features, |E|=44324 edges.

| d | k | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|-----|------------|-----|-----------|
| 4 | 1 | 0.0080 | 0.445 | 0.708 | 1.0000017 |
| 8 | 1 | 0.0160 | 0.444 | 0.757 | 1.0000017 |
| 16 | 1 | 0.0320 | 0.441 | 0.760 | 1.0000038 |
| 16 | 2 | 0.0320 | 0.882 | **0.761** | 1.0000073 |
| 32 | 1 | 0.0640 | 0.434 | 0.749 | 1.0000164 |
| 128 | 1 | 0.2560 | 0.422 | 0.739 | 1.0000701 |
| 128 | 20 | 0.2560 | 8.725 | 0.673 | 1.0015761 |

**Findings — a different, more concerning picture than Cora:**
- **More capacity makes reconstruction *worse*, not better.** Mean F1 falls monotonically as `k` grows (0.745 at `k=1` → 0.716 at `k=20`), and at fixed `k=20`, F1 falls monotonically as `d` grows too (0.742 at `d=4` → 0.673 at `d=128`). Averaged over `k`, F1 peaks at a modest `d=16` (0.754) and *declines* toward `d=128` (0.705) — the opposite of what you'd want from a capacity/rate-distortion curve. `d=128, k=20` happens to be exactly PubMed's Phase 2 NAS-best architecture, so the config tuned for link-prediction AUC is the **worst** point in this grid for compression fidelity.
- **`bits_per_edge` is degenerate — essentially exactly 1.0 bit at every single grid point** (1.0000017 to 1.0015761). Per Phase 0's convention, a logit of exactly 0 (maximum uncertainty) gives precisely 1.0 bit. PubMed's pair space is huge (~194M possible pairs vs. 44,324 real edges, a 0.023% positive rate), so `bits_per_edge` is estimated from a large uniform sample dominated by random, mostly-unrelated node pairs (see `dense_pair_limit`/`sample_node_pairs` in `specs/phase3/plan.md`). The near-exact 1.0 suggests the model is essentially uncertain (logit ≈ 0) about the *bulk* of random pairs — it can apparently still separate real edges from negatives well enough for F1 ≈ 0.7–0.76 on the balanced eval set, but isn't confidently negative almost anywhere else. A plausible driver: the extreme `pos_weight` this dataset's scale requires (~4384×, from `(N²−E)/E`) makes the training loss overwhelmingly dominated by getting rare positive edges right, leaving little pressure to push far-apart negative pairs to confidently negative logits. Worth investigating further, not yet root-caused.
- **`k` still controls edge compression the same way as Cora**: `k=1` gives `|A_z|` at ~42–45% of the input's edge count (genuine compression) across all `d`; `k≥2` again makes A_z denser than the input, up to 8.7× at `k=20`.
- **No grid point met the 0.90 fidelity floor** (max F1 = 0.761 at `d=16, k=2`) — same conclusion as Cora, fires the T3.4 decoder-fallback trigger, and here the case is stronger: F1 at the largest tested capacity (`d=128, k=20`) is 0.673, the *worst* point in the entire grid, not just a plateau.

#### CiteSeer

Full results: [`results/compression/citeseer.csv`](results/compression/citeseer.csv) (36 points, 200 epochs each). Input graph: N=3327 nodes, F=3703 features, |E|=4552 edges.

| d | k | d/F | \|A_z\|/\|E\| | F1 | bits/edge |
|---|---|-----|------------|-----|-----------|
| 4 | 1 | 0.0011 | 0.504 | 0.8087 | 1.1044 |
| 8 | 1 | 0.0022 | 0.499 | 0.8161 | 1.0803 |
| 16 | 1 | 0.0043 | 0.496 | 0.8187 | 1.0632 |
| 16 | 3 | 0.0043 | 1.489 | **0.8188** | 1.0632 |
| 32 | 1 | 0.0086 | 0.494 | 0.8160 | 1.0682 |
| 128 | 20 | 0.0346 | 10.451 | 0.8140 | 1.1422 |

**Findings — a third, distinct pattern:**
- **`k` has essentially zero effect on F1, for a mechanistic reason, not an empirical one.** CiteSeer's Phase 2 NAS-best config is `mp_rounds=0, prior=isotropic`. With `mp_rounds=0`, `z̃ = z` unconditionally (message passing is skipped, per `GVLS.forward` in `src/gvls/models/gvls.py`), so `A_z` never touches `z̃` or the reconstruction logits. With `prior=isotropic`, the KL term (`kl_isotropic`) only uses `μ, log σ²` — it doesn't touch `A_z` either. So for this config, **`A_z` has no path into the loss or the output at all** — varying `k` only changes a value that's computed and then discarded. Confirmed in the data: F1 is bit-identical across all 6 `k` values at `d=4, 8, 128`; at `d=16, 32, 64` there's a residual difference on the order of 1e-4, too large to be float32 rounding noise but with no plausible causal path from `k` found in the code (no stochastic ops in `LatentGraphLearner` — `topk` is deterministic, `log_tau` inits to a constant) — most likely floating-point non-determinism from parallel execution accumulating over 200 epochs, not a real effect of `k`.
- **`|A_z|/|E|` still varies with `k` exactly like Cora and PubMed** (`k=1` → ~49–50% of the input's edge count; `k≥2` → denser than input, up to 10.5× at `k=20`) — but for CiteSeer's config this is a purely decorative axis: it changes the size of a tensor that has zero effect on model behavior.
- **F1 ceiling (0.819) and range (0.809–0.819, a 1-point spread) are close to Cora's** (0.813–0.828) — both datasets plateau well short of the 0.90 floor, unlike PubMed's actively-declining curve.
- **No grid point met the 0.90 floor.** T3.4's trigger fires for CiteSeer too — the third dataset in a row. `configs/compression/citeseer.yaml` was written via the fallback (highest raw F1: `d=16, k=3`, F1=0.8188).

**Cross-dataset pattern, now that all three are in:** none of Cora, CiteSeer, or PubMed reach the 0.90 fidelity floor anywhere in their 36-point grids. Two of three NAS-best configs (CiteSeer, PubMed) use `mp_rounds=0` — meaning GVLS's core "latent message passing" mechanism is inactive in the configurations actually used for the majority of these compression runs, and for CiteSeer specifically, the entire latent graph `A_z` is provably inert. This is a strong, convergent signal that the plain inner-product decoder (or more precisely, the disconnect between `A_z` and the decoding path for 2 of 3 datasets) is the real bottleneck — see T3.4 in `specs/phase3/plan.md`.

## Usage

```bash
# Train with default config (Cora)
python experiments/train_gvls.py

# Train with NAS-found best config
python experiments/train_gvls.py model=best/cora

# Run hyperparameter search
python experiments/nas.py data=cora

# Run the graph-compression rate-distortion sweep
python experiments/compression_sweep.py data=cora
```

## Project structure

```
src/gvls/
  data/        # dataset loaders, edge splitting, full-graph split
  models/      # encoder, latent graph learner, full GVLS model
  losses/      # ELBO with isotropic and graph-MRF KL
  eval/        # link-prediction metrics + compression metrics (F1, ratios)
  nas/         # Optuna search space and objective
  compression/ # rate-distortion sweep (train/eval/select per grid point)
experiments/
  train_gvls.py         # training entry point (Hydra + W&B)
  nas.py                # NAS entry point
  compression_sweep.py  # graph-compression rate-distortion sweep
configs/
  model/best/     # NAS-found best configs per dataset (link prediction)
  compression/    # NAS-found best configs per dataset (compression)
```

Full compression results: `results/compression/{dataset}.csv`.
