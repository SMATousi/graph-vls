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

## Phase 2 — Architecture Search (NAS) ✅ Completed 2026-06-11

**Goal:** Find the best GVLS configuration for each benchmark dataset using Optuna.

### Tasks
- [x] T2.1 — Search space helpers and NAS Hydra config
- [x] T2.2 — Optuna objective function (one trial = train GVLS, return best val AUC)
- [x] T2.3 — NAS entry point with TPE sampler, MedianPruner, SQLite storage, W&B summary
- [x] T2.4 — Run 50-trial search on Cora, CiteSeer, and PubMed; best configs saved for all three

**Search space:** latent_dim ∈ {16,32,64,128}, hidden_dim ∈ {32,64,128,256}, mp_rounds ∈ {0,1,2}, graph_method ∈ {attention, fgp}, k ∈ {5,10,20,50}, prior ∈ {isotropic, graph_mrf}, lr ∈ [1e-4, 5e-2], beta ∈ [1e-5, 0.1].

**Exit criterion met:** Cora — 53 trials (26 completed, 27 pruned), best val AUC=0.9438, `configs/best/cora.yaml` retrained within ±0.02 (val AUC=0.9297, test AUC=0.917). CiteSeer — 50 trials (41 completed, 9 pruned), best val AUC=0.9407, `configs/best/citeseer.yaml` written. PubMed — 51 trials (30 completed, 20 pruned, 1 failed), best val AUC=0.9518, `configs/best/pubmed.yaml` written. All three configs retrained across 20/40/80% splits; results in `README.md` and `reports/midterm_report.md`. 84/84 tests pass; `ruff check src/` clean. See `specs/phase2/validation.md` for full detail.

**Key finding:** FGP cosine similarity (k=20) with an isotropic prior is consistently preferred for Cora and CiteSeer; the graph-MRF prior only helps on PubMed, where the graph is larger and richer. PubMed link prediction at low split ratios (20%) still trails baselines (0.835 vs. VGNAE 0.951) — flagged for ablation follow-up in Phase 4, not a Phase 2 blocker.

---

## Phase 3 — Graph Compression (priority) and Node Classification (Weeks 12–16)

**Goal:** Quantify how compact (z̃, A_z) is relative to the input graph, and trace a rate-distortion curve across latent dimension `d`, latent-graph sparsity `k`, and — as of the 2026-07-09 reframing — latent node count `M`. This is the direct prerequisite for the planned QGNN integration (mission.md, `reports/midterm_report.md` §6), which consumes (z̃, A_z). See `specs/phase3/` for the full plan, requirements, and validation criteria.

**Reprioritized 2026-07-07:** graph compression is now the phase's primary deliverable, ahead of node classification and graph-level tasks. Link prediction across all splits (20/40/80%) is **already complete** — it was run as an extension of Phase 2 using the NAS-best configs; results are in `README.md` and `reports/midterm_report.md`. It is not repeated in Phase 3.

**Reframed 2026-07-09:** the T3.1–T3.3 sweep only ever varied `d` and `k` at a fixed node count (`M=N`) — the latent graph always had as many nodes as the input. That's no longer sufficient: the goal is now a latent graph that is smaller than the input in **both** node count and edge count, not just per-node dimensionality. T3.4 (a decoder tweak at fixed `M=N`) is superseded by T3.6, a learned pooling mechanism that actually reduces `M`. See `mission.md` changelog and `specs/roadmap.md` Phase 3 tasks below.

### Graph Compression (priority)
- [x] T3.1 — Compression metrics: `reconstruction_f1`, `dim_compression_ratio` (d/F), `edge_compression_ratio` (\|A_z\|/\|E\|), sampled `bits_per_edge` for large graphs
- [x] T3.2 — Full-graph split mode (`train_ratio=1.0`, no held-out edges — fidelity is measured on what was actually encoded, not generalization to unseen edges)
- [x] T3.3 — Rate-distortion sweep: `latent_dim ∈ {4,8,16,32,64,128}` × `k ∈ {1,2,3,5,10,20}`, other hyperparameters fixed to each dataset's Phase 2 NAS-best config; compression-optimal configs written to `configs/compression/{dataset}.yaml`. **All three datasets done** (Cora 2026-07-07; PubMed 2026-07-08, remote A100; CiteSeer 2026-07-09). None meet the 0.90 fidelity floor anywhere in their grids: Cora flat at 0.81–0.83; PubMed *decreases* with capacity (0.742→0.673 as `d`: 4→128 at `k=20`, worst point = PubMed's own NAS-best config); CiteSeer's `A_z` is provably inert for its NAS-best config (`mp_rounds=0, prior=isotropic` gives it no path into the loss). Results in `README.md`, `results/compression/{cora,citeseer,pubmed}.csv`. This entire sweep was run at `M=N` (no node-count reduction) — see T3.6.
- [x] ~~T3.4 — A_z-conditioned decoder~~ **Superseded 2026-07-09.** Triggered for all three datasets (F1 < 0.90 at largest tested capacity everywhere), but set aside in favor of T3.6: the low F1 ceiling was measured entirely at `M=N` (no node-count reduction), and the project's compression goal was reframed to prioritize shrinking node count over refining the decoder at a fixed size. Revisit only if T3.6's pooling approach fails to close the fidelity gap on its own.
- [ ] T3.6 — **Node-count compression via learned pooling (new priority, reframes the compression goal).** A DiffPool-style soft assignment `S ∈ [0,1]^{N×M}` pools the `N` encoder-level Gaussians into `M ≪ N` latent Gaussians; the latent graph `A_z` is learned over these `M` pooled nodes; reconstruction unpools back to the full `N×N` adjacency via the same `S` (`Â = S·σ(z̃_pooled z̃_pooledᵀ)·Sᵀ`). `M` is swept as a ratio of `N` (`{0.5, 0.25, 0.125, 0.0625}`), holding `(d, k)` fixed at each dataset's T3.3 compression-optimal config, to isolate node-count compression's effect on fidelity independent of the dimensionality/edge-sparsity axes already explored. This directly reverses `mission.md`'s prior "not a hierarchical pooling model" stance (see mission.md changelog, 2026-07-09). See `specs/phase3/` for full task detail.

### Node Classification (secondary)
- [ ] T3.5 — Linear probe and 2-layer MLP head on frozen z̃ (Phase 2 NAS-best config), semi-supervised setting (20 labels/class); may slip to Phase 4 if compression work runs long

### Graph-Level Tasks (deferred)
- [ ] Global pooling (mean/sum/attention) over z̃ → graph embedding
- [ ] Graph classification head; evaluate on MUTAG, PROTEINS, IMDB-B
- Deferred out of Phase 3 — not connected to the compression/QGNN priority; picked up in Phase 4 or later if time allows

**Exit criterion:** rate-distortion sweep complete on all three datasets; a compression-optimal config identified per dataset with `d/F`, `|A_z|/|E|`, `M/N`, reconstruction F1, and bits-per-edge all reported against the input graph's raw size. See `specs/phase3/validation.md`.

---

## Phase 4 — Ablations, Analysis, and Paper (Weeks 17–22)

**Goal:** Produce results suitable for a research submission.

### Ablations
- [ ] Flat vs. graph-structured prior (isotropic vs. graph-MRF)
- [ ] Latent graph inference method (attention vs. FGP vs. NRI)
- [ ] Number of latent message-passing rounds (0, 1, 2)
- [ ] Effect of β (KL weight) on downstream task performance
- [ ] Input topology vs. learned latent topology (overlap analysis)
- [ ] Pooled node count `M` vs. fidelity/downstream performance (extends T3.6's compression-focused sweep to link prediction and node classification)
- [ ] Soft (DiffPool-style) vs. hard (top-k / Gumbel-softmax) node assignment for pooling

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
- **(New, 2026-07-11, evidence added 2026-07-13)** Phase 2's NAS selected each dataset's `β` (and PubMed's `prior=graph_mrf`) against a KL term that turned out to be an un-normalized sum over node count, since fixed (`src/gvls/losses/elbo.py`, see `specs/phase3/validation.md` V-8) to be normalized by node count instead — a change that alters the loss landscape `elbo()` presents to any training run, not just Phase 3's pooling sweeps. Should Phase 2's NAS search be re-run under the corrected convention, particularly for PubMed, whose NAS-best `β` was calibrated to the old (much larger) KL scale? Supporting evidence for taking this seriously: rerunning just T3.3's PubMed `(d,k)` compression grid under the fix reversed its entire capacity trend (F1 by `d`: 0.742→0.673 decline became 0.7725→0.7771 rise) and changed which `(d,k)` point wins selection — if a *fixed*-`(prior,β)` sweep moves this much, it's plausible the NAS search that chose `(prior,β)` itself would land somewhere different too.
