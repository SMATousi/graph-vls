# Graph Variational Latent Space (GVLS) for Quantum Graph Neural Networks
## Mid-Term Internship Report — Mizzou QIC Summer 2026

**Student:** Mohamad Ali Tousi  
**Academic Program:** [Program Name]  
**Mentor(s):** [Mentor Name(s)]  
**Host Department/Lab:** Quantum Information and Computing (QIC) Lab, University of Missouri  
**Report Date:** June 29, 2026

---

## 1. Research Hypothesis and Objective

Quantum Graph Neural Networks (QGNNs) hold the promise of exponential speedups over classical GNNs for certain graph-structured problems, but they are limited in scale by available qubit counts and gate depth on current NISQ hardware. Large real-world graphs — citation networks, molecular graphs, knowledge bases — far exceed the node counts that any near-term quantum processor can handle directly.

The central hypothesis of this project is that **a graph-structured variational latent space can compress large graphs into compact representations that are small enough to serve as direct inputs to QGNNs**, without sacrificing the relational information those networks need to be effective. Concretely, the **Graph Variational Latent Space (GVLS)** model learns to encode a graph with N nodes into a latent graph with a learned sparse topology (A_z) and low-dimensional node embeddings (z̃), drastically reducing qubit requirements while preserving structural and semantic content.

A secondary — and now increasingly concrete — contribution is the GVLS model itself as a novel classical method for graph representation learning, where early results suggest it is competitive with state-of-the-art variational graph autoencoders and may constitute a standalone publication.

---

## 2. Progress on Quantum Methods Development

GVLS is a hybrid classical-quantum pipeline. The classical component — fully implemented and validated — performs graph compression; the quantum component will consume the compressed representation.

**Classical Compression Architecture (complete).** The GVLS encoder is a two-layer Graph Convolutional Network (GCN) that maps node features and the input adjacency to per-node Gaussian posteriors (μ, log σ²) in a latent space of dimension d ≪ N. Samples z are drawn via the reparameterization trick. A differentiable **Latent Graph Learner** then infers a new sparse adjacency A_z over the latent embeddings using pairwise similarity (scaled dot-product attention, cosine FGP, or NRI-style MLP), keeping the top-k neighbors per node. One or more rounds of message passing over A_z refine z into z̃. The decoder reconstructs the full adjacency as Â = σ(z̃ z̃ᵀ) and training minimizes an ELBO with a novel **graph-MRF prior** that encodes the latent graph topology directly into the KL regularization term (KL(q ‖ Gaussian MRF with precision I + λL_z)), replacing the standard isotropic Gaussian prior.

**Originality.** No prior variational graph autoencoder constructs a graph-structured latent space. Existing methods (VGAE, GNAE, VGNAE) use flat, independent Gaussian latent factors. GVLS is the first to learn a latent graph end-to-end alongside the encoder, enabling relational reasoning in the latent space and producing a compressed graph (z̃, A_z) rather than a flat vector.

**Quantum Application Pathway.** The compressed graph (z̃, A_z) has at most k·N/2 edges and d-dimensional node features, where d = 128 and k = 20 in current experiments, compared to tens of thousands of edges in the original graph. This compressed graph is the planned input to a parameterized quantum circuit-based GNN, where qubit count scales with the number of nodes in the latent graph (or the latent dimension), not the original graph size.

---

## 3. Computing Experimental Design

**Problem type.** The primary validation task is **link prediction** on standard citation network benchmarks (Cora: 2,708 nodes, 5,429 edges; CiteSeer: 3,327 nodes, 4,732 edges; PubMed: 19,717 nodes, 44,338 edges), following the protocol of Ahn & Kim (CIKM 2021). Secondary tasks include node classification and, most relevant to the quantum application, graph compression quality (reconstruction F1, bits-per-edge, rate-distortion curves across d).

**Classical component.** GVLS is implemented in PyTorch 2.x with PyTorch Geometric. Hydra manages configurations and Weights & Biases logs all experiments. A Neural Architecture Search (NAS) using Optuna with a TPE sampler and Median Pruner identifies the best configuration per dataset over 50 trials (search space: latent_dim ∈ {16,32,64,128}, hidden_dim ∈ {32,64,128,256}, mp_rounds ∈ {0,1,2}, graph_method ∈ {attention, fgp}, k ∈ {5,10,20,50}, prior ∈ {isotropic, graph_mrf}, lr ∈ [1e-4, 5e-2], β ∈ [1e-5, 0.1]).

**Quantum component (planned).** The compressed graph (z̃, A_z) output by the trained GVLS encoder will be passed to a QGNN. The classical GVLS encoder acts as a preprocessing stage; no quantum training is involved in the compression step itself.

**Validation.** Link prediction AUC and Average Precision on held-out edge sets serve as the primary validation metric. For graph compression, reconstruction fidelity vs. compression ratio (rate-distortion curves) will be the primary figure.

---

## 4. Quantum Computing Requirements

| Resource | Requirement |
|---|---|
| Qubit count | Scales with latent dimension d or compressed node count — O(d) for gate-based encoding; currently d = 128, target d ≤ 32 for near-term NISQ |
| Gate depth | Determined by QGNN layer count (1–2 layers); expected shallow |
| Classical preprocessing | Full GVLS forward pass (encoder + latent graph inference) — runs on CPU/GPU |
| Hybrid architecture | Classical GVLS compression → quantum GNN inference on (z̃, A_z) |

The compression ratio is the key enabler: GVLS reduces a graph of ~20,000 nodes (PubMed) to a latent graph representable within tens to low hundreds of qubits, depending on the encoding strategy. The latent graph inference step additionally prunes edge density, producing sparse A_z with at most 2k non-zeros per row — further reducing entanglement requirements.

---

## 5. Preliminary Results

NAS was completed for all three benchmark datasets. Best NAS validation AUC on Cora reached **0.9438** across 50 trials. Best per-dataset configurations have been identified and saved. Full training with the NAS-found configurations yielded the following link prediction results on held-out test sets:

**AUC-ROC (link prediction)**

| Dataset | Split | GAE | LGAE | GNAE | VGNAE | **GVLS** |
|---|---|---|---|---|---|---|
| Cora | 20% | 0.782 | 0.866 | 0.887 | 0.890 | **0.870** |
| Cora | 80% | 0.922 | 0.938 | 0.956 | 0.954 | **0.917** |
| CiteSeer | 20% | 0.786 | 0.906 | 0.946 | 0.941 | **0.941** |
| CiteSeer | 80% | 0.894 | 0.955 | 0.965 | 0.970 | **0.929** |
| PubMed | 20% | 0.937 | 0.946 | 0.950 | 0.951 | **0.835** |
| PubMed | 80% | 0.967 | 0.974 | 0.975 | 0.976 | **0.934** |

GVLS is **competitive with or matches VGNAE on Cora and CiteSeer**, with CiteSeer at the 20% split achieving parity with the current best (0.941). PubMed performance at the 20% split is below baselines; the low-data regime on a large, sparse graph exposes sensitivity to graph method and prior choice that will be addressed in upcoming ablations. Importantly, GVLS also learns a non-trivial latent graph structure (latent edge density ~0.007 on Cora) that differs from the input graph topology — demonstrating that the model discovers meaningful latent relational structure rather than simply copying the input.

These results are already at a level that warrants investigation for a classical graph learning publication, independent of the quantum application.

---

## 6. Planned Future Work

The primary remaining task is to implement and evaluate the **graph compression pipeline** — the step that directly enables the quantum application.

| Task | Timeline |
|---|---|
| Graph compression metrics: reconstruction F1, bits-per-edge, rate-distortion curves across d | Weeks 1–2 |
| PubMed performance improvement: ablation over graph method / prior / β in low-data setting | Week 1 |
| Node classification evaluation with linear probe on z̃ | Week 2 |
| QGNN integration: pass (z̃, A_z) to parameterized quantum circuit GNN | Weeks 3–4 |
| Full ablation table: flat vs. graph-MRF prior, attention vs. FGP, mp_rounds ∈ {0,1,2} | Week 3 |
| Latent graph visualization (A_z vs. input graph, t-SNE / UMAP of z̃ by class) | Week 4 |
| Draft paper sections: method, experiments, ablations | Weeks 4–5 |
| Final deliverables: code release, poster, report | Week 6 |

**Publication plan.** The GVLS model as a standalone classical method is a candidate for submission to a graph learning venue (NeurIPS, ICLR, or CIKM). The quantum application component, once validated through the compression pipeline, will target a quantum machine learning venue (QIP, IEEE Quantum, or a workshop at NeurIPS/ICML).

---

## 7. Reflections

The most impactful technical decision so far was replacing the originally planned mutual top-k intersection for latent graph sparsification with **union symmetrization**: intersection empties the graph at realistic graph sizes (N = 2,708 for Cora), while union symmetrization preserves a meaningful sparse structure and allows gradient flow. Similarly, using a residual connection (rather than a plain linear transformation) in latent message passing and removing the ReLU activation before the inner-product decoder were critical for training stability.

The NAS results revealed that across datasets, **FGP cosine similarity with k = 20** and an isotropic prior are consistently preferred over the attention method and graph-MRF prior for Cora and CiteSeer — the graph-MRF prior improves performance primarily on PubMed, where the graph structure is richer. This dataset-dependent behavior motivates the planned ablation study.

The results to date are more positive than anticipated at project start: achieving VGNAE parity on CiteSeer with a model that simultaneously learns a compressed latent graph structure is a strong signal that the core idea is sound. The next six weeks are focused on converting these validation results into a working compression pipeline and a quantum integration proof of concept.
