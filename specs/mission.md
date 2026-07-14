# Mission

## Problem

Variational graph autoencoders (VGAEs) learn a latent representation for graph-structured data, but virtually all existing methods — VGAE, GraphVAE, and their descendants — use a **flat, Euclidean latent space**. Each node is independently mapped to a Gaussian distribution; the latent distributions are unconnected and have no relational structure among themselves.

This creates a fundamental mismatch: the input is a graph, the task often depends on relational structure, but the latent space discards that structure entirely. A flat vector prior treats all latent dimensions as independent and ignores learned relationships between nodes.

## Core Idea

**graph-vls** develops a **Graph Variational Latent Space (GVLS)**: a latent space in which each node maps to a distribution *and* those distributions are connected through a learned latent graph that is **smaller than the input graph in both node count and edge count**.

The key components:

1. **Node-level variational posteriors** — each node is encoded as a Gaussian (mean + variance) using a GNN encoder, retaining the variational formulation.
2. **Learned node-count compression (pooling)** — a differentiable soft-assignment module maps the `N` input-node distributions onto `M ≪ N` latent distributions (a DiffPool-style assignment matrix `S ∈ [0,1]^{N×M}`), so the latent graph has genuinely fewer nodes than the input, not just a smaller per-node dimensionality.
3. **Latent graph inference** — a differentiable module learns the adjacency of the `M`-node latent graph end-to-end, decoupled from the input graph topology.
4. **Graph-structured latent message passing** — distributions in the latent space communicate through the inferred `M`-node graph, allowing relational dependencies between latent factors to be captured.
5. **Graph-aware prior and ELBO** — the KL divergence term accounts for the latent graph structure, using a graph-dependent prior (e.g., a Gaussian Markov Random Field defined over the inferred latent adjacency) rather than an independent isotropic Gaussian.
6. **Unpooling decode** — reconstruction of the full `N`-node graph reuses the same assignment matrix `S` learned during pooling: `Â = S · σ(z̃_pooled z̃_pooledᵀ) · Sᵀ`.

## What This Is Not

- This is not a graph *generation* model (though generation is a natural extension).
- This is not a simple extension of attention mechanisms — the latent graph is an explicit, interpretable adjacency structure with its own inductive bias.

**Changelog (2026-07-09):** earlier drafts of this document stated GVLS was explicitly *not* a hierarchical pooling model and that the latent graph would not be a coarsened version of the input. That decision is reversed: node-count reduction (via learned soft assignment/pooling) is now a core goal, motivated by wanting the latent graph of distributions to be smaller than the input graph in both nodes and edges, not just in per-node dimensionality. See `specs/roadmap.md` Phase 3 and `specs/phase3/` for the resulting task changes.

**Changelog (2026-07-14):** the quantum application this project has always been building toward (see "What This Is Not" and `reports/midterm_report.md` §6) is now a concrete phase, not just a stated future direction: Phase 4 pairs GVLS's fixed-node-count pooling with a Qiskit-based Quantum Graph Neural Network (Verdon et al.-style, entangling gates placed on the learned A_z's edges) to classify Pythia8 quark/gluon jets. This is also the project's first inductive, many-small-graphs task — all prior phases operated on a single large transductive graph. See `specs/roadmap.md` Phase 4 and `specs/phase4/`.

## Goals

The project targets three downstream tasks as primary evaluation criteria:

| Task | Why it validates the approach |
|---|---|
| Node classification | Tests whether graph-structured latent factors encode discriminative relational information |
| Link prediction | Tests whether the latent graph topology reflects meaningful proximity in the input |
| Graph compression | Tests whether the GVLS provides a compact, lossless-enough encoding of graph structure — in node count, edge count, *and* per-node dimensionality |
| Quantum graph classification (new, 2026-07-14) | Tests whether the compressed (z̃, A_z) is small and structured enough to serve as a QGNN's direct input, and whether that hybrid pipeline can perform a real classification task (Pythia8 quark/gluon jet tagging) — the project's original motivating application (see "Quantum Application Pathway" in `reports/midterm_report.md`) |

## Success Criteria

- GVLS outperforms or matches VGAE and its variants on standard benchmarks (Cora, CiteSeer, PubMed, selected TU datasets) on link prediction and node classification.
- Graph compression quality (reconstruction fidelity vs. latent size) is measurably better than flat VGAE baselines, where latent size is reported across three separate axes: node-count ratio (`M/N`), edge-count ratio (`|A_z|/|E|`), and dimensionality ratio (`d/F`).
- The latent graph has genuinely fewer nodes than the input graph (`M < N`) at the compression-optimal operating point, not merely a smaller per-node embedding.
- The inferred latent graph is interpretable and differs meaningfully from the input graph (demonstrating that the model learns non-trivial latent structure).
- Results are reproducible and suitable for a research paper submission.
