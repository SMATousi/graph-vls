# Mission

## Problem

Variational graph autoencoders (VGAEs) learn a latent representation for graph-structured data, but virtually all existing methods — VGAE, GraphVAE, and their descendants — use a **flat, Euclidean latent space**. Each node is independently mapped to a Gaussian distribution; the latent distributions are unconnected and have no relational structure among themselves.

This creates a fundamental mismatch: the input is a graph, the task often depends on relational structure, but the latent space discards that structure entirely. A flat vector prior treats all latent dimensions as independent and ignores learned relationships between nodes.

## Core Idea

**graph-vls** develops a **Graph Variational Latent Space (GVLS)**: a latent space in which each node maps to a distribution *and* those distributions are connected through a learned latent graph.

The key components:

1. **Node-level variational posteriors** — each node is encoded as a Gaussian (mean + variance) using a GNN encoder, retaining the variational formulation.
2. **Latent graph inference** — a differentiable module learns the adjacency of the latent graph end-to-end, decoupled from the input graph topology.
3. **Graph-structured latent message passing** — distributions in the latent space communicate through the inferred graph, allowing relational dependencies between latent factors to be captured.
4. **Graph-aware prior and ELBO** — the KL divergence term accounts for the latent graph structure, using a graph-dependent prior (e.g., a Gaussian Markov Random Field defined over the inferred latent adjacency) rather than an independent isotropic Gaussian.

## What This Is Not

- This is not a graph *generation* model (though generation is a natural extension).
- This is not a hierarchical pooling model — the latent graph is not a coarsened version of the input; it is learned independently.
- This is not a simple extension of attention mechanisms — the latent graph is an explicit, interpretable adjacency structure with its own inductive bias.

## Goals

The project targets three downstream tasks as primary evaluation criteria:

| Task | Why it validates the approach |
|---|---|
| Node classification | Tests whether graph-structured latent factors encode discriminative relational information |
| Link prediction | Tests whether the latent graph topology reflects meaningful proximity in the input |
| Graph compression | Tests whether the GVLS provides a compact, lossless-enough encoding of graph structure |

## Success Criteria

- GVLS outperforms or matches VGAE and its variants on standard benchmarks (Cora, CiteSeer, PubMed, selected TU datasets) on link prediction and node classification.
- Graph compression quality (reconstruction fidelity vs. latent dimensionality) is measurably better than flat VGAE baselines.
- The inferred latent graph is interpretable and differs meaningfully from the input graph (demonstrating that the model learns non-trivial latent structure).
- Results are reproducible and suitable for a research paper submission.
