# Tech Stack

## Language and Runtime

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Standard for ML research; type hints improve readability |
| Package manager | `uv` or `conda` | Fast dependency resolution; reproducible environments |

---

## Core ML Framework

**PyTorch 2.x** — primary framework.

- `torch.compile` for training speed without leaving PyTorch
- Native support for custom autograd (needed for differentiable graph learning)
- Dominant choice in graph ML research; best ecosystem compatibility

**PyTorch Geometric (PyG) 2.x** — graph operations.

- `MessagePassing` base class for encoder GNNs and latent message passing
- Built-in datasets: `Planetoid` (Cora, CiteSeer, PubMed), `TUDataset` (MUTAG, PROTEINS, IMDB-B)
- `torch_geometric.transforms` for data preprocessing
- `torch_geometric.utils` for adjacency manipulation and sparse ops

**torch-scatter / torch-sparse** — PyG sparse primitives (required by PyG).

---

## Latent Graph Learning

The latent graph inference module is the central technical novelty. Key references and their implementations to draw from:

| Approach | Reference | Notes |
|---|---|---|
| Attention-based adjacency | Velickovic et al. (GAT) | Differentiable, O(N²) dense or sparse top-k |
| Feature Graph Prior (FGP) | Franceschi et al. (LDS, 2019) | Bilevel optimization; harder to train |
| NRI-style edge inference | Kipf et al. (NRI, 2018) | Encoder over node pairs → edge type |

Start with attention-based; gate the others behind the ablation phase.

---

## Variational Inference Utilities

No external VI library is needed. Implement directly:

- Diagonal Gaussian: `torch.distributions.Normal`
- Gaussian MRF KL: closed-form when Σ is diagonal; numerical for full precision matrix
- Reparameterization: `.rsample()` on `Normal`

If MRF KL becomes intractable at scale, fall back to Monte Carlo KL estimation using `torch.distributions.kl_divergence`.

---

## Experiment Management

| Tool | Purpose |
|---|---|
| **Hydra 1.3** | Hierarchical config management; enables sweep + override from CLI without editing code |
| **Weights & Biases** | Experiment tracking, metric logging, sweep orchestration, latent graph visualizations |
| **PyTorch Lightning** (optional) | Boilerplate reduction for training loops; adopt only if training loop complexity warrants it |

Config structure (Hydra):
```
configs/
  model/        # encoder, latent_graph, decoder configs
  data/         # dataset, split, preprocessing
  train/        # optimizer, scheduler, loss weights
  experiment/   # named experiment overrides
```

---

## Quantum Component (Phase 4, new 2026-07-14)

**Qiskit + Qiskit Machine Learning** — the QGNN that Phases 0–3's compression work was always described as building toward (`mission.md`, `reports/midterm_report.md` §6).

- `EstimatorQNN` wraps the ansatz circuit; `TorchConnector` (Qiskit Machine Learning) embeds it as a standard `torch.nn.Module`, so the existing PyTorch/Hydra/W&B training loop carries over unchanged — only the model gains a quantum sub-module
- Gradients through the circuit use Qiskit Machine Learning's built-in parameter-shift-rule differentiation, exposed to PyTorch autograd via `TorchConnector`; no custom backward pass is written
- Ansatz: a Verdon et al.-style **Quantum Graph Neural Network** layer (Verdon, Broughton, McClean et al., "Quantum Graph Neural Networks," arXiv:1909.12264) — one qubit per pooled latent node `M`, entangling `RZZ` gates placed on the learned `A_z`'s actual edges (not a generic hardware-efficient ansatz), giving the circuit's topology a direct, literal correspondence to GVLS's learned latent graph
- Simulation only, via Qiskit Aer's noiseless statevector simulator — no real hardware or noise-model execution planned for Phase 4 (that's Phase 5+ ablation material if pursued)
- Chosen over PennyLane (better PyTorch integration, but user preference favored Qiskit) and TensorFlow Quantum (would require a parallel TF training loop) — see `specs/phase4/plan.md` Design Decision 1

**`energyflow`** (assumed, not yet confirmed — `specs/phase4/plan.md` Design Decision 1) — source for the Pythia8 quark/gluon jet dataset (`qg_jets`).

---

## Datasets

| Dataset | Task | Source |
|---|---|---|
| Cora, CiteSeer, PubMed | Link prediction, node classification | PyG `Planetoid` |
| MUTAG, PROTEINS, IMDB-B | Graph classification | PyG `TUDataset` |
| ogbn-arxiv (stretch) | Large-scale node classification | OGB |
| ogbl-collab (stretch) | Large-scale link prediction | OGB |
| Pythia8 quark/gluon jets (Phase 4) | Graph classification (QGNN) | `energyflow.qg_jets` (assumed — to confirm) |

---

## Evaluation

- **Link prediction**: AUC-ROC, Average Precision (AP) — `sklearn.metrics`
- **Node classification**: Accuracy, macro-F1 — `sklearn.metrics`
- **Graph compression**: Reconstruction F1, bits-per-edge (derived from cross-entropy of adjacency predictions)
- **Latent graph analysis**: Jaccard similarity between inferred A_z and input A; community alignment (NMI)

---

## Testing and Code Quality

| Tool | Purpose |
|---|---|
| **pytest** | Unit tests for model components, loss functions, data loaders |
| **ruff** | Linting and formatting (replaces flake8 + black) |
| **mypy** | Static type checking |

Test targets: ELBO value on a toy graph (check sign and scale), KL divergence implementations (compare against analytical values), latent graph inference (verify gradient flow).

---

## Repository Structure

```
graph-vls/
  specs/              # mission, roadmap, tech_stack
  src/
    gvls/
      models/         # encoder, latent_graph, decoder, full model
      losses/         # ELBO, KL variants
      data/           # dataset wrappers, splits
      eval/           # metrics, visualization
  experiments/        # scripts for each paper experiment
  configs/            # Hydra configs
  tests/
  notebooks/          # exploratory analysis, figure generation
```

---

## Not Used (and Why)

| Tool | Reason excluded |
|---|---|
| TensorFlow / JAX | PyTorch ecosystem dominates graph ML research tooling |
| DGL | PyG is more widely used in VGAE literature; easier to compare against baselines |
| Ray Tune | Hydra + W&B sweeps sufficient for this scale |
| ONNX / TorchScript | Not needed for research prototype |
