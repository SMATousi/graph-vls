# CVPR 2027 paper preparation plan: classical GVLS

**Written:** 2026-10-07

**Target:** CVPR 2027 main conference

**Scope:** classical Graph Variational Latent Space (GVLS) on image-derived graphs. The QGNN and jet experiments are outside this paper plan. This document is a proposed research plan, not a claim that the experiments have been run.

## 1. Submission clock and paper decision

The [official CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) lists **paper registration on November 10, 2026**, **submission on November 16, 2026**, and **supplementary material on November 23, 2026**, all Anywhere on Earth. The dates are stated to be fixed. All authors need current OpenReview profiles. As of this document, the linked 2027 author-guidelines page is unavailable; check it when published for the final template, page limit, anonymity, supplementary, and code rules. The [2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) used an eight-page main paper, but that is a planning assumption, not a verified 2027 rule.

**Candidate claim:** an image region graph can be compressed into a smaller *learned relational latent graph* while preserving useful region-level visual information. The paper needs evidence on both sides: competitive region recognition at a stated storage/compute budget, and a causal gain from the learned latent graph beyond ordinary pooling. A compression result alone, or a good classifier that ignores `A_z`, does not establish the claim.

**Go/no-go by October 28:** submit this story only if at least one vision dataset shows a reproducible accuracy–rate advantage over a capacity-matched pooling baseline and a meaningful learned-graph ablation. If it does not, report the negative result internally and consider a narrower graph-learning venue or a later CVPR cycle. Do not turn the unrun JEPA proposal into a claimed contribution.

## 2. What the repository currently establishes

- The [mission](mission.md) calls for node-count, edge-count, and dimensional compression with learned pooling, latent message passing, a graph-aware prior, and unpooling reconstruction.
- [Phase 2 validation](phase2/validation.md) records citation-network link prediction on Cora, CiteSeer, and PubMed. These are useful supporting results, but are not vision evidence. The historical NAS choices may be affected by the later KL-normalization fix; compare against new runs under one declared loss convention before using them in a paper table.
- [Phase 3 validation](phase3/validation.md) records a reconstruction F1 ceiling below 0.90 on all three citation graphs, a flat response to latent capacity, and `A_z` being inert or nearly inert under the CiteSeer/PubMed selected configurations. The graph-conditioned decoder experiment did not provide a reliable overall gain. The published paper must not imply that these issues are solved.
- [Phase 5 validation](phase5/validation.md) found inert variational components in the jet configuration. Its occupancy-aware posterior improved frozen-feature classification, but that result is domain-specific and the QGNN is excluded here. The proposed mixture prior and stochastic `A_z` were not implemented at the last recorded validation point.
- The current `PooledGVLS` returns pooled distributions, `A_z`, assignment `S`, and full-graph reconstruction. The current top-k `LatentGraphLearner` becomes complete when `k >= M-1`; the default attention score has no parameters of its own. For node prediction on image graphs, add a readout from the **pooled graph back to input regions**, and confirm that graph edges affect that readout.

## 3. Datasets and tasks

| Priority | Dataset | Task and metric | Reason / protocol |
|---|---|---|---|
| Required pilot | [PascalVOC-SP](https://github.com/vijaydwivedi75/lrgb) | Superpixel node classification; official macro F1 | Image-derived graphs, manageable pilot, official splits and evaluation. Use the benchmark's graph construction and supplied node features first. |
| Required confirmation | [COCO-SP](https://github.com/vijaydwivedi75/lrgb) | Superpixel node classification; official macro F1 | Larger, more varied images and the same task family. Use official splits; do not tune on its test set. |
| Supporting continuity | Cora, CiteSeer, PubMed | Link prediction AUC/AP and graph reconstruction | Re-run selected comparisons under a consistent modern implementation; place in supplementary unless the vision claim depends on them. |

The [LRGB paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) defines PascalVOC-SP and COCO-SP as image-derived superpixel node-prediction tasks; its [official repository](https://github.com/vijaydwivedi75/lrgb) supplies loaders and baseline protocols. Reproduce the exact graph variant, split, metric aggregation, node/edge features, and preprocessing version used in each comparison. The [LRGB reassessment](https://openreview.net/forum?id=rIUjwxc5lj) warns that baseline protocol and tuning can materially change the ranking, so reproduce strong comparators rather than copying only the original table.

**Optional only if the required pair succeeds early:** one image-level task on the same graphs, using a documented image label and a graph readout, to demonstrate a second downstream use. Do not add a new dataset solely to fill a table. Do not present ImageNet/COCO pixel-level performance claims without an image backbone and a fair pixel-based baseline; these experiments use supplied region-graph features.

## 4. Model and experiment gates

### Gate A — make node prediction and compression well-defined

1. Add a region readout that maps pooled features back to the original `N` superpixels, e.g. `H_hat = S @ f(A_z, Z_p)` followed by a shared node classifier. Preserve the existing reconstruction interface. Compare against `S @ Z_p` with **no** latent message passing and against a matched graph pooling model whose pooled adjacency is `S.T @ A_input @ S`.
2. Make `M` configurable as a fixed count and as a fraction of each image graph's `N`; record the actual `M/N` distribution. Avoid `k >= M-1` in the primary sparse setting. A separate full-graph arm is useful as an ablation.
3. Select checkpoint and hyperparameters by **validation macro F1** for recognition experiments. Use validation reconstruction metrics only for the rate–distortion study. Train probes on training labels only; fit thresholds/calibration on validation data only.
4. Keep node labels out of unsupervised pretraining. For supervised fine-tuning, report the exact label budget and distinguish it from frozen linear-probe results. The same image and its graph must remain in one official split.

### Gate B — show the latent graph is actually used

5. At `M` and `k` used in the main table, log graph density, fraction of complete/empty graphs, edge-weight distribution, and variation across images. Report `||∂L_task/∂A_z||` and compare predictions after setting `A_z=0`, shuffling edges between images, replacing it with a fixed complete graph, and using `S.T @ A_input @ S`. An architectural claim requires a material change in held-out performance under these interventions.
6. If `A_z` is inert, add an explicit graph-dependent route to the region readout and/or decoder and compare with a parameter-matched graph-free route. A learned symmetric pairwise edge scorer with a sparsity budget is the first candidate. Keep deterministic top-k as a baseline. Implement Concrete/Bernoulli edges or a graph prior only if the deterministic graph has established a gain and the added probabilistic component answers a specific remaining question.
7. Audit the objective before large sweeps: positive-edge weighting, loss normalization across varying `N`, KL contribution, assignment-link-loss share, posterior standard deviation, effective cluster occupancy, and gradients to the encoder, assignment, edge scorer, and message-passing layers. Sweep a few plausible loss weights on validation data. Keep the chosen convention fixed across the main comparison.

### Gate C — measure the claimed compression

8. Plot **macro F1 versus total representation bits**, not just versus `d` or `M/N`. Count quantized pooled means and variances if transmitted, latent edges and weights, node-to-cluster assignment, `M`/shape metadata, and any other per-image data needed to decode or predict. State the quantization scheme and whether shared model weights are amortized. Report `M/N`, `|A_z|/|E_input|`, and `d/F` separately as diagnostic axes.
9. Use the same rate budget and decoder/readout capacity for flat VGAE, input-graph coarsening, learned pooling with no latent edges, and full GVLS. Include a no-compression GNN ceiling. Report encoder latency, peak memory, parameter count, and decode/readout latency; compare on the same hardware and software environment.
10. For adjacency fidelity, score held-out **images**, with positive and negative pairs defined consistently. Report AP/AUC and calibrated F1 plus bits per edge. A full-graph training reconstruction on the same graph is a memorization diagnostic, not generalization. A 0.5 threshold or a balanced sampled-pair metric alone can hide poor calibration on sparse adjacency.

## 5. Required comparison matrix

| Group | Runs to include | Question answered |
|---|---|---|
| Strong vision-graph baselines | Official LRGB GatedGCN and a strong graph-transformer/GPS-style baseline, reproduced or carefully protocol-matched | Is the result relevant to the actual vision task? |
| Compression controls | Same encoder + learned pooling without `A_z`; DiffPool-style `S.T A S`; flat VGAE at matched bit budget; simple graph coarsening | Which gain comes from pooling, graph learning, or extra capacity? |
| GVLS components | `M`, `d`, edge budget `k`; no/one/two latent message-passing rounds; learned versus input-coarsened versus random/complete/empty `A_z`; isotropic versus graph-MRF prior | Is each claimed mechanism useful? |
| Assignment and uncertainty | Soft versus hard assignment at inference; assignment auxiliary losses on/off; occupancy-aware posterior on/off; mean-only versus mean plus variance features | What information survives compression, and what is the storage cost? |
| Robustness | At least 5 seeds for headline rows, per-dataset mean ± standard deviation; validation-selected settings; image-size and graph-density strata | Does the effect survive seed and graph variation? |

Implement baselines under matched splits, node features, training budget, and, where possible, parameter count. Label literature-only numbers separately. The original [VGAE](https://arxiv.org/abs/1611.07308) is a necessary flat-latent reference, but a task-specific supervised GNN and a matched pooling model are more direct competitors for image-region node prediction. Report negative ablations and the total search budget. Do not describe `A_z` as novel merely because it differs visually from the input graph; Graph-JEPA and other graph representation methods already study learned latent structure.

## 6. Proposed JEPA-style training track

This is an **experimental extension**, not a prerequisite to the baseline GVLS paper. [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf) predicts target-block representations from a context view with a slowly updated target encoder. [Graph-JEPA](https://arxiv.org/abs/2309.16014) already applies masked-subgraph latent prediction to graph representation learning; [HP-JEPA](https://arxiv.org/abs/2608.00491) explores multiple graph-partition scales. The contribution here would have to be the interaction with a **variational, compressed, learned latent graph**, demonstrated by matched ablations, not “the first graph JEPA.”

### Concrete design

1. Sample one or more connected target **superpixel blocks** per image, at multiple sizes and boundary/interior locations. Build a context graph after removing target node appearance/features and incident edges that reveal the target. Give the predictor target positional/shape metadata only if the same metadata is available at inference; never pass target RGB statistics, labels, or hidden adjacency through `S` or another path. Random-node masks are an ablation, not the sole mask policy.
2. Online context encoder: `GVLS` encoder → assignment `S_c` → pooled posterior `(mu_c, log_var_c)` → learned `A_z,c` → message passing. A predictor queries this compact graph with target position tokens to produce predicted target embeddings. The predictor must consume `A_z,c`; include its graph-free counterpart.
3. EMA target encoder sees the **unmasked** image graph during training and emits per-target-node embeddings *before pooling*, so there is a stable target index and no cluster-permutation alignment problem. Stop gradients through the target branch. Predict normalized target means with smooth L1 or cosine loss. Do not treat squared-error regression to a single target mean as a calibrated uncertainty objective.
4. Start with `L = L_GVLS + alpha * L_predict`. Compare GVLS only, JEPA only with the same encoder and capacity, and the joint objective at matched training steps. If the joint objective helps, test whether a distributional target (teacher posterior `mu, log_var` with a stop-gradient Gaussian divergence) improves calibration; retain the reconstruction/ELBO and report the KL separately so “variational” remains justified. If the predictive-only arm wins, describe it as a new representation learner rather than calling its objective an ELBO.
5. Monitor representation variance/covariance, effective rank, posterior variance, cluster occupancy, and validation target-prediction error to catch collapsed constant embeddings. Compare EMA with a frozen target encoder and predictor/no-predictor controls. Test masking fractions and block scale on validation data; prohibit full-view shortcuts in the context path.

### Decision rule

Run a small PascalVOC-SP pilot by October 24. Keep JEPA in the main paper only if it improves held-out macro F1 or the accuracy–rate frontier across seeds, beyond equal-compute GVLS and a [GraphMAE](https://arxiv.org/abs/2205.10803) masked-pretraining control, while the learned-graph intervention remains material. Otherwise place a concise negative result in supplementary or defer it to a follow-up. This avoids claiming novelty from adding a fashionable loss to an otherwise unchanged model.

## 7. Execution schedule and artifacts

| Date (2026) | Deliverable | Acceptance check |
|---|---|---|
| Oct 7–12 | Freeze research question, literature map, LRGB loader/protocol, and baseline config manifest | Data hashes/splits and metric implementation documented; original and reassessed LRGB comparators identified |
| Oct 13–19 | Node readout, graph-influence audit, PascalVOC-SP pilot, matched pooling/VGAE controls | `A_z` has a causal path; at least one non-degenerate sparse setting; pilot includes a fair accuracy–rate point |
| Oct 20–26 | COCO-SP confirmation, main ablations, optional JEPA pilot | Both vision datasets have preliminary tables; JEPA decision recorded |
| Oct 27–Nov 2 | Freeze model and hyperparameters; five-seed final runs and rate/compute sweeps | Primary claim passes October 28 gate; every headline number traced to config, seed, checkpoint, and CSV |
| Nov 3–9 | Main paper, supplementary, figures, internal review, anonymity pass | Paper registration by Nov 10; claims match measured results; all authors' OpenReview profiles ready |
| Nov 10–16 | Final edits and submission | Submit by Nov 16 AoE using the current 2027 template and policies |
| Nov 17–23 | Supplementary/code package and reproducibility audit | Upload supplement by Nov 23 AoE; check final anonymization rules |

Create `experiments/paper_cvpr2027/` for immutable run manifests, `configs/paper_cvpr2027/` for frozen configs, `results/paper_cvpr2027/` for per-seed raw metrics and aggregation, and `reports/cvpr2027/` for manuscript/figures. Save dataset version/hash, split identifier, commit, seed, hardware, runtime, training budget, checkpoint-selection rule, and feature/bit accounting with each run. The main paper should contain the method diagram, vision-task table, F1-versus-bits plot, graph-causality ablation, and examples of assignments/latent edges; supplementary should hold full grids, citation-network continuity, failure cases, and reproducibility details.

## 8. Paper integrity checklist

- State the actual likelihood and loss normalization; do not call a weighted BCE objective an exact graph ELBO without deriving the implied model.
- State whether `A_z` is deterministic or stochastic in each experiment. A deterministic top-k graph is **not** a variational edge posterior.
- Compare image-derived graph methods with image-derived graph methods; do not claim pixel-level vision performance from superpixel features alone.
- Keep test images unseen until the final frozen protocol; use validation for all model, threshold, and operating-point choices.
- Present failure cases, zero-effect components, and compute costs alongside positive results. If the paper claim fails its gate, revise the claim before submission rather than selecting a favorable seed.
- Recheck the official [CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) and author guidelines before registration and submission; policy details may change after this plan was written.
