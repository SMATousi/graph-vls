# CVPR 2027 paper preparation plan: generative GVLS for visual graphs

**Written:** 2026-10-07. **Revised:** 2026-10-07 after the decision to center graph generation.

**Target:** CVPR 2027 main conference.

**Scope:** classical Graph Variational Latent Space (GVLS). QGNN and jet classification are outside this plan. Proposed experiments below are unrun unless explicitly identified as existing results.

## 1. Paper thesis and submission clock

**Proposed question:** Can a smaller, learned *graph of latent distributions* model a useful distribution over visual relationships, rather than merely reconstruct one adjacency matrix? Given an image and a fixed object inventory (object boxes and labels), the model should sample plausible **directed, typed relationship graphs** and assign probabilities to annotated relations. The object inventory is conditioning information, not generated output. We should call this *conditional scene-graph generation* or *conditional relation-graph generation*, not unconditional graph or image generation.

The central claim requires three pieces of evidence: (1) valid test-time samples from a learned prior, with more than one plausible graph per image; (2) competitive relation prediction and calibrated probabilities on held-out images; and (3) a causal benefit from compressing into a **learned latent graph**, beyond a flat VAE or an ordinary pooled graph. Reconstruction F1 from the encoder's posterior alone cannot establish any of these.

The [official CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) lists paper registration on **November 10, 2026**, main submission on **November 16, 2026**, and supplementary material on **November 23, 2026**, all Anywhere on Earth and stated to be fixed. All authors need current OpenReview profiles. The linked 2027 author-guidelines page was unavailable when this plan was written; check the final template, length, anonymity, and code rules when published. The [2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) used an eight-page main paper, which is only a working layout assumption.

**Go/no-go by October 28:** the generative model must beat a matched flat/pooled VAE on held-out conditional prediction or sample quality, and removing/shuffling `A_z` must measurably change the result. If prior samples fail while posterior reconstructions look good, do not submit a generative claim on that evidence. A narrower graph representation paper or later submission is the fallback decision, not a change to what was measured.

## 2. What must change from the current repository

The [mission](mission.md) describes a variational encoder, learned pooling `S`, latent adjacency `A_z`, message passing, graph-aware KL, and unpooling. [Phase 3 validation](phase3/validation.md) shows reconstruction F1 below 0.90 on Cora/CiteSeer/PubMed; some selected configurations give `A_z` no meaningful route into the output. [Phase 5 validation](phase5/validation.md) found the variational loss weak and the jet latent graph complete under its production configuration. These are constraints to investigate, not paper results to hide.

The current `PooledGVLS` is an **autoencoder for an observed graph**. At decoding it reuses `S`, which was computed from that graph's encoded nodes, and it returns an undirected binary adjacency reconstruction. It has no learned prior that can produce a new `S`, node count, node attributes, or relation types at test time. Its deterministic top-k adjacency also becomes complete whenever `k >= M-1`. In particular, sampling a standard Gaussian and calling the existing decoder is not yet a valid standalone graph generator.

The smallest credible CVPR scope is therefore **conditional** generation: object count, boxes, labels, and image/ROI features are given at both training and test time. Derive `S` from that *shared conditioning information* so the decoder can use the same assignment at inference. Model the missing relationship graph; leave object creation and image synthesis for future work. State in the paper that the compressed representation excludes the given conditioning information, and measure its bit/parameter cost separately if claiming a storage advantage.

## 3. Generative model specification

Let `c` contain an image, a fixed set of `N` object boxes/classes, and fixed ROI image features. Let `R` be the annotated directed predicate graph, including a declared treatment of unannotated ordered pairs. Let `S(c) in [0,1]^(N x M)` be an assignment learned **only from c**, with `M < N` where feasible. The proposed conditional model is:

```text
p(Z, A_z, R | c) = p_psi(Z | c) p_omega(A_z | Z, c)
                    product_(i != j) p_theta(R_ij | c, S(c), Z, A_z).
q_phi(Z, A_z | c, R) approximates the posterior during training.
```

Here `Z` consists of `M` latent node variables; `A_z` is a symmetric or directed latent edge variable whose convention must be chosen and documented. The **output** relation graph is directed and typed regardless of that choice. The training objective is the conditional ELBO, evaluated on the *observed annotation set*:

```text
E_q [log p_theta(R_observed | c, S, Z, A_z)]
    - KL(q_phi(Z, A_z | c, R) || p_psi(Z, A_z | c)).
```

If `A_z` is initially a deterministic function of sampled `Z`, its edge distribution is induced by `Z` and there is **no separate edge KL**; describe it that way. Only claim a variational edge posterior after implementing explicit `q(A_z | ...)`, `p(A_z | ...)`, and their KL. A conditional prior for `Z` is required even in the minimal version, because samples and test likelihood estimates must use a prior available without ground-truth `R`.

### Architecture tasks

1. **Data adapter:** represent each image's objects as nodes with class, box geometry, and frozen ROI visual features. Build a stable mapping between object indices, ordered pairs, and predicate labels. Keep ROI extraction/checkpoint fixed across all methods; cache features and report their provenance. Begin with ground-truth boxes/classes (the standard PredCls setting), then test robustness to detector predictions only if the first stage works.
2. **Shared assignment:** make `S(c)` depend only on information available at generation time; do not use true relation edges, target predicates, or posterior-only node features to build it. Record `M/N`, cluster occupancy, and whether hardening `S` changes samples. A latent node count that is not smaller than `N` is a baseline setting, not compression.
3. **Posterior and prior:** implement `q_phi(Z | c,R)` and `p_psi(Z | c)` with matched Gaussian shapes. At training time, sample from `q`; at test time, sample from `p`. Track posterior/prior divergence, active dimensions, variance, and posterior collapse. Start with a conditional diagonal Gaussian prior; add a mixture or graph-MRF prior only after the basic conditional VAE works and an ablation points to a limitation.
4. **Latent graph:** infer sparse `A_z` from sampled `Z` with an edge scorer and edge budget that cannot silently yield a complete graph. Compare deterministic top-k, input-derived coarsening `S.T A_input S` where an input relation graph is actually observable, and a parameter-matched graph-free model. Add Bernoulli/Concrete edges with a sparsity prior if the stochastic extension improves a defined test metric or sample property; retain the deterministic path for attribution.
5. **Typed generative decoder:** use `S(c)` to unpool latent context to object nodes, then score each ordered pair for relation categories. Define whether pairs may have multiple predicates; if yes, use a multilabel likelihood, and if no, include a documented background/no-annotation category. Pairwise scoring must condition on object/ROI features and *processed* latent graph features, not only raw `Z`; otherwise `A_z` can be decorative. Avoid claiming a normalized likelihood over true scene relations when the dataset supplies only partial annotations.
6. **Generation and completion API:** implement `sample(c, seed, num_samples)` that never reads ground-truth `R`. A separate `complete(c, R_context, mask, seed)` may condition a prior on observed relations and sample missing ones. For completion, remove target relations from all context-encoder, assignment, and pair-feature paths; train and evaluate with masks defined before encoding. Do not substitute posterior reconstructions for either API.
7. **Loss audit:** quantify relation likelihood, node KL, edge KL if present, and each auxiliary assignment term per image. Define reductions over object pairs so loss strength is comparable across graph sizes. Check positive/unknown pair weighting and calibrate using validation data. If using a weighted surrogate instead of the stated ELBO, name it and separate likelihood estimates from surrogate training loss.

## 4. Datasets and protocols

| Priority | Dataset | Use | Conditions |
|---|---|---|---|
| Main | [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) with a documented VG150-style scene-graph split | Conditional relation-graph generation and masked relation completion | Freeze exact image IDs, object/predicate vocabulary, boxes, ROI feature extractor, and annotation preprocessing. Report standard PredCls results separately from generative results. |
| Independent confirmation | [VRD](https://cs.stanford.edu/people/ranjaykrishna/vrd/) | Smaller second relation-graph dataset; generation/completion under its own vocabulary | Keep its official test split, carve validation only from training images, and disclose the smaller data budget. Retrain under a matched protocol; do not call it zero-shot transfer unless label mapping and image overlap are handled explicitly. |
| Optional scale/stress test | [Open Images V6 visual relationships](https://storage.googleapis.com/openimages/web/download.html) | Larger, different annotation regime | Add only after the two required datasets work and its relationship-evaluation protocol is implemented correctly. |
| Supplementary continuity | Cora, CiteSeer, PubMed | Classical binary link prediction/reconstruction diagnostic | Re-run under one declared KL convention. These graphs do not validate a visual relationship generator. |

The [Visual Genome source](https://link.springer.com/article/10.1007/s11263-016-0981-7) supplies image scene graphs. Standard [scene-graph evaluation modes](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.pdf) distinguish predicate classification with given boxes/classes from tasks that also predict objects. This paper's main setting is the former, with an additional **sampling** test. Do not compare its PredCls number directly with SGCls or SGDet. Visual Genome relationships are sparse/incomplete; [work on limited-label scene graphs](https://cs.stanford.edu/people/ranjaykrishna/limitedlabels/index.html) documents that issue. Treat unannotated pairs as unknown unless a specified benchmark protocol asks for background; include sensitivity analyses for negative sampling and avoid interpreting “unannotated” as “false.”

## 5. Evaluation: generation first

**Primary table:** held-out VG and VRD relation prediction using standard `R@20/50/100` and `mR@20/50/100` (or each dataset's official equivalents), with the exact graph constraint and candidate-pair setting stated. Include a no-latent discriminative relation predictor so the VAE's cost is visible. Report per-predicate results or frequency strata, not only a head-class-dominated mean.

**Generative evidence:** for each held-out image, draw multiple independent samples from `p(Z,A_z|c)` and decode relation graphs. Report (a) test annotation log score or an importance-weighted conditional log-likelihood bound where a valid likelihood is defined; (b) positive-relation coverage and precision under a fixed output budget; (c) diversity at matched quality, e.g. unique sampled triplet sets and pairwise set distance; and (d) graph statistic distributions (relation counts, predicate frequencies, object–predicate–object motifs, node degrees) versus held-out annotated graphs. Show several samples for the **same** image, including failure cases. A diverse collection of implausible or duplicate samples is not success.

**Completion evidence:** mask a fixed fraction of *annotated* relations in each held-out graph, condition only on retained relations, and score the held-out ones. Compare masks that remove random edges versus connected subgraphs. Evaluate multiple mask fractions and record the exact candidate-pair/unknown-label policy. Use image-level train/val/test separation; the test graph must not be encoded in full when scoring hidden edges.

**Calibration and compute:** on pairs with reliable labels, report negative log likelihood/Brier score and calibration; identify where sparse annotations make them uninterpretable. Measure parameters, training and inference time, peak memory, number of prior samples, and dependence on `N` and `M`. Use at least five seeds for headline rows, with frozen hyperparameters and mean ± standard deviation. Select model/checkpoints using validation **generative** metrics, not training reconstruction F1.

## 6. Necessary baselines and ablations

| Comparison | Exact question |
|---|---|
| Frequency/geometry/ROI-only predictor and a strong supervised PredCls model such as [Neural Motifs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.pdf) under the same boxes/features | Is GVLS competitive at the visual relation task? |
| Conditional flat VGAE/CVAE with `N` object latents and a matched decoder; [GraphVAE](https://arxiv.org/abs/1802.03480) where its graph-size/output assumptions fit | Does the learned compressed graph add value over a conventional variational graph autoencoder? |
| Learned pooling with no `A_z`, coarsened observed graph, random/fixed/complete latent graph, and full GVLS with matched parameters | Is the learned latent topology causal, or merely an unused visualization? |
| [VarScene](https://proceedings.mlr.press/v162/verma22b.html) on a **matched** graph-synthesis setting | How does sample quality compare with an existing scene-graph VAE? Its unconditional generation numbers are not directly comparable with our image-conditioned PredCls numbers. |
| `M`, latent dimension, edge budget, message-passing rounds, assignment loss, beta/normalization, isotropic vs conditional/graph prior, deterministic vs stochastic edges | What actually controls likelihood, sample quality, sparsity, and collapse? |
| Posterior reconstruction versus prior sampling; single sample versus marginal prediction from several samples | Does the model genuinely generate at test time? |

Run capacity/compute-matched comparisons on the same split and ROI features. Use the same candidate pairs, unknown-label treatment, and output triplet budget. For causal `A_z` tests, set it to zero, shuffle it between images, and replace it with a fixed graph **at inference**; also report `||∂L/∂A_z||`, graph density, complete/empty fraction, and cross-image variation. Repeat training without each component to separate test-time perturbation from adaptation during training. Report negative results. Do not claim to be the first variational scene-graph generator: [GraphVAE](https://arxiv.org/abs/1802.03480) and [VarScene](https://proceedings.mlr.press/v162/verma22b.html) are prior generative work.

## 7. JEPA-style training as a generative-model extension

The primary paper objective remains a conditional likelihood/ELBO with a test-time prior. A JEPA loss can be an **auxiliary representation objective**; it is not itself a graph likelihood and does not make the model generative. [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf) and [Graph-JEPA](https://arxiv.org/abs/2309.16014) already establish latent prediction from masked context, so any contribution here must come from improving the variational latent graph's prior samples or completion, not from adopting a familiar loss.

**Suggested pilot:** mask a connected group of relationship edges before the online encoder. The online branch receives `c` plus visible relations, pools into `M` latents, and predicts teacher embeddings for masked object pairs or target subgraphs. An EMA teacher sees the full training graph; gradients stop at its targets. Keep target IDs aligned through the object inventory rather than matching permuted latent clusters. Prevent target predicates, edge features, or full-graph-derived `S` from leaking into the online path. At test time the teacher is absent; generate using the trained conditional prior/decoder.

Compare four matched-step arms: VAE alone, JEPA-style prediction alone, VAE + JEPA, and VAE + an equal-cost masked-relation-reconstruction auxiliary loss. Also compare prediction with and without `A_z`. Monitor embedding effective rank, posterior/prior KL, active latent dimensions, relation log score, prior-sample diversity, and completion quality. If JEPA improves only posterior embeddings but harms prior samples or likelihood, keep it out of the main generative claim. If a distributional target is attempted (teacher `mu, log_var`), state and test its divergence separately; a point-target MSE does not calibrate the variational posterior.

Run the pilot after the basic VAE generates nontrivial prior samples, by October 24 if feasible. Keep JEPA in the paper only if it improves a prespecified held-out **generative** metric across seeds under matched compute. Otherwise document the negative result in supplementary or defer it.

## 8. Work plan and artifacts

| Date (2026) | Tasks and deliverable | Gate |
|---|---|---|
| Oct 7–12 | Finalize VG/VRD split and annotation policy; literature matrix; frozen ROI features; minimal typed-graph loader | No image leakage; repeatable graph/feature mapping and relation labels |
| Oct 13–19 | Implement shared `S(c)`, conditional posterior/prior, typed decoder, and prior-only sampler; tiny-graph checks | Prior sampler runs without ground-truth `R`; sample shapes, edge direction, KL, and log probability are correct |
| Oct 20–26 | VG pilot, matched flat/pooled VAE, PredCls baseline, `A_z` causal audit, optional JEPA pilot | Nontrivial prior samples and fair first generative comparison |
| Oct 27–Nov 2 | Decide claim on Oct 28; VRD confirmation, five-seed final runs, sampling/completion/compute sweeps | Primary claim passes; test protocols and configs frozen |
| Nov 3–9 | Write method/ELBO derivation, results, sample figures, limitations, supplementary; internal review and anonymization | Every number maps to a run, seed, checkpoint, and evaluation protocol; register by Nov 10 |
| Nov 10–16 | Final paper and submission | Submit by Nov 16 AoE under the actual 2027 author rules |
| Nov 17–23 | Supplementary/code and reproducibility audit | Upload supplement by Nov 23 AoE |

Create `configs/paper_cvpr2027/`, `experiments/paper_cvpr2027/`, `results/paper_cvpr2027/`, and `reports/cvpr2027/`. Preserve per-run commit, dataset version and split hash, ROI extractor version, seed, conditioning fields, pair mask, loss convention, checkpoint criterion, hardware, runtime, generated sample seeds, and raw predictions. Keep a small fixed set of held-out images for qualitative panels chosen **before** seeing model outputs. The main paper needs a graphical-model diagram, a clear conditional-ELBO derivation, prediction and sample-quality tables, prior-vs-posterior samples, a latent-graph causal ablation, and failure cases. Supplementary should contain all dataset preprocessing, baselines, complete ablations, calibration details, and citation-network continuity.

## 9. Claims that must be stated precisely

- Given boxes/classes and ROI features means **conditional relation-graph generation**, not object, box, or image generation. Do not compare directly with full SGDet without a shared detector and an explicit end-to-end protocol.
- The current weighted adjacency BCE is not automatically a normalized likelihood or valid ELBO. Derive the observation model for the new typed decoder; disclose any surrogate weighting and the treatment of unknown pairs.
- A deterministic graph computed from random `Z` may vary across samples, but it is not a separately learned variational edge posterior. Claim the latter only if its posterior/prior and KL are implemented and tested.
- Posterior reconstruction, masked-edge prediction, prior sampling, and unconditional graph synthesis answer different questions. Give each its own table and do not use one as evidence for another.
- Scene-graph annotations are incomplete. Generated extra relations can be plausible despite being unannotated; qualitative examples and calibrated scores need that caveat. Report prior-sample quality and diversity together.
- Recheck the [CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) and eventual author guidelines before registration and submission.
