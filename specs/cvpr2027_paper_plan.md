# CVPR 2027 paper preparation plan: conditional relation-graph generation with a learned latent graph

**Written:** 2026-10-07. **Revised:** 2026-10-07 (twice): first to center graph generation, then to cut the scope to what can plausibly be finished by the submission date.

**Target:** CVPR 2027 main conference, with an explicit fallback (section 8) if the early gates fail.

**Implementation and experiment tasks:** see [`cvpr2027_implementation_tasks.md`](cvpr2027_implementation_tasks.md).

**Scope:** classical Graph Variational Latent Space (GVLS) applied to **one** visual task on **one** dataset. QGNN, jet classification, citation-network results, JEPA training, a second dataset and scene-graph completion are *not* part of the core plan; they are listed as stretch items in section 7 and are cut first. Proposed experiments below are unrun unless explicitly identified as existing results.

## 1. Thesis, minimum paper, and submission clock

**Proposed question:** Given an image and a fixed object inventory (boxes and class labels), can a smaller, learned *graph of latent distributions* model a distribution over directed, typed relationships, such that sampling from a learned prior gives plausible, varied relation graphs? We call this *conditional relation-graph generation*. It is not unconditional graph generation, object generation or image generation.

**Minimum paper (the only thing the plan commits to):**

1. A conditional variational model with a learned prior that, at test time, samples relation graphs without reading ground-truth relations.
2. Held-out relation prediction that is competitive with a matched no-latent discriminative predictor, and held-out annotation log score that is competitive with a matched **flat** conditional VAE.
3. A causal ablation showing the learned latent graph `A_z` changes the output: zeroing, shuffling or fixing `A_z` at inference measurably hurts, and retraining without it hurts too.

If (3) fails, there is no GVLS-specific contribution and the paper should not be submitted as one. Reconstruction F1 from the encoder's posterior alone establishes none of these three.

The [official CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) lists paper registration on **November 10, 2026**, main submission on **November 16, 2026** and supplementary material on **November 23, 2026**, all Anywhere on Earth and stated to be fixed. All authors need current OpenReview profiles. The 2027 author-guidelines page was unavailable when this plan was written; check the final template, length, anonymity and code rules when published. The [2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) used an eight-page main paper, which is only a working assumption.

**Honest feasibility note:** the repository has no vision code (no loader, ROI features, typed decoder or conditional prior), and about 40 days remain. This plan is built so that the minimum paper is reachable by October 28 only if the first-week spike (section 8) goes well. Treat every item outside the minimum paper as optional.

## 2. What must change from the current repository

The [mission](mission.md) describes a variational encoder, learned pooling `S`, latent adjacency `A_z`, message passing, graph-aware KL and unpooling. [Phase 3 validation](phase3/validation.md) shows reconstruction F1 below 0.90 on Cora/CiteSeer/PubMed, flat across the whole `(d, k)` grid, and some selected configurations (`mp_rounds=0`) give `A_z` no route into the output. [Phase 5 validation](phase5/validation.md) found the variational loss weak and the jet latent graph complete under its production configuration. These are constraints to investigate, not results to hide.

The current `PooledGVLS` is an **autoencoder for an observed graph**. At decoding it reuses `S`, computed from that graph's encoded nodes, and returns an undirected binary adjacency. It has no learned prior that can produce `S`, node attributes or relation types at test time, and its deterministic top-k adjacency is complete whenever `k >= M-1`. Sampling a standard Gaussian and calling the existing decoder is not a valid standalone generator.

The reduced scope is therefore **conditional** generation: object count, boxes, labels and ROI features are given at training and test time. Derive `S` from that shared conditioning information so the decoder can use the same assignment at inference. Only the relation graph is modelled. Report the storage cost of the latent separately if claiming a compression advantage; the conditioning information is excluded from that cost.

## 3. Generative model specification

Let `c` contain an image, a fixed set of `N` object boxes/classes and fixed ROI features. Let `R` be the annotated directed predicate graph, with a declared treatment of unannotated ordered pairs. Let `S(c) in [0,1]^(N x M)` be an assignment learned **only from c**, with `M < N` where feasible.

```text
p(Z, A_z, R | c) = p_psi(Z | c) p_omega(A_z | Z, c)
                    product_(i != j) p_theta(R_ij | c, S(c), Z, A_z).
q_phi(Z, A_z | c, R) approximates the posterior during training.
```

`Z` is `M` latent node variables. `A_z` is a latent edge variable whose symmetric-or-directed convention must be chosen and documented; the *output* relation graph is directed and typed regardless. The training objective is the conditional ELBO on the observed annotation set:

```text
E_q [log p_theta(R_observed | c, S, Z, A_z)]
    - KL(q_phi(Z, A_z | c, R) || p_psi(Z, A_z | c)).
```

**Simplifications adopted for the reduced scope:**

- `A_z` is a **deterministic** function of sampled `Z` (edge scorer plus a hard edge budget that cannot yield a complete graph). Its edge distribution is induced by `Z`, so there is no separate edge KL and the paper must not claim a variational edge posterior. Bernoulli/Concrete edges with their own KL are a stretch item.
- The prior is a **conditional diagonal Gaussian** `p_psi(Z | c)`. Mixture or graph-MRF priors only if an ablation points to a limitation.
- A conditional prior is required even in this minimal version, because samples and test likelihoods must not depend on ground-truth `R`.

### Architecture tasks

1. **Data adapter:** per image, nodes carry class, box geometry and frozen ROI features. Fix the ROI extractor and cache features across *all* methods. Use ground-truth boxes/classes (PredCls). Restrict to a **documented subset** (for example images with at most 20 objects) to keep `N` small, so `M < N` is meaningful and training is cheap; report the subset rule and image IDs.
2. **Shared assignment:** `S(c)` may use only information available at generation time, never relation edges, targets or posterior-only features. Record `M/N`, cluster occupancy and whether hardening `S` changes samples. A latent size not smaller than `N` is a baseline setting, not compression.
3. **Posterior and prior:** `q_phi(Z | c,R)` and `p_psi(Z | c)` with matched Gaussian shapes; sample from `q` in training, from `p` at test time. Track posterior/prior KL, active dimensions, variance and collapse.
4. **Latent graph:** infer `A_z` from sampled `Z` with an edge scorer and a hard budget. Compare against a parameter-matched graph-free model (section 6).
5. **Typed decoder:** unpool latent context to object nodes through `S(c)`, then score each ordered pair over predicate classes plus a documented background/unknown treatment. The pair scorer must see object/ROI features and **processed** latent-graph features (after message passing over `A_z`), not only raw `Z`; otherwise `A_z` is decorative. Specify whether pairs can carry several predicates (multilabel likelihood) or one (include a background class).
6. **Sampler:** `sample(c, seed, num_samples)` must never read ground-truth `R`. Do not substitute posterior reconstructions.
7. **Loss audit:** log relation likelihood, node KL and any auxiliary assignment term per image, with reductions over object pairs that are comparable across graph sizes. If a weighted surrogate replaces the stated ELBO, name it and keep likelihood reporting separate from the training loss.

## 4. Dataset and protocol

| Role | Dataset | Notes |
|---|---|---|
| **Only required dataset** | [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), VG150-style split, small-`N` subset | Freeze image IDs, vocabulary, boxes, ROI extractor, annotation preprocessing and split hash. Image-level train/val/test separation. |
| Stretch, only after the minimum paper passes | [VRD](https://cs.stanford.edu/people/ranjaykrishna/vrd/) | Smaller second dataset under its own vocabulary and official test split; retrain, do not call it transfer. |
| Supplement, only if already available | Cora, CiteSeer, PubMed | Existing link-prediction diagnostics as a continuity note. They do not validate a visual relationship generator; do not spend new compute on them. |

Standard [PredCls evaluation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.pdf) uses given boxes/classes; do not compare numbers with SGCls or SGDet. VG relations are sparse and incomplete ([limited-label scene graphs](https://cs.stanford.edu/people/ranjaykrishna/limitedlabels/index.html)), so unannotated pairs are *unknown*, not false, unless a benchmark protocol says otherwise. Include a negative-sampling sensitivity check.

## 5. Evaluation (reduced to what supports the minimum paper)

**Headline metric, chosen now:** held-out **annotation log score** (per-annotated-pair log likelihood of the observed predicate, plus a documented treatment of unknown pairs), using an importance-weighted bound where the prior makes the exact value intractable. This is the metric the go/no-go gate uses. Do not use precision against sparse labels as the headline.

**Required tables:**

1. **Relation prediction:** standard `R@20/50/100` and `mR@20/50/100` with the graph constraint and candidate-pair setting stated, versus the no-latent discriminative predictor and the flat CVAE. Report frequency strata or per-predicate results, not only a head-class mean.
2. **Generation quality:** for each held-out image, draw several samples from `p(Z,A_z|c)`. Report positive-relation coverage under a fixed output budget, diversity at matched quality (unique triplet sets, pairwise set distance) and graph-statistic distributions (relation counts, predicate frequencies, node degrees) versus held-out graphs. Show several samples for the **same** image, including failures. Diverse but implausible or duplicate samples are not success.
3. **Latent-graph causal audit:** the section 6 ablations, plus `||dL/dA_z||`, density, complete/empty fraction and cross-image variation of `A_z`.
4. **Compute:** parameters, train/inference time, peak memory, number of prior samples used, and sensitivity to `M`.

**Protocol:** three seeds for headline rows (raise to five only if time remains), frozen hyperparameters, mean ± standard deviation. Select checkpoints on validation **generative** metrics, never on training reconstruction. Calibration (NLL/Brier/reliability) is reported only where annotations are reliable and is otherwise supplementary.

## 6. Baselines and ablations (minimum set)

| Comparison | Question |
|---|---|
| Frequency/geometry/ROI-only predictor and a plain discriminative pair-classifier on the same ROI features | Is the model competitive at the relation task, and what does the latent cost? A stronger published PredCls model such as [Neural Motifs](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zellers_Neural_Motifs_Scene_CVPR_2018_paper.pdf) is a stretch item. |
| Conditional **flat** VGAE/CVAE with `N` object latents and a matched decoder | Does the compressed latent graph add value over a conventional variational graph autoencoder? Parameter-match it. |
| Learned pooling **without** `A_z`; fixed/random/complete latent graph; full GVLS | Is the learned topology causal? |
| `M` and edge budget; deterministic vs stochastic edges is a stretch | What controls likelihood, sample quality and collapse? |
| Posterior reconstruction vs prior sampling | Does the model genuinely generate at test time? |

For the causal `A_z` test, set `A_z` to zero, shuffle it across images and replace it with a fixed graph **at inference**; separately retrain without it to distinguish test-time perturbation from adaptation. Report negative results. Do not claim to be the first variational scene-graph generator: [GraphVAE](https://arxiv.org/abs/1802.03480) and [VarScene](https://proceedings.mlr.press/v162/verma22b.html) are prior work, and VarScene is a stretch comparison only if a matched setting can be set up cheaply.

## 7. Stretch items, cut first and in this order

1. Longer seed lists (5 instead of 3) and the full compute/`M` sweep.
2. VRD confirmation run.
3. Stochastic `A_z` (Bernoulli/Concrete with KL), only if it demonstrably improves a defined metric.
4. Masked relation completion (`complete(c, R_context, mask, seed)`), with target relations removed from every context, assignment and pair-feature path before encoding.
5. Neural Motifs and VarScene comparisons.
6. JEPA-style auxiliary objective ([I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf), [Graph-JEPA](https://arxiv.org/abs/2309.16014)). It is not a graph likelihood and not part of the core claim; keep it only if it improves a prespecified held-out generative metric under matched compute.
7. Open Images V6 relationships and detector-predicted boxes.

## 8. Work plan and gates

| Date (2026) | Tasks and deliverable | Gate |
|---|---|---|
| Oct 7–14 | **Feasibility spike:** VG small-`N` subset loader with frozen ROI features; shared `S(c)`; conditional prior and posterior; typed decoder; prior-only sampler; flat CVAE baseline on the same data. | **Oct 14 gate:** prior sampler runs without `R`; samples are not trivially empty/complete/duplicated; GVLS is within reach of the flat CVAE on held-out log score; zeroing `A_z` changes the output at all. If not, switch to the fallback below. |
| Oct 15–21 | Fix the issues the spike exposed (assignment, KL convention, edge budget); add no-`A_z` and discriminative baselines; first three-seed run. | Comparable numbers for GVLS, flat CVAE and discriminative predictor. |
| Oct 22–28 | `A_z` causal audit, generation-quality tables, hyperparameter freeze. | **Oct 28 go/no-go:** GVLS beats or matches the matched flat CVAE on held-out log score *and* `A_z` ablations show a measurable effect. |
| Oct 29–Nov 4 | Final three-seed runs and figures (sample panels, causal ablation). Only start stretch items that cannot delay the paper. | Protocols and configs frozen; every number maps to a run, seed and checkpoint. |
| Nov 5–9 | Write method/ELBO derivation, results, limitations; supplement; anonymization; register by Nov 10. | Registered. |
| Nov 10–16 | Final paper and submission under the actual 2027 rules. | Submit by Nov 16 AoE. |
| Nov 17–23 | Supplementary and code audit. | Upload by Nov 23 AoE. |

**Fallback if the Oct 14 or Oct 28 gate fails:** do not submit a generative claim on posterior reconstructions. The recorded options are (a) a narrower representation or negative-results write-up for a workshop, or (b) a fuller study aimed at a later venue. The decision is made on the measured gate, not by changing what was measured.

Create `configs/paper_cvpr2027/`, `experiments/paper_cvpr2027/`, `results/paper_cvpr2027/` and `reports/cvpr2027/`. Per run, preserve commit, dataset version and split hash, ROI extractor version, seed, conditioning fields, pair mask, loss convention, checkpoint criterion, hardware, runtime, sample seeds and raw predictions. Choose a small set of held-out images for qualitative panels **before** seeing model outputs.

The main paper needs a graphical-model diagram, the conditional-ELBO derivation, the prediction and generation tables, prior-vs-posterior samples, the latent-graph causal ablation and failure cases. Everything else goes in the supplement.

## 9. Claims that must be stated precisely

- Given boxes, classes and ROI features means **conditional relation-graph generation**, not object, box or image generation, and no direct comparison with SGDet without a shared detector.
- The weighted adjacency BCE in the current code is not automatically a normalized likelihood or valid ELBO. Derive the typed decoder's observation model and disclose any surrogate weighting and the unknown-pair policy.
- A deterministic `A_z` computed from random `Z` varies across samples but is not a learned variational edge posterior.
- Posterior reconstruction, prior sampling and (if added) completion answer different questions; give each its own table and never use one as evidence for another.
- Annotations are incomplete: an unannotated generated relation can be plausible. Report sample quality and diversity together and state that caveat next to qualitative examples.
- Recheck the CVPR 2027 call and final author guidelines before registration and submission.
