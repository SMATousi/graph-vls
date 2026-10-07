# CVPR 2027 implementation and experiment tasks

**Written:** 2026-10-07. **Companion to:** [`cvpr2027_paper_plan.md`](cvpr2027_paper_plan.md) (the *what and why*; this file is the *how and in what order*).

**Scope (from the plan):** conditional relation-graph generation on one dataset (Visual Genome, small-`N` subset, PredCls setting), with a learned prior over an `M`-node latent graph. JEPA, VRD, completion, stochastic `A_z`, and extra baselines are stretch items and are **not** tasked here except where noted in section 9.

Every threshold below marked **(proposed)** must be frozen in `configs/paper_cvpr2027/gates.yaml` and committed *before* the run it governs, so the go/no-go decisions cannot drift after seeing results.

---

## 0. Conventions

- **Code lives in a new package** `src/gvls/scene/` with tests in `tests/scene/`. Existing code in `src/gvls/models/` is **not modified**; the paper path reuses its ideas but not its single-graph, unbatched tensors. Existing tests must keep passing (`pytest tests/`).
- **Batching:** the new modules work on padded batches `x: (B, N_max, ·)` with a boolean `node_mask: (B, N_max)`, and latent tensors `(B, M, ·)`. Nothing in `src/gvls/models/` supports this, so do not try to reuse those classes directly. Copy the *formulas* (degree-normalised residual message passing, top-k edge budget) and re-test them.
- **Configs:** Hydra, under `configs/paper_cvpr2027/`. **Experiments:** scripts under `experiments/paper_cvpr2027/`. **Outputs:** `results/paper_cvpr2027/<run_id>/` with the provenance record from task R1.
- **ELBO convention (fixed for the whole paper, task L1):** per-image **sum** over latent nodes and dimensions for the KL, per-image **sum** over scored pairs for the likelihood, then divided by the number of annotated relations in that image **only for reporting**. Do **not** reuse `kl_isotropic`/`kl_graph_mrf` from `gvls.losses.elbo`: those are node-count-normalised surrogates (see their docstrings and `phase3/validation.md` V-8), which are not an ELBO.
- **Naming of the conditioning set:** `c = (objects, boxes, classes, ROI features)`; `R` = annotated relations.
- **Effort tags:** S ≈ ≤0.5 day, M ≈ 1 day, L ≈ 2 days. These are guesses for one engineer and will be wrong; re-estimate at the Oct 14 gate.

## 1. Workstreams and critical path

```text
C0 env/access ─► D1 data subset ─► D2 ROI cache ─► D3 loader ─► (spike) M1…M6 ─► L1/L2 ─► T0 toy ─► X1 spike ─► [Oct 14 gate]
                                                         │                                                       │
                                                         └─► B1 freq, B2 discriminative, B3 flat CVAE ───────────┘
                                                                                                                 ▼
                                                  E1…E5 metrics ─► X2 main runs ─► X3 ablations ─► X4 M-sweep ─► [Oct 28 gate] ─► P1…P4
```

The spike (X1) needs only: D1–D3, M1–M6, L1–L2, B3 (flat CVAE) and the log-score half of E1. Everything else waits for the gate. **The single biggest unknown is C0 (can this environment download Visual Genome images and run a vision backbone?);** do it first.

---

## 2. Setup and risk tasks

### C0 — Environment, compute and data-access audit (S) — *first task, blocks everything*
- Record GPU model/count, RAM, free disk, PyTorch/PyG versions (`environment.yml`, `pyproject.toml`).
- Check whether Visual Genome images and annotations can be fetched from this container. The sandbox egress proxy blocks some domains (it blocked `icml.cc` earlier in this project's work), so test the actual download hosts before assuming anything. Visual Genome images are roughly 15 GB; a small-`N` subset is much smaller.
- If images cannot be downloaded here, decide *now*: pre-compute features elsewhere and upload only the cached tensors, or switch to a data location the environment can reach.
- **Done when:** a short `reports/cvpr2027/c0_environment.md` states what works, what is blocked and the chosen data path. **If neither downloading nor uploading features is possible, the vision plan is infeasible in this environment; stop and escalate.**

### C1 — Frozen design decisions (S)
Commit `configs/paper_cvpr2027/decisions.yaml` with these choices (defaults given; change only with a recorded reason):

| ID | Decision | Default |
|---|---|---|
| D-1 | Max objects per image (`N_max`) | 20 |
| D-2 | Pair labelling | one predicate per ordered pair (keep the most frequent if several), plus a **background** class; unannotated pairs are *unknown*, not false: training uses annotated pairs plus `n_neg` sampled unannotated pairs as background (`n_neg = 3 × #positives`, proposed) |
| D-3 | Latent size `M` | 6 (`M/N` ≈ 0.3 at `N=20`); sweep in X4 |
| D-4 | Latent dim `d` | 64 |
| D-5 | Latent edge budget `k_z` | 2 with union-symmetrisation, so degree ≤ 4 < `M-1` = 5; asserted in code, never complete |
| D-6 | `A_z` type | deterministic function of sampled `Z` (no edge KL); undirected |
| D-7 | Prior | conditional diagonal Gaussian `p_psi(Z | c)` |
| D-8 | ROI feature extractor | one frozen ImageNet ResNet with ROIAlign, version and weights hash recorded |
| D-9 | Seeds | 3 (`0, 1, 2`) |

### C2 — Pre-registered gates (S)
Write `configs/paper_cvpr2027/gates.yaml` containing every number used by the Oct 14 and Oct 28 decisions (section 8) and commit it before the spike run.

---

## 3. Data tasks

### D1 — Dataset subset and split (M)
- Parse Visual Genome object/relationship annotations into the VG150-style vocabulary (150 object classes, 50 predicates). Use the standard split definition if it can be reproduced; otherwise define one and document it.
- Keep images with `3 ≤ N ≤ N_max` annotated objects and ≥ 1 relation. Do not subsample objects inside an image (that would change the relation set).
- Image-level train/val/test split; write image IDs and a **split hash** to `results/paper_cvpr2027/data/split.json`.
- **Tests:** no image ID appears in two splits; every relation's subject/object index is within `[0, N)`; predicate labels are in the vocabulary; counts printed and recorded (images, objects, positive pairs, predicate histogram).
- **Done when:** a report lists dataset statistics and the predicate frequency distribution; `n_neg` and the unknown-pair rule are justified against the measured positive-pair density.

### D2 — ROI feature cache (M)
- Extract one feature vector per ground-truth box with the frozen extractor (D-8); store as fp16 tensors keyed by image ID plus object index.
- Record extractor name, weights hash, input resolution and ROI-pool settings in the cache metadata.
- **Tests:** deterministic (two extractions of the same image match within tolerance); feature count equals object count for every image.

### D3 — Batched typed-graph loader (M)
- `src/gvls/scene/data.py`: yields `x_cls (B,N)`, `box (B,N,4)` normalised to `[0,1]`, `roi (B,N,F)`, `node_mask`, and relations as a dense label tensor `rel (B,N,N)` with values `0..50` (0 = background) and a boolean `pair_known (B,N,N)` marking annotated pairs. The diagonal and padded pairs are masked out everywhere.
- Relative box geometry per ordered pair (`dx, dy, log(w_j/w_i), log(h_j/h_i)`, IoU, relative area) computed once.
- **Tests:** permuting object order together with `rel` leaves any permutation-equivariant statistic unchanged; collation of variable-`N` images gives the right masks.

---

## 4. Model tasks (all in `src/gvls/scene/model.py` unless noted)

### M1 — Object encoder (M)
Per-object features `h_i` from class embedding, box geometry and ROI feature, followed by a small set-transformer or GNN layer over objects (so each `h_i` sees the whole inventory). **Uses only `c`.**
- **Test:** `h` does not change when `rel` is changed (guards against leakage).

### M2 — Shared assignment `S(c)` (M)
`S = softmax_M(f(h_i))` per object (DiffPool-style, `S ∈ [0,1]^(N×M)`), masked for padded nodes. Depends only on `h` from M1.
- Return and log: occupancy per latent node, assignment entropy, and `S` under hardening (argmax) for later comparison.
- **Tests:** rows sum to 1 for real nodes and are zero for padded ones; **gradient/path test proving `S` has no dependency on `rel`** (change `rel`, outputs identical).
- **Risk:** assignment collapse (all objects to one cluster) is the known failure of the old `PooledGVLS` (`phase3/validation.md` V-8). Add an optional entropy/occupancy regulariser (task L3) and always log occupancy.

### M3 — Pooling (S)
`P = S^T H` normalised by occupancy (so scale does not depend on `N`), giving `(B, M, d_h)`.

### M4 — Prior and posterior (M)
- Prior `p_psi(Z | c)`: head on `P` giving `mu_psi, logvar_psi ∈ (B, M, d)`.
- Posterior `q_phi(Z | c, R)`: a relation encoder pools the annotated relations onto object nodes (embedding of predicate plus neighbour features), then pools with the **same** `S` and combines with `P`.
- **Tests:** the prior path never receives `rel` (assert by hook or by comparing outputs with `rel` removed); KL of identical Gaussians is 0 and matches `torch.distributions.kl_divergence`.
- Reparameterised sampling in training; at test time sample from the prior.

### M5 — Latent graph `A_z` (M)
Edge scorer on `Z` (start with scaled dot-product attention; allow an MLP scorer), per-image top-`k_z` with union-symmetrisation, zero diagonal, mask out nothing (all `M` latent nodes are real). Reuse the logic of `LatentGraphLearner`, batched.
- **Tests:** max degree ≤ `2·k_z`; **assert `2·k_z < M-1`** so the graph can never be complete; symmetric; zero diagonal; gradients flow to the scorer.
- **Note:** `attention` has no learnable scorer parameters. For `A_z` to carry learned structure, either use the MLP scorer or put learned projections on `Z` before the dot product. Decide in the spike and record why.

### M6 — Latent message passing and typed decoder (L)
- `Z̃`: `L ≥ 1` rounds of degree-normalised residual message passing over `A_z` (same formula as `LatentMessagePassing`, batched). Because the old code showed that `mp_rounds=0` makes `A_z` irrelevant, the main model **must** use `L ≥ 1`.
- Unpool to objects: `u_i = Σ_m S_im Z̃_m`.
- Pair scorer: MLP over `[h_i, h_j, u_i, u_j, geometry_ij]` producing logits over 51 classes (background + 50 predicates).
- **Tests:** shape and mask correctness; output changes when `A_z` changes (non-zero `||∂logits/∂A_z||`); no label leakage.

### M7 — Sampler API (M)
`sample(c, seed, num_samples)` in `src/gvls/scene/sampling.py`: draw `Z ~ p_psi(·|c)`, build `A_z`, run M6, sample a predicate per ordered pair from the categorical (including background), return relation graphs. **Never accepts `R`.**
- **Tests:** identical seeds reproduce samples; different seeds differ; calling with `rel` removed from the batch works (the function signature must not take it); output contains background and non-background pairs.
- Also provide `mean_prediction(c, n_samples)` (average probabilities over prior samples) for the R@K protocol.

---

## 5. Loss and training tasks

### L1 — Conditional ELBO (M)
`src/gvls/scene/losses.py`:
```text
loss = -Σ_{pairs in train set} log p_theta(r_ij | c, S, Z, A_z) + β · KL(q_phi(Z|c,R) || p_psi(Z|c))
```
where the pair set is annotated pairs plus `n_neg` sampled unannotated pairs (D-2) and `β` is annealed from 0 to its target. State plainly in the code and paper that, with negative sampling, this is a **surrogate**, not a normalised ELBO; log likelihood estimates are computed separately (E1).
- Closed-form diagonal-Gaussian KL, **summed** over `M·d`, per image.
- **Tests:** KL matches `torch.distributions`; loss decreases on a tiny overfit batch; total loss is additive over images.

### L2 — Trainer (M)
Training loop with Hydra config, seeds, AdamW, gradient clipping, mixed precision optional, checkpointing on **validation generative score** (not training reconstruction), per-step logging of: likelihood term, KL, `β`, KL per latent node and dimension, active-dimension count (units with posterior-vs-prior divergence above a threshold), `||∂L/∂A_z||`, `A_z` density and complete/empty fraction, `S` occupancy and entropy.
- **Done when:** a run on a few hundred images overfits and logs every item above.

### L3 — Assignment regulariser (S, optional)
Entropy/occupancy term to avoid collapsed `S`. Include it only if the spike shows collapse; its weight is a logged hyperparameter. It must not read `R`.

### L4 — Loss audit (S)
After the first real run, report each loss component per image and per pair, and the relative size of the KL term. A KL below 1% of the loss (the Phase 5 finding for the jet model) triggers `β` and free-bits adjustment *before* any claim about the prior.

---

## 6. Baseline tasks

### B1 — Frequency and geometry baseline (S)
`p(predicate | subj class, obj class)` from training counts, with a variant that adds box-geometry bins. No learned latent.

### B2 — Discriminative pair classifier (M)
Same M1 encoder and same M6 pair-scorer shape, **no latent at all**. This is the "what does the latent cost" reference for R@K. It also provides the strongest sanity check on the data pipeline: its R@K should be in a plausible range for a PredCls model on VG150 (compare against published PredCls numbers *qualitatively* only; do not claim parity).

### B3 — Flat conditional VAE (M)
`N` per-object latents with per-object conditional prior and posterior, the same decoder body, no pooling, no `A_z`. **Parameter-match to the GVLS model within 5%** (adjust hidden width) and record parameter counts. This is the baseline the Oct 14 and Oct 28 gates compare against.

### B4 — Pooled model without latent graph (S)
GVLS with `S`, `M` latents, prior/posterior, but no `A_z` and no message passing.

### B5 — Fixed-graph variants (S)
Same as GVLS but `A_z` is replaced by (a) a random graph with the same density, (b) a ring/`k_z`-regular graph, (c) the complete graph. These are *trained* variants; the inference-time perturbations are in E4.

---

## 7. Evaluation tasks (`src/gvls/scene/eval/`)

### E1 — Held-out log scores (M)
- **Positive-pair score:** mean `log p(r_ij | c)` over annotated pairs with the predicate distribution renormalised over the 50 non-background classes (answers "which predicate, given a relation exists").
- **Full score:** mean over annotated pairs plus the same fixed set of sampled unannotated pairs used for training (seeded), so the number is comparable across methods.
- Because `Z` is latent, estimate `log p(R|c)` with an importance-weighted bound using the learned `q` as proposal (`K = 50`, report `K = 1, 10, 50` to show convergence).
- **Tests:** on a toy case with a known likelihood the estimator converges; the bound is ≥ the single-sample ELBO in expectation.

### E2 — Relation prediction R@K / mR@K (M)
`R@20/50/100`, `mR@20/50/100` with the graph constraint and candidate-pair policy stated in the report; computed from `mean_prediction` over `n` prior samples (and, separately, from the posterior as a reference that is **never** reported as generation).

### E3 — Generation-quality metrics (L)
Per held-out image, draw `S = 10` independent prior samples:
- **Sanity:** fraction of empty graphs, fraction with a complete relation set, exact-duplicate rate across samples.
- **Coverage/precision at a budget:** with budget `B` = number of annotated relations, union recall of annotated triplets over samples, and per-sample precision. State clearly that precision against incomplete annotations is a lower bound on plausibility.
- **Diversity:** mean pairwise Jaccard distance between sampled triplet sets, reported **together with** coverage.
- **Distributional match to held-out graphs:** relation-count distribution (Wasserstein-1), predicate-frequency distribution (total variation), top-100 `(subject, predicate, object)` triplet frequency (KL), object degree distribution (Wasserstein-1).
- **Tests:** each metric returns its known value on hand-built graphs (identical graphs give distance 0, disjoint give 1, etc.).

### E4 — Latent-graph causal audit (M)
For a trained GVLS model, evaluate E1 with the following **at inference only**: `A_z` set to zero; `A_z` shuffled across images in the batch; `A_z` replaced by a fixed random graph of equal density; plus the trained variants B4 and B5. Also report `||∂L/∂A_z||`, density, complete/empty fraction and cross-image variance of `A_z`.
- **Done when:** one table gives, per perturbation, the change in positive-pair and full log score with seed-level mean ± std and a paired bootstrap interval over images.

### E5 — Compute and size (S)
Parameters, training time per epoch, peak memory, inference time per image as a function of `S`, and the latent size in values (`M·d` for GVLS vs `N·d` for the flat model). Latent *storage* claims exclude the conditioning set `c`.

### E6 — Calibration (S, supplementary)
ECE and Brier score on pairs with reliable labels only; skip if time is short.

---

## 8. Experiments and gates

### T0 — Toy sanity check (S) — before any real data
Synthetic data where the relation between two objects depends on a hidden cluster variable shared by several objects and the number of latent clusters is known. The GVLS model must (a) recover the cluster structure in `S`, (b) generate samples matching the true relation statistics, (c) show a measurable `A_z` effect **if the data was built to require one** (e.g. relations depend on a link between clusters). If the toy test fails, there is a bug and nothing on VG is interpretable.

### X1 — Feasibility spike on real data (L) — finishes the Oct 14 gate
Run GVLS and the flat CVAE (B3) and the discriminative baseline (B2), **one seed**, on the VG subset (or a tuned-down sub-subset if compute is short).

**Oct 14 gate (all thresholds proposed, frozen in `gates.yaml`):**
1. Prior sampler produces graphs with a non-degenerate relation count (not empty, not complete) and duplicate rate < 50% across 10 samples.
2. GVLS positive-pair log score within **0.05 nats** of the flat CVAE (B3); the discriminative baseline (B2) is reported as the reference.
3. Setting `A_z` to zero at inference changes the positive-pair score by at least **0.005 nats** (any effect at all; the strict threshold is applied at Oct 28).
4. Posterior is not collapsed: at least 25% of latent dimensions active.

**If any of these fail:** diagnose with L4 (loss audit), try the single prescribed fixes (free bits, `β` schedule, MLP edge scorer, assignment regulariser L3) for **at most two days**, then take the fallback in the plan, section 8.

### X2 — Main runs (L)
GVLS, B1, B2, B3, B4 with seeds `0,1,2` at the frozen configuration. Select checkpoints on validation positive-pair log score. Produce the tables for E1, E2, E3, E5.

### X3 — Causal ablations (M)
E4 on all GVLS seeds, plus the trained B5 variants (random, ring, complete graph) with three seeds.

### X4 — Sensitivity (M)
`M ∈ {3, 6, 10}`, `k_z ∈ {1, 2, 3}` (with the `2·k_z < M-1` constraint), `L ∈ {1, 2}`, `β` target. One seed each unless a difference is within noise, then rerun with three.

**Oct 28 go/no-go (thresholds proposed, frozen):**
1. GVLS positive-pair log score ≥ flat CVAE (B3) **minus 0.02 nats**, on the mean over three seeds.
2. Zeroing or shuffling `A_z` at inference degrades the positive-pair score by **≥ 0.01 nats and ≥ 2 seed-level standard deviations**, and the trained no-`A_z` model (B4) is worse than full GVLS by a margin that exceeds the seed standard deviation.
3. GVLS is competitive with B2 on R@50/mR@50 (within 2 points absolute, proposed), or the gap is reported with an explanation.
4. Prior samples pass the sanity checks of E3 on all three seeds.

All four must hold for a generative-GVLS claim. If (2) fails, report it as a negative result and take the plan's fallback; do not reword the claim to fit.

### X5 — Qualitative panels (S)
Using the image IDs chosen **before** any model output (the plan requires this), render several samples per image with prior and posterior side by side, including failure cases. Store the choice of IDs in `results/paper_cvpr2027/panels.json` with its commit hash.

---

## 9. Reproducibility and paper tasks

### R1 — Provenance record (S)
Each run writes `provenance.json`: git commit, dataset version, split hash, ROI extractor version, seed, config, loss convention, checkpoint criterion, hardware, runtime, sample seeds, raw predictions path. A script `experiments/paper_cvpr2027/check_provenance.py` fails if any table cell cannot be traced to a run.

### P1 — Method and ELBO write-up (M)
Graphical model, conditional ELBO with the surrogate caveat from L1, definition of `A_z` as a deterministic function of `Z`, the explicit statement that no edge KL exists.

### P2 — Results tables and figures (M)
Prediction table (E2), log-score and generation table (E1/E3), causal ablation (E4), compute (E5), and the sample panels (X5).

### P3 — Limitations (S)
Incomplete annotations, deterministic `A_z`, single dataset, PredCls-only setting, subset of VG, any gate that passed narrowly.

### P4 — Supplement and code audit (M)
Preprocessing, all configs, full ablations, calibration (E6), any citation-network continuity note drawn from *existing* results only.

### Stretch (not tasked; start only after the Oct 28 gate passes and only if they cannot delay P1–P4)
In the plan's order: 5-seed reruns, VRD, stochastic `A_z` with KL, masked completion, Neural Motifs/VarScene comparisons, JEPA auxiliary objective, Open Images. JEPA was explicitly deferred; it must not enter the main claim.

---

## 10. Schedule mapped to the plan

| Window (2026) | Tasks | Exit condition |
|---|---|---|
| Oct 7–9 | C0, C1, C2, D1 | Data path confirmed; decisions and gates frozen; subset statistics reported |
| Oct 9–11 | D2, D3, M1–M5 | Features cached; model components pass unit tests |
| Oct 11–13 | M6, M7, L1, L2, T0, B3 | Toy passes; overfit on a small batch; flat CVAE trains |
| Oct 13–14 | X1 | **Oct 14 gate** |
| Oct 15–21 | L3/L4 as needed, B1, B2, B4, B5, E1–E3 | First three-seed run in progress |
| Oct 22–28 | X2, X3, X4, E4, E5 | **Oct 28 go/no-go** |
| Oct 29–Nov 4 | X5, final runs, P2 tables | Configs frozen; provenance check passes |
| Nov 5–9 | P1, P3, supplement draft, register | Registered by Nov 10 |
| Nov 10–16 | Final paper | Submit by Nov 16 AoE |
| Nov 17–23 | P4 | Supplement by Nov 23 AoE |

**Feasibility reminder:** this schedule assumes the C0 audit shows a working data path and enough GPU time for three seeds of five methods. Section 1's critical path is unforgiving: a slip of more than two days before Oct 14 means the Oct 28 decision becomes about whether to submit a narrower paper. The gates exist so that decision is made on measurements.

## 11. Risks and the first response to each

| Risk | Signal | First response |
|---|---|---|
| Data or image download blocked | C0 fails | Pre-compute features off-site and upload tensors; otherwise stop and re-plan |
| Posterior collapse | Few active dimensions, KL < 1% of loss | Free bits, longer `β` warm-up, lower `β` (L4) |
| Assignment collapse | One cluster holds most objects | Occupancy regulariser (L3), temperature on `S` |
| `A_z` decorative | Zeroing it changes nothing | MLP edge scorer, ≥ 2 rounds, check `||∂L/∂A_z||`; if still nothing, report it and take the fallback |
| Prior samples unrealistic while posterior is fine | High posterior R@K, bad sample metrics | Stronger conditional prior, check the KL weight; do not publish reconstructions as generation |
| Flat CVAE wins clearly | Gate 1 fails | Report it; the contribution reduces to a negative result, and the fallback applies |
| Sparse annotations make metrics noisy | Large seed variance, unstable ranking | Positive-pair score as the headline; bootstrap over images; report unknown-pair sensitivity |
| Compute shortfall | Epoch time too high for 15 runs | Reduce subset size, then reduce `S` and seeds in that order, and record the change |
