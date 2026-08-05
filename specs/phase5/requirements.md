# Phase 5 — Requirements

## Functional Requirements

### FR-1: Consistent per-jet ELBO normalization (T5.1)
- `elbo()`'s reconstruction and KL terms must be normalized on a consistent
  basis, so that the effective KL weight a jet is trained under does **not**
  depend on that jet's particle count `N`
- Current defect: reconstruction is mean-reduced over `N²` node pairs while
  `kl_isotropic`/`kl_graph_mrf` divide by node count `M`, giving
  `β_eff = β · N²/M`. Measured, `N²` spans 100–19,321 (193×) and averages
  3,092 vs 1,315 across the two classes — the regularization is **2.35×
  stronger on one class than the other**, and `corr(N, label) = −0.556`
- Acceptance: for a fixed `β`, the KL-to-reconstruction ratio is invariant to
  `N` within floating-point tolerance across synthetic jets of very different
  size
- The chosen convention must be documented in `elbo()`'s docstring alongside
  (and explicitly superseding) the T3.6/`specs/phase3/validation.md` V-8
  node-count normalization history
- Phases 0–3's citation-network behavior operates at a single fixed `N`; the
  change must not silently alter their loss *ranking* (`plan.md` T5.1)
- `train_pooled_gvls_on_jets` must expose a checkpoint-selection criterion
  other than validation reconstruction F1, which correlates only `+0.245`
  with downstream accuracy in the usable `β` range (`plan.md` Design
  Decision 2); whichever criterion a run uses must be recorded with its
  results
- Because this changes the loss landscape for every jet run, the `(k, β)`
  operating point must be re-established afterwards — the motivating sweep's
  `β` conclusions were drawn under the old convention and do not
  automatically carry over

### FR-2: Variational output reaches the downstream task (T5.2)
- `extract_latent_features` must be able to carry the pooled posterior's
  `log_var` (and `mu`, if it differs usefully from `z̃`) into `JetFeatures`,
  not only `(z̃, A_z)`
- Current defect: `mu`/`log_var` are discarded at extraction and, because
  extraction runs under `model.eval()`, `z̃` is the deterministic mean path —
  **no classifier in this project has ever seen a variance**
- `jet_features_to_array` must support a selectable feature set, and the
  ablation `(z̃, A_z)` vs `(z̃, A_z, log_var)` must be measured and reported
- Backward compatibility is required: the default feature set must be
  byte-identical to today's, so Phase 4's numbers remain reproducible and
  existing checkpoints keep loading
- The QGNN's own circuit input is **not** changed by this requirement
  (`plan.md` Design Decision 3)

### FR-3: Occupancy-aware pooled posterior (T5.3)
- `LatentGraphPooling` must be able to scale the pooled Gaussian's precision
  by un-normalized cluster mass `n_m = Σ_i S[i,m]`, so that the pooled
  posterior's variance encodes how many input particles a cluster summarizes
- Current defect: pooling uses column-normalized weights (`w = s / col_sum`),
  so moment matching runs *after* occupancy has been divided out; a cluster
  pooling 2 particles and one pooling 60 produce identically-scaled Gaussians
- Justification is statistical, not heuristic: the mean of `n` samples has
  variance `σ²/n`
- Must be gated behind a constructor flag whose default reproduces current
  output bit-identically, so the change is an A/B and Phase 3/4 remain
  reproducible
- Must not produce NaN or infinite precision for empty or near-empty clusters
- The concatenation control (`(z̃, A_z) + N`, measured at 0.7683) must be
  reported alongside, per `plan.md` Design Decision 5

### FR-4: Learned mixture prior (T5.4)
- A `K`-component learned mixture prior (VampPrior-style pseudo-inputs or
  VaDE-style free parameters — one, with the choice documented) must be
  selectable in place of `N(0,I)`
- Rationale: every measured `β` increase reduced downstream accuracy
  monotonically (0.7068 → 0.6343 across `β` 0.001 → 1.0), and at `β=1.0`
  isotropic the posterior collapses to the prior (`σ = 0.89`) with
  reconstruction at the trivial floor. An isotropic prior pulls all jets to a
  single mode, which is precisely what removes between-class separation
- KL to a mixture has no closed form; a single-sample Monte Carlo estimate
  using the already-drawn reparameterized sample is acceptable
  (`specs/tech_stack.md` already sanctions this fallback)
- Must support free bits (`max(KL_d, λ_free)` per dimension) and a `β`
  warm-up schedule — without them this cannot be evaluated at any `β` where
  the prior matters, since everything above `β=0.01` currently degrades
  (`plan.md` Design Decision 6)
- `K=1` must reduce to the isotropic case within tolerance
- Gradients must reach the prior's own parameters

### FR-5: Graph-MRF `λ` sweep (T5.5)
- `lambda_` must be swept (e.g. `{0.1, 1, 10, 100}`) jointly with `β`; it has
  been hardcoded at 1.0 in every run this project has performed
- Rationale: `Ω = I + λ·L_z` penalizes disagreement between *connected*
  latent nodes rather than magnitude — a smoothness prior rather than a
  shrinkage prior — and is the only place any positive evidence for a
  graph-structured prior exists in this project: at `β=1.0` graph_mrf beat
  isotropic by **+10.3 points** (0.6858 vs 0.5827) and avoided the collapse
  isotropic suffered (F1 0.7254 vs 0.6822)
- `λ=0` must reduce `kl_graph_mrf` to the isotropic case within tolerance
- The log-det term must stay finite and first-order (the existing `A_z`
  detach preserved) across the whole swept range, including large `λ` where
  `Ω` becomes ill-conditioned
- Must report whether a graph prior can *help* at a `β` where it is not
  merely surviving — the phase's most direct test of mission component 5
- If T5.6 lands first, this sweep must be repeated on top of it: a
  variational `A_z` changes `L_z`'s distribution, so results do not transfer

### FR-6: Variational latent graph `A_z` (T5.6)
- Each potential latent edge must be modelable as a Bernoulli random variable
  with a Concrete/Gumbel-Softmax relaxation, a sparsity prior, and its own KL
  term contributing to the ELBO
- Current state: `A_z` is a deterministic top-k over a parameter-free dot
  product (`graph_method="attention"` has **zero learnable parameters**), and
  at `M=4, k=3` it is the complete graph on 100% of jets — there is no
  topology being learned
- Sampled `A_z` must stay symmetric with a zero diagonal, matching the
  existing contract
- Temperature → 0 must approach hard binary edges; the edge-KL must be
  non-negative; gradients must reach the edge-logit parameters
- The existing deterministic path must remain selectable — Phases 1–4's
  results depend on it
- This is the mission's defining claim and is currently unimplemented; it also
  restores the justification for the QGNN's topology-equivariant ansatz
  (`specs/phase4/plan.md` Design Decision 2), which is vacuous while every
  jet's circuit is structurally identical

### FR-7: Final QGNN run and `README.md` (T5.7)
- One QGNN run, in Phase 4's frozen configuration (`plan.md` Design
  Decision 3), on whichever GVLS configuration wins T5.1–T5.6
- `README.md` gains the results section outstanding since Phase 4
  (`specs/phase4/validation.md` exit criteria), covering both phases'
  numbers, the `N`-only bar, and the Lorentz-EQGNN comparison with its
  bibliographic status still flagged per Phase 4's NFR-5

---

## Non-Functional Requirements

### NFR-1: Acceptance bar
- **The bar is `0.7552`** — test accuracy from a logistic regression on
  particle count `N` alone (1 feature, AUC 0.838, test base rate 0.5144) —
  not Phase 4's `0.6688` QGNN number and not GVLS's current `0.7034`
- Every task must report against **both** the `N`-only bar and the current
  GVLS baseline. A result that improves on 0.7034 but not 0.7552 must be
  stated as such explicitly (see NFR-5)
- Rationale: reporting only deltas against the project's own history is how
  Phase 4 arrived at a pipeline that appeared to be improving while sitting
  below a one-feature classifier

### NFR-2: Evaluation protocol held fixed
- Downstream accuracy is measured with Phase 4's exact protocol: frozen GVLS
  features, `gvls.eval.classical_baseline`'s logistic regression and shallow
  MLP, 5 balanced-800 training subsets drawn from the fixed 10,000-jet
  Lorentz pool, scored on the same fixed 1,250-jet test set
- The classical baseline — not the QGNN — is the working metric for the whole
  phase (`plan.md` Design Decision 4), both because Phase 4 established the
  QGNN sits below it on identical features and because it costs seconds
  rather than ~849s/trial
- Reconstruction F1 and bits-per-edge remain reported as diagnostics so
  compression claims stay honest, but are not the selection criterion
  (`plan.md` Design Decision 2)
- The QGNN stage is frozen for the phase; the revert recommended in
  `specs/phase4/validation.md` V-11 was declined by the user (2026-08-04) and
  that row should be read as won't-do, not pending

### NFR-3: Performance — thread pinning and parallelism
- Any new sweep script must reuse `experiments/gvls_prior_sweep.py`'s design:
  single-threaded workers (BLAS env vars set *before* torch is imported) plus
  parallel processes across configs
- Jets are ~43 particles, so every tensor op is a ~43×43 matrix and BLAS
  thread dispatch costs more than the arithmetic: measured 0.71 ms/jet-step
  at 1 thread vs 0.82 at 8, with the penalty growing with core count
- **Raising thread counts is the bug, not the cure** — it cost ~9× on a
  remote machine once already (`specs/phase4/validation.md`, the sweep's own
  history). Add workers instead
- Results must be flushed per config so a killed run leaves a readable
  partial CSV

### NFR-4: Backward compatibility and reproducibility
- Every model/objective change in T5.1–T5.6 must be gated so the Phase 1–4
  behavior remains selectable and bit-identical by default, with the single
  exception of FR-1's normalization, which is a defect fix that necessarily
  changes the loss (and whose effect on Phases 0–3 must therefore be checked,
  not assumed)
- Existing checkpoints must keep loading; new fields default to prior
  behavior
- Fixed seeds for dataset subsetting, GVLS pretraining, and any classifier;
  same config + seed reproduces the same reported metrics within tolerance

### NFR-5: Honesty about results, including negative ones
- Carry forward Phase 4's convention: record what was measured, including
  interventions that failed, rather than only what worked. Phase 4's V-11
  documents three consecutive failed QGNN interventions and is more useful
  for it
- Specifically required here: if a task improves reconstruction F1 but not
  downstream accuracy, say so; if it beats 0.7034 but not 0.7552, say so; if
  the concatenation control (0.7683) still beats the encoded variant (FR-3),
  say so plainly rather than reporting only the encoded number
- Do not present an assumed detail as confirmed. The Lorentz-EQGNN source
  paper's bibliographic status remains unconfirmed from Phase 4 (NFR-5 there)
  and that flag carries into any `README.md` text written by T5.7

### NFR-6: Test coverage
- Every changed module gets at least one shape/correctness test, per the
  per-task test lists in `plan.md`
- The invariants that matter most and must be tested directly rather than
  inspected: FR-1's `N`-invariance, FR-3's `1/n_m` variance scaling and its
  empty-cluster guard, FR-4's `K=1` → isotropic reduction, FR-5's `λ=0` →
  isotropic reduction, and FR-6's symmetry/zero-diagonal contract
- `ruff check src/` passes with zero warnings on every file touched
- `pytest tests/` passes in full (263/263 as of Phase 4's close)

---

## New Dependencies

- None expected. The mixture prior (FR-4) and Concrete edge relaxation
  (FR-6) are implementable with `torch.distributions` and existing
  primitives; no new library is anticipated. If one becomes necessary, record
  it here with the reason before adding it.
