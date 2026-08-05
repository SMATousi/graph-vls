# Phase 5 — Plan

## Objective

Make GVLS's **variational** machinery do real work. Phases 1–4 built every
component `specs/mission.md` claims, but a direct audit of the running code
(2026-08-04) found three of the six effectively inert in the production jet
configuration, and a 24-config sweep (2026-08-04, `results/compression/
qg_jets_prior_sweep.csv`) confirmed that none of the obvious config-level
fixes raises performance. This phase changes the model, not the config.

The phase's success is judged on **downstream quark/gluon classification
accuracy from frozen GVLS features**, measured with the same protocol as
Phase 4's comparability run, against a bar established below that is
considerably higher than Phase 4's own numbers.

---

## Evidence this phase is built on

All numbers below are measured, not estimated. Sources: the 2026-08-04 audit
(reproduced in `validation.md` V-0), `results/compression/
qg_jets_prior_sweep.csv`, and `specs/phase4/validation.md` V-11.

### The audit: three of six mission components are inert

| Mission component | State in the production jet config |
|---|---|
| 1. Node-level variational posteriors | Present but inert — `β·KL` is **0.25%** of the post-training loss; pooled posterior `σ = 0.163` against a prior of 1.0 |
| 2. Learned node-count compression `M ≪ N` | Genuine (`M/N = 0.112`) |
| 3. Latent graph inference | Degenerate at `M=4` — `k=3` makes `A_z` the **complete graph on every jet** (6/6 edges on all 200 held-out jets), since `LatentGraphLearner` uses `k = min(self.k, N-1)`. `graph_method="attention"` also has **zero learnable parameters** |
| 4. Graph-structured latent message passing | Load-bearing (`d(ELBO)/dA_z` norm 3.26 at `mp_rounds=1`; exactly 0 at `mp_rounds=0`) |
| 5. Graph-aware prior and ELBO | **Not active** — `prior: isotropic` means `kl_graph_mrf` is never called; `A_z` never enters the prior |
| 6. Unpooling decode reusing `S` | Correct |

The loss is dominated by `assignment_link_loss` (**80.6%**), then
reconstruction (18.0%), assignment entropy (1.2%), and the KL last (0.25%).

### The sweep: config-level fixes do not help

24 configs, `k ∈ {1,2,3}` × `β ∈ {0.001, 0.01, 0.1, 1.0}` × `prior ∈
{isotropic, graph_mrf}` at `M=4`, production protocol, 30 epochs.

- Best (`k=2, β=0.001`): **0.7179** logreg accuracy. Production
  (`k=3, β=0.001, isotropic`): **0.7139**. A `+0.4`-point difference, roughly
  one standard deviation.
- `β` is **strictly monotonic downward**: 0.001 → 0.7068, 0.01 → 0.6983,
  0.1 → 0.6620, 1.0 → 0.6343. Raising the variational weight only ever costs
  accuracy.
- `prior` is an exact wash in the usable range: isotropic 0.7026,
  graph_mrf 0.7026.
- **But** at `β=1.0`, graph_mrf beats isotropic by **+10.3 points**
  (0.6858 vs 0.5827) and avoids the collapse to the trivial-classifier floor
  entirely (F1 0.7254 vs 0.6822). The graph prior does real work — only in a
  regime nothing else survives.
- Reconstruction F1 correlates with downstream accuracy at `+0.922` overall
  but only `+0.245` within the usable range (`β ≤ 0.01`) — the high
  correlation is an artifact of the collapsed rows. **F1 is close to
  uninformative as a checkpoint-selection criterion**, which is what
  `train_pooled_gvls_on_jets` currently selects on.

### The bar: the pipeline is beaten by counting particles

| Features | Test accuracy | AUC |
|---|---|---|
| Particle count `N` **alone** (1 feature, logistic regression) | **0.7552** | 0.838 |
| GVLS `(z̃, A_z)` (38 features) | 0.7034 | 0.771 |
| GVLS `(z̃, A_z)` **+ `N`** | **0.7683** | 0.841 |
| QGNN (Phase 4 best, `specs/phase4/validation.md` V-11) | 0.6688 | 0.722 |
| Lorentz-EQGNN (literature target) | 0.7400 | — |

Test-set base rate is 0.5144, so all of these are meaningful. Particle
multiplicity is the classic quark/gluon discriminant (mean `N` = 53.3 vs
33.7 by class, `corr(N, label) = −0.556`), and **fixed-`M` pooling destroys
it by construction**: every jet becomes exactly `M=4` nodes whether it had 10
particles or 139, and `LatentGraphPooling` column-normalizes
(`w = s / col_sum`), explicitly dividing cluster occupancy out. Some
multiplicity leaks back through the degree-normalized GCN
(`max |corr(z̃ dim, N)| = 0.67`) but not enough to recover the signal.

**`0.7552` is this phase's acceptance bar, not `0.6688`.** A GVLS
representation that cannot beat a single scalar is not a useful compression
of a jet, regardless of how it compares to Phase 4's own history.

---

## Design Decisions

1. **This phase changes the model, not the configuration.** Phase 4's
   followup (T4.13, `specs/phase4/plan.md` Design Decision 17) already
   established that three consecutive QGNN-side architecture interventions
   failed to move accuracy, and this phase's own motivating sweep establishes
   the same for GVLS's config-level knobs (`k`, `β`, `prior`). Continuing to
   sweep hyperparameters is not expected to pay; the tasks below are code
   changes to the model and objective.

2. **Downstream accuracy is the primary metric; reconstruction F1 is
   demoted to a reported diagnostic.** Measured correlation between the two
   is `+0.245` within the usable `β` range. Every task below is judged on
   classical-baseline downstream accuracy (`gvls.eval.classical_baseline`,
   5 balanced-800 trials, fixed 1250-jet test set — Phase 4's exact
   protocol), with F1/bits-per-edge still recorded so compression claims stay
   honest. This also means `train_pooled_gvls_on_jets`'s best-val-**F1**
   checkpoint selection (T4.11, Design Decision 14) is selecting on close to
   the wrong signal — addressed in T5.1.

3. **The QGNN stage is held fixed for the whole phase.** Phase 4's current
   QGNN configuration (`num_layers=8`, `readout_mode=learned`) stays as-is —
   the revert recommended in `specs/phase4/validation.md` V-11 was considered
   and **declined by the user (2026-08-04)**; that row should be read as
   won't-do, not as pending. Freezing the quantum stage keeps every accuracy
   change in this phase attributable to GVLS. A final QGNN run happens once,
   at the end (T5.7), on whichever GVLS configuration wins.

4. **The classical baseline is the working metric, not the QGNN.** Phase 4
   established the QGNN sits below a plain logistic regression on identical
   frozen features (0.6688 vs 0.6907, V-11 Step 2). Until that is closed, the
   classical number is the ceiling GVLS work has to move, and it costs
   seconds per config against the QGNN's ~849s/trial. Established in Phase 4;
   restated here because every task depends on it.

5. **Multiplicity must be recovered *through* the variational parameters, not
   concatenated as a scalar.** Appending `N` to the feature vector reaches
   0.7683 today and is a legitimate engineering answer, but it is not a
   result: it concedes that the latent space discards the most important
   property of the input. A posterior whose *variance* encodes how many
   particles were pooled is both the statistically correct summary of a
   mixture (T5.3) and a defensible compression claim. **The concatenation
   variant is still measured, as a control** (T5.3), so the gap between
   "encoded" and "appended" is on the record rather than assumed away.

6. **Free bits and KL warm-up are treated as a prerequisite inside T5.4/T5.5,
   not as their own headline task.** The sweep shows anything above
   `β = 0.01` degrades, which means a mixture prior (T5.4) or a swept `λ`
   (T5.5) literally cannot be evaluated at a meaningful `β` without
   anti-collapse machinery first. It is implemented as part of whichever of
   those tasks lands first, and shared. Listed under deferred/enabling scope
   rather than promoted, per the user's scoping of this phase to the Tier 1
   and Tier 2 ideas.

7. **Every task reports against the `N`-only bar (0.7552) and the current
   GVLS baseline (0.7034), both.** Reporting only the delta against GVLS's own
   history is how Phase 4 arrived at a pipeline that looked like it was
   improving while sitting below a one-feature classifier. Per NFR-5, a task
   that improves on 0.7034 but not 0.7552 is recorded as such explicitly.

8. **Tasks are ordered so that defects are fixed before enhancements are
   evaluated.** T5.1 re-points checkpoint selection off a signal that barely
   correlates with the phase's objective; T5.2 (variational output discarded
   downstream) severs the variational machinery from the metric. Evaluating a
   mixture prior or a variational `A_z` on top of either of those would
   produce results that have to be re-run afterwards.

9. **A measured correction outranks a plausible derivation (added
   2026-08-05).** T5.1 was specced around a class-correlated-`β` confound
   inferred from the 2.35× per-class ratio of mean `N²`. Implementing it
   surfaced that the inference was wrong — `legacy`'s KL term is
   `N`-independent by construction, so the real per-class difference is 2%.
   The task was rewritten mid-implementation rather than shipped on its
   original premise. Two process points carry forward: quantities that look
   like they must propagate (a 2.35× input ratio) need measuring at the
   output before a task is built on them, and this phase's remaining
   evidence should be re-checked the same way before each task starts.

---

## Scope

### In scope

**Tier 1 — defects and the multiplicity path**

- **T5.1** — Checkpoint-selection criterion, and an ELBO normalization
  switch. **Rewritten 2026-08-05 mid-implementation**: the original framing
  (an `N`-dependent `β_eff = β·N²/M` amounting to a label-correlated
  confound) was measured false — see the task section below and
  `validation.md` V-1. What survives is (a) re-pointing checkpoint selection
  away from reconstruction F1, which correlates only `+0.245` with this
  phase's objective (Design Decision 2), and (b) a documented,
  **default-off** `per_jet` normalization under which `β` has its standard
  β-VAE meaning.
- **T5.2** — Surface the variational output to the downstream task.
  `extract_latent_features` returns only `(z̃, A_z)`; `mu` and `log_var` are
  discarded, and at eval `z̃` is the deterministic `mu` path — the classifier
  has never seen a variance. The variational apparatus is currently severed
  from the metric it is supposed to improve.
- **T5.3** — Occupancy-aware pooled posterior. `LatentGraphPooling`'s
  moment matching is applied *after* column normalization, so a cluster
  pooling 2 particles and one pooling 60 yield identically-scaled Gaussians.
  Track un-normalized cluster mass `n_m = Σ_i S[i,m]` and scale the pooled
  precision by it (the mean of `n` samples has variance `σ²/n`), putting
  multiplicity into `log_var_p` as a variational quantity.

**Tier 2 — making the variational term an asset**

- **T5.4** — Learned mixture prior (VampPrior / VaDE-style) replacing
  `N(0,I)`. An isotropic prior pulls every jet toward one blob, which is
  exactly what destroys between-class separation and why every `β` increase
  measured worse. A `K`-component mixture lets the aggregate posterior be
  multimodal.
- **T5.5** — Graph-MRF `λ` sweep, jointly with `β`. `lambda_` is hardcoded
  at 1.0 everywhere in the codebase and has never been swept, yet it is the
  single knob controlling how graph-structured the prior is — and the one
  place the sweep found a graph prior doing real work (+10.3 points at
  `β=1.0`).
- **T5.6** — Variational latent graph `A_z`. Only node features are
  variational today; `A_z` is a deterministic top-k over a parameter-free
  dot product. Model edges as Concrete/Bernoulli random variables with a
  sparsity prior and their own KL term.

**Closing**

- **T5.7** — One QGNN run on the winning GVLS configuration, plus a
  `README.md` results section (Phase 4's own unfinished exit criterion).

### Stretch / explicitly deferred

- **Free bits and KL warm-up** — implemented as a shared prerequisite inside
  T5.4/T5.5 (Design Decision 6), not tracked as an independent task.
- **`aux_link_weight` annealing** — `assignment_link_loss` is 80.6% of the
  loss and was introduced purely as a T3.6 collapse remedy; nothing in
  T5.1–T5.6 can express itself strongly while the ELBO is 0.3% of the
  gradient. Worth doing, but the user scoped this phase to Tier 1 + Tier 2;
  revisit if T5.1–T5.6 underdeliver.
- **`pos_weight`'s `N`-dependence (added 2026-08-05, surfaced by T5.1)** — the
  actual measured `N`- and class-dependence in the objective sits in the
  reconstruction term, not the KL: `pos_weight = (N²−E)/E` interacting with
  k-NN edge density gives mean reconstruction 28.2 vs 22.1 across the two
  classes (1.28×), which is what drives the 1.42× asymmetry in the effective
  KL:recon ratio. Changing it is a different edit from T5.1's, with its own
  risk to Phases 0–4's reproduced numbers, so it is recorded here rather than
  folded in silently. Worth doing if T5.2/T5.3 leave the gap open.
- **QGNN-side work of any kind** — frozen for the phase (Design Decision 3).
- **Re-running Phase 2's NAS under the corrected KL convention** — the
  standing open question in `specs/roadmap.md`; T5.1 changes the KL
  normalization again, which strengthens the case for eventually doing it,
  but it is not this phase's job.
- **Electron-Photon dataset** — still deferred from Phase 4.

---

## File Map

```
src/gvls/
  losses/
    elbo.py                     # T5.1 — consistent per-jet normalization;
                                #        T5.4 — mixture-prior KL;
                                #        free bits / warm-up (DD 6)
  models/
    pooling.py                  # T5.3 — occupancy-aware pooled posterior
    latent_graph.py             # T5.6 — variational (Concrete) edge sampling
    prior.py                    # T5.4 (new) — learned mixture prior module
  compression/
    jet_sweep.py                # T5.1 — checkpoint-selection criterion
  qgnn_training.py              # T5.2 — carry mu/log_var into JetFeatures
  eval/
    classical_baseline.py       # T5.2 — feature-vector variants
experiments/
  gvls_variational_sweep.py     # T5.4/T5.5 (new) — parallel, mirrors
                                #   gvls_prior_sweep.py's worker design
configs/
  train/
    jet_pretrain.yaml           # new knobs for every task above
tests/
  test_elbo.py                  # T5.1, T5.4
  test_pooling.py               # T5.3
  test_latent_graph.py          # T5.6
  test_qgnn_training.py         # T5.2
```

**Performance note carried forward from Phase 4:** any new sweep script must
reuse `experiments/gvls_prior_sweep.py`'s threading and parallelism design —
single-threaded workers plus parallel processes. Jets are ~43 particles, so
every tensor op is a ~43×43 matrix and BLAS thread dispatch costs more than
the arithmetic; wall-clock gets *worse* with more threads (0.71 ms/jet-step
at 1 thread vs. 0.82 at 8). Getting this wrong cost ~9× on a remote machine
once already.

---

## Tasks

### T5.1 — Checkpoint-selection criterion, and an ELBO normalization switch

**Files:** `src/gvls/losses/elbo.py`, `src/gvls/compression/jet_sweep.py`,
`configs/train/jet_pretrain{,_final}.yaml`, `experiments/gvls_prior_sweep.py`

**Rewritten 2026-08-05, mid-implementation.** This task was originally
specced around a claim that turned out to be false, and the correction
changed what it should deliver. Both versions are recorded here because the
false version is what the surrounding evidence sections were written against.

**What was claimed, and why it was wrong.** `elbo()` mean-reduces
reconstruction over `N²` node pairs while `kl_isotropic`/`kl_graph_mrf`
divide by node count `M`, so `β_eff = β·N²/M`. Mean `N²` is 3,092 vs 1,315
across the two classes, and the original spec inferred from that ratio that
`legacy` applies **2.35× stronger regularization to one class than the
other** — a label-correlated confound. Measured directly on 600 real
validation jets, that inference does not hold: `legacy`'s KL term is
**N-independent by construction** (it divides by `M`, fixed at 4), so its
per-class means differ by 2% (0.004389 vs 0.004291), not 2.35×, and
`corr(N, KL:recon ratio) = −0.28`. See `validation.md` V-1.

**What is actually true.** (a) `legacy` gives `β` no per-graph-ELBO meaning —
a real interpretability defect, but not a confound. (b) The measured 1.42×
class asymmetry in the effective ratio comes from the **reconstruction**
side — `pos_weight = (N²−E)/E` interacting with k-NN edge density (mean
recon 28.2 vs 22.1 by class) — which no KL normalization addresses. (c)
`per_jet` makes `β_eff = β` for every jet, but as a consequence the *raw*
KL:recon ratio becomes strongly `N`-dependent (corr −0.45): measured ratio
`3.5e-6` at `N < 25` versus `1.0e-7` at `N ≥ 80`. That is correct likelihood
behaviour, but it makes the variational term roughly **35× weaker on large
jets** — the opposite of this phase's goal.

**Revised deliverable.** The load-bearing half of this task is the
checkpoint-selection criterion; the normalization becomes a documented,
default-off option rather than a fix.

- **`selection_metric` on `train_pooled_gvls_on_jets`** — `"reconstruction_f1"`
  (the pre-T5.1 behaviour), `"val_loss"`, or `"probe_accuracy"` (a logistic
  probe fit and scored within the validation split; the test split is never
  touched). Selecting on reconstruction F1 when it correlates `+0.245` with
  the phase's actual objective is straightforwardly wrong, and this stands on
  its own evidence regardless of the normalization question.
  **`probe_accuracy` becomes the production default.**
- **`normalization` on `elbo()`** — `"legacy"` (default, unchanged) or
  `"per_jet"` (a true per-graph ELBO scaled to O(1)). Kept because `β`
  having a standard meaning is worth something and the two modes may behave
  differently once `β` matters at all (T5.4/T5.5), but **not adopted as a
  default**, since (c) above suggests it weakens the variational term exactly
  where this phase wants it stronger.
- **Both default to current behaviour**, and the sweep exposes them as
  independent flags, so the `(k, β)` operating point can be re-established
  **one flag at a time**. Changing both at once would make any difference
  unattributable.

**Moved out of this task.** The `pos_weight` interaction in (b) is the actual
`N`-dependence in the objective and is a different change from the one specced
here; it is recorded under deferred scope rather than silently folded in.

Tests: `per_jet`'s `β_eff` equals `β` for every `N`; `legacy`'s equals
`β·N²/M`, so the two conventions are characterized against each other rather
than the new one asserted in isolation; the two modes agree exactly at `β=0`,
pinning the change to where it is claimed to be; omitting `normalization`
reproduces `legacy` byte-identically; each `selection_metric` runs end-to-end
and logs the value it selected on; `val_loss` selection picks the lowest
rather than the highest (a direction-agnostic loop would silently keep the
worst epoch); NaN guard still fires.

---

### T5.2 — Surface the variational output downstream

**Files:** `src/gvls/qgnn_training.py`, `src/gvls/eval/classical_baseline.py`

`extract_latent_features` unpacks `_mu, _log_var, _z, a_z, z_tilde, _s,
_recon_logits` and keeps only `z_tilde` and `a_z`. `mu`/`log_var` are
dropped, and since extraction runs under `model.eval()`, `z_tilde` is the
deterministic mean path. **No variance ever reaches any classifier**, which
is a large part of why raising `β` only ever cost accuracy: the model pays
the regularization and the metric receives none of the information.

- Extend `JetFeatures` with `log_var` (and `mu` if it differs usefully from
  `z_tilde`), defaulting to the current behavior so existing checkpoints and
  callers keep working.
- Extend `jet_features_to_array` with a selectable feature set, and report
  the ablation: `(z̃, A_z)` — today's baseline — versus `(z̃, A_z, log_var)`.
- The QGNN's own input is **not** changed here (Design Decision 3); this task
  is about whether the information is present and useful at all, measured
  classically.

Tests: `JetFeatures` round-trips the new fields; the default feature set is
byte-identical to today's; feature-vector width matches the selected set.

---

### T5.3 — Occupancy-aware pooled posterior

**File:** `src/gvls/models/pooling.py`

`LatentGraphPooling.forward` computes `w = s / col_sum` and pools with
column-normalized weights, so `mu_pooled`/`var_pooled` are scale-free in
cluster size. The law-of-total-variance step is correct but occupancy has
already been divided out before it runs. Consequently `M=4` pooled Gaussians
look the same whether the jet had 10 or 139 particles — discarding the
dataset's single most discriminative feature (see the bar table above).

- Track un-normalized cluster mass `n_m = Σ_i S[i,m]` and scale the pooled
  precision by it: the mean of `n` samples has variance `σ²/n`, so
  `var_pooled ← var_pooled / n_m` (guarded for empty clusters). Multiplicity
  then lives in `log_var_p`, reaching the downstream task via T5.2.
- Gate behind a constructor flag so the Phase 3/4 behavior stays reproducible
  and the change is measurable as an A/B rather than a silent redefinition.
- **Measure the concatenation control** (Design Decision 5): report
  `(z̃, A_z) + N` (0.7683 today) alongside the occupancy-aware variant, so
  the cost of insisting on an encoded representation is on the record.

Tests: pooled variance scales as `1/n_m` on a synthetic assignment with known
cluster masses; the flag's default reproduces current output bit-identically;
empty/near-empty clusters do not produce NaN or infinite precision;
`PooledGVLS`'s output shapes are unchanged.

---

### T5.4 — Learned mixture prior

**Files:** `src/gvls/models/prior.py` (new), `src/gvls/losses/elbo.py`

Every measured `β` increase reduced downstream accuracy, monotonically. The
mechanism is structural rather than a tuning failure: `N(0,I)` pulls all jets
toward a single mode, and between-class separation is exactly what that
destroys. At `β=1.0` with the isotropic prior the posterior collapses
(`σ = 0.89` against a prior of 1.0) and reconstruction hits the trivial
floor.

- Implement a `K`-component learned mixture prior (VampPrior-style
  pseudo-inputs, or VaDE-style free mixture parameters — pick one, document
  why). The aggregate posterior can then be multimodal, so KL pressure
  encourages cluster structure instead of erasing it.
- KL to a mixture has no closed form; use the standard single-sample Monte
  Carlo estimate via the already-drawn reparameterized sample (the codebase
  already has the fallback precedent in `specs/tech_stack.md`).
- **Prerequisite (Design Decision 6):** free bits (`max(KL_d, λ_free)` per
  dimension) and a `β` warm-up schedule, so this can be evaluated at a `β`
  where the prior actually matters. Shared with T5.5.
- Sweep `K` (e.g. `{2, 4, 8}`) against `β`, reusing the parallel sweep
  design.

Tests: mixture KL is non-negative and → 0 as the posterior approaches one
component; `K=1` reduces to the isotropic case within tolerance; free bits
clamps at the configured floor; gradients reach the prior's own parameters.

---

### T5.5 — Graph-MRF `λ` sweep

**Files:** `configs/train/jet_pretrain.yaml`,
`experiments/gvls_variational_sweep.py`

`kl_graph_mrf`'s `lambda_` sets the prior precision `Ω = I + λ·L_z` and has
been fixed at 1.0 in every run this project has ever done. It is the only
knob controlling how much of the prior is graph-structured versus plain
identity shrinkage — and the sweep found the one piece of positive evidence
for a graph prior anywhere in this project: at `β=1.0`, graph_mrf scored
`+10.3` points over isotropic and avoided collapse entirely. The mechanism is
that `Ω` penalizes *disagreement between connected latent nodes* rather than
magnitude, so it is a smoothness prior, not a shrinkage prior, and does not
erase information the same way.

- Sweep `λ` (e.g. `{0.1, 1, 10, 100}`) × `β`, with free bits/warm-up from
  T5.4 available, at the `(k, M)` operating point re-established by T5.1.
- Report whether a graph-structured prior can be made to *help* at a `β`
  where it is not merely surviving — this is the phase's most direct test of
  mission component 5.
- Note the interaction with T5.6: a variational `A_z` changes `L_z`'s
  distribution, so if T5.6 lands first this sweep should be repeated on top
  of it rather than assumed to transfer.

Tests: `λ=0` reduces `kl_graph_mrf` to the isotropic case within tolerance;
the log-det term stays finite and first-order (the existing detach is
preserved) across the swept `λ` range, including large `λ` where `Ω` becomes
ill-conditioned.

---

### T5.6 — Variational latent graph `A_z`

**File:** `src/gvls/models/latent_graph.py`

The project is named for a graph *variational* latent space, but only node
features are variational: `A_z` is a deterministic top-k over a
parameter-free dot product (`attention` has zero learnable parameters).
At `M=4, k=3` it is additionally the complete graph on every jet, so there
is no topology being learned at all.

- Model each potential edge as a Bernoulli random variable with a
  Concrete/Gumbel-Softmax relaxation for differentiability, plus a sparsity
  prior and its own KL term contributing to the ELBO.
- This subsumes the top-k sparsification with a learned, probabilistic one,
  and gives `A_z` gradient signal that does not have to travel through
  message passing.
- Payoff beyond accuracy: it makes the QGNN's entangling topology genuinely
  stochastic and jet-specific, which is the entire justification for choosing
  the Verdon topology-equivariant ansatz over a generic circuit
  (`specs/phase4/plan.md` Design Decision 2) — a justification that is
  currently vacuous at `M=4`, where every jet's circuit is structurally
  identical.
- Keep the existing deterministic path selectable; the Phase 1–4 results
  depend on it.

Tests: sampled `A_z` stays symmetric with a zero diagonal; temperature → 0
approaches hard binary edges; gradients reach the edge-logit parameters;
the edge-KL is non-negative; the deterministic path is unchanged.

---

### T5.7 — Final QGNN run and `README.md`

**Files:** `scripts/run_qgnn_lorentz_comparability.sh`, `README.md`

- One QGNN run (Phase 4's frozen configuration, Design Decision 3) on
  whichever GVLS configuration wins T5.1–T5.6, so the phase ends with a
  quantum number and not just a classical one.
- Write the results section `specs/phase4/validation.md` still lists as
  outstanding, covering both phases' numbers, the `N`-only bar, and the
  Lorentz-EQGNN comparison — with its bibliographic status still flagged per
  Phase 4's NFR-5.

---

## Deliverable

- `src/gvls/losses/elbo.py` with an `N`-invariant per-jet normalization, and
  a documented answer to whether the sweep's `β` conclusions survived it
- `JetFeatures`/`jet_features_to_array` carrying the variational output, with
  a measured ablation of what it is worth
- An occupancy-aware `LatentGraphPooling`, measured against both today's
  behavior (0.7034) and the concatenation control (0.7683)
- A learned mixture prior and a swept graph-MRF `λ`, each reported at a `β`
  where the prior can actually express itself
- A variational `A_z` with learned, jet-specific topology
- `results/compression/qg_jets_variational_sweep.csv` and a populated
  `specs/phase5/validation.md`, including the negative results
- One QGNN run on the winning configuration, and the `README.md` results
  section outstanding since Phase 4
