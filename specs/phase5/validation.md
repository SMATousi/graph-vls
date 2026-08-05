# Phase 5 — Validation

**Status (2026-08-05): T5.1–T5.3 complete; T5.4–T5.8 open.** V-0 records the
measured evidence this phase was created from — the 2026-08-04 mission-
conformance audit, the 24-config `(k, β, prior)` sweep, and the multiplicity
baseline that reset the phase's acceptance bar.

Progress on the acceptance bar (NFR-1), classical logistic regression on
frozen features, best **encoded** representation (no raw `N` appended):

| | accuracy | vs. `N`-only bar (0.7552) |
|---|---|---|
| Pre-Phase-5 (`z̃, A_z`) | 0.7034 | −0.052 |
| after T5.2 (`+ log_var`) | 0.7379 | −0.017 |
| after T5.3 (`+ occupancy-aware variance`) | **0.7555 ± 0.0138** | **tie** |

The bar is reached but not beaten, and the concatenation control
(`(z̃, A_z) + N` = 0.7685) still leads the encoded representation by 0.0130 —
so Design Decision 5's "encoding beats appending" claim is narrowed, not
established. T5.1's premise was measured false mid-implementation and the task
was rewritten (V-1); that correction is recorded rather than quietly dropped,
per NFR-5.

## Exit Criteria

- [x] (T5.1, FR-1) Checkpoint-selection criterion selectable, defaulting to
      `probe_accuracy` in production; `normalization` switch implemented and
      documented, `legacy` retained as default on measured grounds — see V-1
      for the correction to this task's original premise
- [ ] (T5.1, FR-1) `(k, β)` operating point re-established under whichever
      flags are adopted — not yet run; both default to prior behaviour, so a
      bare re-run reproduces the original grid
- [x] (T5.2, FR-2) `log_var` reaches the downstream classifier; ablation
      measured and reported — **+0.0296 accuracy** (0.7074 → 0.7370), and
      `logvar_only` alone beats the entire pre-T5.2 feature set. See V-2
- [x] (T5.3, FR-3) Occupancy-aware pooled posterior implemented and measured
      against both today's behaviour and the concatenation control — every
      variance-bearing feature set improves (+0.011 to +0.018) and the encoded
      representation reaches 0.7555, **tying** the `N`-only bar; the
      concatenation control still leads by 0.0130. See V-3
- [ ] (T5.4, FR-4) Learned mixture prior implemented, with free bits/warm-up,
      evaluated at a `β` where the prior can express itself
- [ ] (T5.5, FR-5) Graph-MRF `λ` swept jointly with `β`; a documented answer
      to whether a graph prior can help rather than merely survive
- [ ] (T5.6, FR-6) Variational `A_z` with learned, jet-specific topology;
      deterministic path still selectable
- [ ] (T5.8, FR-8) `log_var` encoded into the circuit on a second (`RX`)
      rotation axis, with the angle-scaling risk measured before the first run
      and the pub-count cost measured rather than argued — **queued, not
      implemented**; runs after the in-flight T5.1 comparability sweep
- [ ] (T5.7, FR-7) One QGNN run on the winning GVLS configuration
- [ ] (T5.7, FR-7) `README.md` results section written — outstanding since
      Phase 4
- [ ] (NFR-1) Every task reported against **both** the `N`-only bar (0.7552)
      and the GVLS baseline (0.7034)
- [ ] (NFR-6) `pytest tests/` passes in full; `ruff check src/` clean on
      every touched file

---

## V-0: Evidence this phase was created from ✅ Measured 2026-08-04

Three independent findings, all measured on real cached `qg_jets` data under
the production Lorentz protocol (10,000/1,250/1,250, `min_particles=10`).

### Part A — Mission-conformance audit of `src/gvls/models/`

Audit of the running code against `specs/mission.md`'s six claimed
components, using the production jet config
(`configs/train/jet_pretrain.yaml`: `M=4, d=8, k=3, attention, isotropic,
mp_rounds=1, β=0.001`). Post-training measurements taken on 200 held-out real
jets after 20 epochs on 1,050 training jets.

| Component | Verdict | Evidence |
|---|---|---|
| 1. Node-level variational posteriors | ⚠️ Present, inert | `β·KL` = **0.25%** of post-training loss; raw KL 12.3 nats/node; pooled `σ = 0.163` vs prior 1.0 (sharp, not collapsed, but near-deterministic) |
| 2. Learned node-count compression | ✅ Genuine | `M/N = 0.112` (`M=4`, mean `N` 43.5) |
| 3. Latent graph inference | ⚠️ Degenerate | `A_z` complete (6/6 undirected edges) on **all 200** jets; structural, since `LatentGraphLearner` uses `k = min(self.k, N-1)` and `k=3` at `M=4` selects every off-diagonal entry. Edge weights 0.325–0.778 — never near zero, so no implicit sparsity either. Already visible in `results/compression/qg_jets_pooling.csv` as `avg_num_latent_edges = 6.0` with zero variance across 1,500 jets. `graph_method="attention"` has **0 learnable parameters** (`fgp` has 1, `nri` 145) |
| 4. Graph-structured message passing | ✅ Load-bearing | `d(ELBO)/dA_z` norm 3.26 at `mp_rounds=1`; exactly 0.000 at `mp_rounds=0` (the CiteSeer inert-`A_z` failure mode is avoided, but via this one path only) |
| 5. Graph-aware prior and ELBO | ❌ Not active | `prior: isotropic` ⇒ `kl_graph_mrf` never called. The path works when selected (`d(kl_graph_mrf)/dA_z` norm 3.88) — it simply is not |
| 6. Unpooling decode reusing `S` | ✅ Correct | — |

Post-training loss decomposition (200 held-out jets): **`assignment_link_loss`
80.6%**, reconstruction 18.0%, assignment entropy 1.2%, `β·KL` **0.25%**.

Also noted: the encoder-level `(mu, log_var)` over all `N` nodes is **never**
KL-penalized — only the pooled `(M, d)` Gaussian is — and `S` is computed
from the *sampled* `z`, so there is a second, unregularized noise injection
with no probabilistic counterpart.

### Part B — 24-config `(k, β, prior)` sweep

`results/compression/qg_jets_prior_sweep.csv`, `experiments/
gvls_prior_sweep.py`. `k ∈ {1,2,3}` × `β ∈ {0.001, 0.01, 0.1, 1.0}` ×
`prior ∈ {isotropic, graph_mrf}` at `M=4`, 30 epochs, full 10,000-jet
training pool, mean 1,720s/config.

Marginal mean logistic-regression accuracy, restricted to the usable range
(`β ≤ 0.01`):

| Axis | Values |
|---|---|
| `k` | 1 = 0.6950, **2 = 0.7110**, 3 = 0.7017 |
| `β` | **0.001 = 0.7068**, 0.01 = 0.6983 (full range: 0.1 = 0.6620, 1.0 = 0.6343) |
| `prior` | isotropic = 0.7026, graph_mrf = 0.7026 |

| Finding | Detail |
|---|---|
| No config-level fix helps | Best (`k=2, β=0.001`) 0.7179 vs production (`k=3, β=0.001, isotropic`) 0.7139 — `+0.4` points, ≈1 std |
| Raising `β` monotonically hurts | 0.7068 → 0.6983 → 0.6620 → 0.6343. At `β=1.0` isotropic the posterior collapses (`σ = 0.89`), reconstruction hits the trivial floor (F1 0.6822), downstream falls to 0.58 |
| The graph prior is real, but only where everything else fails | At `β=1.0`: graph_mrf 0.6858 (F1 0.7254) vs isotropic 0.5827 (F1 0.6822) — **+10.3 points**, and no collapse. `Ω = I + λL_z` penalizes disagreement between connected nodes rather than magnitude, so it is a smoothness prior, not a shrinkage prior |
| Reconstruction F1 is a poor selection criterion | `corr(F1, downstream)` = `+0.922` across all 24 rows but only **`+0.245`** within `β ≤ 0.01` — the high figure is an artifact of the collapsed rows. Best-F1 config (0.7567) has downstream 0.7107, below the best-downstream config. `train_pooled_gvls_on_jets` currently selects on exactly this |
| Cross-run observation | Production pretrains 200 epochs; the identical config at 30 epochs scored 2.3 points *higher* downstream (0.7139 vs V-11's 0.6907). Not a controlled A/B — flagged, not acted on |

### Part C — The multiplicity baseline (resets the acceptance bar)

| Features | Test accuracy | AUC |
|---|---|---|
| Particle count `N` **alone** (1 feature, logreg) | **0.7552** | 0.838 |
| GVLS `(z̃, A_z)` (38 features) | 0.7034 ± 0.0048 | 0.771 |
| GVLS `(z̃, A_z)` **+ `N`** | **0.7683 ± 0.0030** | 0.841 |
| QGNN (`specs/phase4/validation.md` V-11 best) | 0.6688 | 0.722 |
| Lorentz-EQGNN (literature target) | 0.7400 | — |

Test-set base rate 0.5144, so every row is meaningful. Class statistics:
mean `N` = 53.3 vs 33.7, `corr(N, label) = −0.556`, `N` range 10–139
(`N²` range 100–19,321, **193×**), mean `N²` 3,092 vs 1,315 by class
(**2.35×** — but see V-1: this ratio does **not** propagate to the KL term,
contrary to what the first version of this section claimed).

**Interpretation.** Particle multiplicity is the classic quark/gluon
discriminant, and fixed-`M` pooling destroys it by construction — every jet
becomes `M=4` nodes regardless of size, and `LatentGraphPooling`
column-normalizes (`w = s / col_sum`), explicitly dividing occupancy out.
Some leaks back through the degree-normalized GCN
(`max |corr(z̃ dim, N)| = 0.67`, mean 0.385) but not enough. **The full
GVLS→QGNN pipeline currently performs worse than counting particles.**

An `N`-dependence also appears inside the objective: reconstruction is
mean-reduced over `N²` pairs while the KL is divided by `M`, so
`β_eff = β · N²/M ≈ 551β` at the mean — which explains Part B's collapse at
`β=1.0` (`β_eff ≈ 551`) and why `β=0.001` (`β_eff ≈ 0.55`) sits near a proper
ELBO.

**Corrected 2026-08-05 (see V-1).** An earlier version of this paragraph went
further and claimed that, since `N` is class-correlated, the regularization
*strength* is therefore a function of the label (2.35×). That inference was
measured false while implementing T5.1: `legacy`'s KL term is `N`-independent
by construction, so the real per-class difference is 2%. The measured class
asymmetry that does exist comes from the reconstruction side, not the KL.

---

## V-1: Checkpoint-selection criterion and ELBO normalization switch (T5.1) ✅ Implemented 2026-08-05 — original premise measured false, task rewritten

**Files:** `src/gvls/losses/elbo.py` (`normalization` argument),
`src/gvls/compression/jet_sweep.py` (`SELECTION_METRICS`,
`selection_metric`, `probe_accuracy`, `validation_loss`,
`jet_probe_features`), `configs/train/jet_pretrain{,_final}.yaml`,
`experiments/gvls_prior_sweep.py` (`--normalization`,
`--selection-metric`), `experiments/pretrain_gvls_jets_final.py`.
Tests: 9 new (7 in `tests/test_elbo.py`, 7 in `tests/test_jet_sweep.py`);
suite 270 → 279.

### The premise this task was specced on, and its correction

T5.1 was written around the claim that `legacy` normalization
(`β_eff = β·N²/M`) makes KL regularization **2.35× stronger on one class than
the other**, inferred from the per-class ratio of mean `N²` (3,092 vs 1,315)
combined with `corr(N, label) = −0.556`. **Measured on 600 real validation
jets, that inference is false.**

| Quantity | quark | gluon | ratio |
|---|---|---|---|
| `legacy` KL term | 0.004389 | 0.004291 | **1.02×** |
| reconstruction term | 28.24 | 22.08 | 1.28× |
| effective KL:recon ratio | 0.000156 | 0.000222 | 1.42× |

`legacy` divides the KL by `M`, which is fixed at 4, so its KL term is
`N`-independent by construction and cannot carry an `N`-derived class
asymmetry: `corr(N, legacy KL:recon ratio) = −0.28`, and what asymmetry
exists traces to the **reconstruction** side (`pos_weight = (N²−E)/E` ×
k-NN edge density), which no KL normalization touches.

A second measured finding argues against adopting `per_jet` at all:

| Mode | corr(`N`, raw KL:recon ratio) | ratio at `N < 25` | ratio at `N ≥ 80` |
|---|---|---|---|
| `legacy` | −0.28 | 3.5e-6 | 1.7e-6 |
| `per_jet` | **−0.45** | 3.5e-6 | **1.0e-7** |

`per_jet` gives `β` its standard β-VAE meaning, but as a direct consequence
the variational term becomes ~35× weaker on the largest jets (correct
likelihood behaviour — more observed pairs outweigh the prior). For a phase
whose goal is making the variational term matter, that is the wrong
direction, so `per_jet` ships as a documented option and **not** as a
default.

### What was delivered

| Check | Pass condition | Result |
|---|---|---|
| `selection_metric` implemented | `reconstruction_f1` / `val_loss` / `probe_accuracy` selectable on `train_pooled_gvls_on_jets` | ✅ `probe_accuracy` is the production default (`jet_pretrain_final.yaml`); fits a logistic probe within the validation split, test split never touched |
| Selection direction handled | `val_loss` picks the lowest, not the highest | ✅ `test_val_loss_selection_picks_the_lowest_not_the_highest` — mocks a non-monotonic sequence (5.0, 1.0, 9.0) and asserts the epoch-1 weights come back |
| Criterion recorded in the run's own logs | The selected value appears in per-epoch metrics | ✅ `val_loss` / `val_probe_accuracy` keys; parametrized test over all three |
| `normalization` implemented | `legacy` / `per_jet` selectable on `elbo()` | ✅ default `legacy` |
| `per_jet` is a true per-graph ELBO | `β_eff == β` for every `N` | ✅ `test_per_jet_effective_beta_is_invariant_to_graph_size` (`N ∈ {10, 40, 139}`) |
| `legacy` characterized, not just contrasted | `β_eff == β·N²/M` | ✅ `test_legacy_effective_beta_scales_with_n_squared_over_m` |
| Change is confined to the KL term | The two modes agree exactly at `β=0` | ✅ `test_normalizations_agree_when_kl_weight_is_zero` |
| Backward compatibility (NFR-4) | Omitting `normalization` reproduces `legacy` byte-identically; `selection_metric` defaults to the pre-T5.1 criterion | ✅ two tests; production configs changed only where intended |
| Attribution preserved | Sweep exposes both as independent flags, both defaulting to pre-Phase-5 behaviour | ✅ `--normalization` / `--selection-metric`; changing one at a time is documented as required |
| Full suite / lint | `pytest tests/`, `ruff check src/` | ✅ 279/279; `src/` clean. Two pre-existing `E501`s in `tests/test_jet_sweep.py` (lines 74–75) confirmed present on `HEAD` before this work and left alone, per Phase 4 V-7's precedent |

### Notes and follow-ups

- **Test-construction bug found and fixed during this work:** the first
  version of the `β_eff` tests derived the KL contribution by subtracting two
  nearly-equal `elbo()` calls in float32. Under `per_jet` at `N=139` that
  difference is ~1e-5 against a ~0.69 reconstruction term, and the
  cancellation destroyed nearly every significant digit — the tests were
  failing on their own arithmetic. They now run in float64.
- **Wrong config edited first:** `configs/train/jet_pretrain.yaml` is T4.3's
  M-grid sweep config; the production path
  (`pretrain_gvls_jets_final.py` → `run_qgnn_lorentz_comparability.sh`) uses
  `jet_pretrain_final.yaml`. Both now carry the knobs.
- **`probe_accuracy` deliberately does not import
  `extract_latent_features`/`jet_features_to_array`** — both live in modules
  that import `gvls.models.qgnn`, hence qiskit, and pulling the quantum stack
  into GVLS's classical pretraining path would make pretraining fail on a
  machine without qiskit. `jet_probe_features` is a small local
  reimplementation with a matching feature layout, tested for width.
- **Not yet run:** the `(k, β)` re-establishment sweep. With both flags
  defaulting to prior behaviour, a bare re-run reproduces the original grid;
  the intended comparison is one flag at a time.
- **Deferred out of this task:** `pos_weight`'s `N`-dependence, which is where
  the measured class asymmetry actually lives. See `plan.md`'s deferred scope.

## V-2: Variational output reaches the downstream task (T5.2) ✅ Complete 2026-08-05 — +3.0 accuracy points from a tensor that was being discarded

**Files:** `src/gvls/qgnn_training.py` (`JetFeatures.mu`/`.log_var`,
`extract_latent_features`), `src/gvls/eval/classical_baseline.py`
(`FEATURE_SETS`, `jet_features_to_array(..., feature_set)`,
`evaluate_classical_baselines(..., feature_set)`),
`experiments/variational_feature_ablation.py`,
`configs/variational_feature_ablation_config.yaml`. Tests: 11 new in
`tests/test_classical_baseline.py`; suite 279 → 290.

**The defect.** `extract_latent_features` kept only `(z_tilde, A_z)` and
dropped the pooled posterior's `mu`/`log_var`. Because extraction runs under
`model.eval()`, `z_tilde` is the deterministic mean path — so **no classifier
in this project had ever seen a variance.** GVLS paid for its variational
term in the objective and the metric received none of the information, which
is a large part of why every `β` increase measured worse in V-0 Part B.

### Ablation result

Frozen GVLS checkpoint (`M=4, d=8, k=2, β=0.001, isotropic`, 30 epochs,
`selection_metric=probe_accuracy` per T5.1), Lorentz protocol, 5 balanced-800
training subsets, fixed 1,250-jet test set — the same protocol as
`classical_baseline_diagnostic.py`, so these are directly comparable.

| Feature set | logreg accuracy | AUC | vs. `z_a` |
|---|---|---|---|
| `z_a` (pre-T5.2 baseline) | 0.7074 ± 0.0060 | 0.7752 | — |
| `z_a_logvar` | **0.7370 ± 0.0065** | 0.8003 | **+0.0296** |
| `z_a_mu_logvar` | 0.7379 ± 0.0055 | 0.8040 | +0.0306 |
| `logvar_only` | 0.7237 ± 0.0119 | 0.7886 | +0.0163 |

Three findings, in order of how much they matter:

1. **Adding `log_var` is worth ~3.0 accuracy points and ~2.5 AUC points**, at
   the cost of carrying a tensor the forward pass already computed. This is
   the largest single improvement any intervention in Phases 4–5 has produced.
2. **`logvar_only` (32 numbers) beats the entire pre-T5.2 feature set**
   (0.7237 vs 0.7074, 38 numbers). The tensor being discarded was more
   discriminative than everything that was kept.
3. **`mu` adds nothing beyond `log_var`** (+0.0009, well inside one std),
   confirming it is redundant with `z_tilde` — which is `mu` pushed through
   latent message passing at eval time — and that message passing is not
   discarding anything a linear model can use.

**Why the variance carries signal — measured, not assumed.** Predicting
particle count `N` from the features by linear regression: `logvar_only`
reaches `R² = 0.718`, slightly *above* `z_a`'s `0.708`. The pooled posterior's
spread partly encodes cluster occupancy already, via the law-of-total-variance
term in `LatentGraphPooling`, even though the column normalization divides
occupancy out of the *mean*. This is direct evidence for T5.3's premise, and
it also caps expectations: `R² ≈ 0.72` is a partial recovery, which is
consistent with the ablation improving on the GVLS baseline (0.7034) while
still falling short of the `N`-only bar.

| Check | Pass condition | Result |
|---|---|---|
| `mu`/`log_var` carried | `extract_latent_features` populates them | ✅ `test_extract_latent_features_now_carries_mu_and_log_var` |
| Default unchanged (NFR-4) | `z_a` byte-identical to pre-T5.2, ignoring the new fields entirely | ✅ `test_default_feature_set_is_byte_identical_to_pre_t52`; the `z_a` row (0.7074) also reproduces V-0 Part C's 0.7034 within run-to-run variation |
| Ablation measured and reported (FR-2) | `(z̃, A_z)` vs `(z̃, A_z, log_var)` | ✅ table above |
| Feature widths correct | Parametrized over all four sets | ✅ |
| Extra columns are real | The `log_var` columns contain `log_var`, not zeros or duplicates | ✅ `test_log_var_columns_actually_carry_log_var` — width alone would pass a broken implementation |
| Old features fail loudly | Requesting a variance-bearing set on pre-T5.2 features raises rather than silently narrowing | ✅ `test_variance_bearing_sets_reject_pre_t52_features` |
| Mechanism check | If the class signal lives *only* in the variance, `z_a` misses it and `logvar_only` finds it | ✅ `test_log_var_only_separation_is_detected` |
| QGNN input unchanged | The circuit still receives `(z̃, A_z)` only | ✅ no change to `QGNNClassifier`/`collate_jet_features` |
| Bars reported (NFR-1) | Against both the GVLS baseline and the `N`-only bar | ✅ best 0.7379 clears the 0.7034 GVLS baseline; **does not clear the 0.7552 `N`-only bar** |
| Full suite / lint | `pytest tests/`, `ruff check src/` | ✅ 290/290; `src/` clean |

### Side observation: T5.1's premise corroborated

The checkpoint used here was selected by T5.1's new `probe_accuracy`
criterion, which picked **epoch 18** at probe accuracy 0.7552 while
`val_reconstruction_f1` sat flat at 0.740–0.748 for all 30 epochs. Selecting
on F1 would have been close to picking an epoch at random. That is
independent support for T5.1's revised deliverable (`plan.md` T5.1), measured
on a different run from the correlation that motivated it.

### Follow-ups

- **The `N`-only bar (0.7552) is still not cleared** by the best feature set
  (0.7379). The variance recovers part of what fixed-`M` pooling discards but
  not all of it — `R² = 0.718` for `N` says why. Closing that is T5.3.
- **The QGNN does not yet see `log_var`.** FR-2 deliberately scoped this to
  the classical measurement (Design Decision 3 freezes the quantum stage), but
  a ~3-point gain in what the features support is only realized downstream if
  T5.7's final run feeds them in. Encoding `d` more values per qubit interacts
  with the re-uploading bottleneck Phase 4 found (`specs/phase4/validation.md`
  V-11 Step 2), so it needs its own decision rather than being assumed free.
- The checkpoint used is a locally-trained 30-epoch one, not the production
  100-epoch `gvls_jets_m4_lorentz800.pt`; the comparison is internally
  consistent (all four feature sets share one frozen checkpoint) but absolute
  numbers will shift on a production rerun.

## V-3: Occupancy-aware pooled posterior (T5.3) ✅ Complete 2026-08-05 — encoded representation reaches the `N`-only bar, but only ties it

**Files:** `src/gvls/models/pooling.py` (`LatentGraphPooling.occupancy_aware`,
`MIN_OCCUPANCY`), `src/gvls/compression/jet_sweep.py` (`build_pooled_gvls`),
`src/gvls/qgnn_training.py` (`JetFeatures.num_nodes`),
`src/gvls/eval/classical_baseline.py` (`z_a_n` / `z_a_logvar_n` / `n_only`
control feature sets), `configs/train/jet_pretrain{,_final}.yaml`,
`configs/variational_feature_ablation_config.yaml`. Tests: 14 new (7 in
`tests/test_pooling.py`, 5 in `tests/test_classical_baseline.py`, plus
parametrized cases); suite 290 → 304.

### Premise re-verified before implementing (Design Decision 9)

| Claim | Check | Result |
|---|---|---|
| Column normalization divides occupancy out | Tile one node set 1×/2×/5× — same cluster composition, k× the mass | `var_pooled` **bit-identical** across all three |
| Multiplicity is recoverable if variance carries mass | Does `Σ_m n_m == N`? | ✅ exactly, at `N ∈ {10, 43, 139}` — every row of `S` sums to 1 |

The second identity is what makes this work: dividing the pooled variance by
`n_m = Σ_i S[i,m]` puts `−log(n_m)` offsets into the `M` log-variances whose
masses total `N` precisely.

### Result — A/B on two checkpoints identical except for the flag

Same config (`M=4, d=8, k=2, β=0.001, isotropic`, 30 epochs,
`selection_metric=probe_accuracy`), same Lorentz protocol, same 5
balanced-800 subsets, same fixed test set.

| Feature set | `occupancy_aware=False` | `occupancy_aware=True` | Δ |
|---|---|---|---|
| `z_a` | 0.7074 ± 0.0060 | 0.7075 ± 0.0103 | +0.0001 |
| `z_a_logvar` | 0.7370 ± 0.0065 | 0.7482 ± 0.0089 | **+0.0112** |
| `z_a_mu_logvar` | 0.7379 ± 0.0055 | **0.7555 ± 0.0138** | **+0.0176** |
| `logvar_only` | 0.7237 ± 0.0119 | 0.7357 ± 0.0095 | **+0.0120** |
| *control:* `n_only` | 0.7555 ± 0.0045 | 0.7555 ± 0.0045 | — |
| *control:* `z_a_n` | 0.7643 ± 0.0059 | 0.7685 ± 0.0058 | +0.0042 |
| *control:* `z_a_logvar_n` | 0.7677 ± 0.0032 | 0.7670 ± 0.0069 | −0.0007 |

1. **The intervention does what it was designed to do.** Every
   variance-bearing feature set improves (+0.0112, +0.0176, +0.0120), and
   `z_a` — which reads only the pooled *mean*, untouched by the flag — moves
   by +0.0001. The change is isolated to where it was claimed to be.
2. **The encoded representation reaches the `N`-only bar for the first time**:
   `z_a_mu_logvar` = 0.7555 ± 0.0138 against the bar's 0.7552. **This is a
   tie, not a win** — the margin is 0.0003 against a std of 0.0138, and the
   occupancy-aware runs are noticeably noisier than their counterparts.
3. **The concatenation control still wins.** Best control 0.7685 vs. best
   encoded 0.7555 — a `0.0130` gap. T5.3 narrowed it (it was `0.0298` before)
   but did not close it. Design Decision 5's claim, that encoding multiplicity
   is worth more than appending it, is **not yet established**; appending is
   still better.

| Check | Pass condition | Result |
|---|---|---|
| Variance scales as `1/n_m` | Tiling k× drops `log_var` by exactly `log k` | ✅ `test_pooled_variance_scales_as_one_over_cluster_mass`, `k ∈ {2,5}` |
| Contrast pinned | With the flag off, the same tiling changes nothing | ✅ `test_occupancy_off_leaves_variance_scale_free` |
| Default bit-identical (NFR-4) | Flag defaults off; output byte-for-byte unchanged | ✅ `test_occupancy_aware_defaults_to_off_and_is_bit_identical` |
| Only the variance moves | `mu_pooled` identical with the flag on and off | ✅ `test_mu_pooled_unaffected_by_occupancy_flag`; corroborated by the `z_a` row's +0.0001 |
| Empty clusters safe | No inf/NaN when a cluster's mass vanishes | ✅ `test_near_empty_cluster_does_not_explode` — forces two of three clusters to ~0 mass. `MIN_OCCUPANCY = 1.0` floors the divisor; without it a 1e-8 mass gives a std of ~1e4 straight into the reparameterization |
| Gradients flow | Encoder and assignment both receive gradient | ✅ |
| Control measured (FR-3) | `(z̃, A_z) + N` reported alongside | ✅ and `n_only` reproduces the bar (0.7555) **from the same code path** as every other row, rather than quoting 0.7552 from a separate script |
| Bars reported (NFR-1) | Against both | ✅ |
| Full suite / lint | `pytest tests/`, `ruff check src/` | ✅ 304/304; `src/` clean |

### Bug found and fixed in the ablation's own reporting

`variational_feature_ablation.py` computed "did we clear the `N`-only bar?"
as `max()` over **all** feature sets — including `z_a_n`, which contains the
raw particle count. It therefore reported the bar cleared on the strength of a
feature set that literally contains `N`, which is circular: the bar exists to
ask what the *latent representation* encodes. Fixed to take the best over
encoded sets only, report the control separately as a gap, and distinguish a
tie (within one std) from an outright clear. The first T5.3 run was read
against the broken version, and both numbers above are from the corrected one.

### Follow-ups

- **The remaining 0.0130 gap to the concatenation control.** The pooled
  variance now carries `n_m` per cluster, but the classifier has to recover
  `N` from four log-variances rather than reading it directly, and the
  occupancy-aware runs' larger std suggests that inference is noisy. Worth
  checking whether `M` (currently 4) is the binding constraint.
- **`z_a_mu_logvar` beats `z_a_logvar` here** (+0.0073) where under T5.2 they
  were a wash (+0.0009). With occupancy scaling, `mu` and `log_var` are no
  longer redundant in the same way — worth a look if the gap matters.
- **T5.8's angle-scaling risk just got sharper.** Occupancy scaling shifts
  `log_var` by `−log(n_m)`, i.e. by up to `−log(139/4) ≈ −3.5` nats relative
  to before. FR-8 already requires measuring `log_var`'s empirical range
  before encoding it as a rotation angle; that measurement must be taken on
  an occupancy-aware checkpoint if T5.3 is adopted, not the T5.2 one.

## V-4: Learned mixture prior (T5.4) ⬜ Not started

## V-5: Graph-MRF `λ` sweep (T5.5) ⬜ Not started

## V-6: Variational latent graph `A_z` (T5.6) ⬜ Not started

## V-8: `log_var` encoded into the QGNN circuit (T5.8) ⬜ Queued, not implemented

Specced 2026-08-05 in response to V-2's finding. Nothing measured yet. The two
numbers to establish before any training run, per FR-8:

1. the empirical range of `log_var` over real jets (angles wrap at `2π`; a
   badly-scaled encoding would alias distinct values together and look like
   the idea failing rather than the encoding failing);
2. pub count per `.backward()` versus the pre-T5.8 circuit — expected
   unchanged, but the `readout_mode="learned"` "free" claim was measured false
   under the same expectation (`specs/phase4/validation.md` V-11 Step 3c).

## V-7: Final QGNN run and `README.md` (T5.7) ⬜ Not started
