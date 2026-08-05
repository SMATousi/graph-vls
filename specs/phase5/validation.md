# Phase 5 — Validation

**Status: not started (spec written 2026-08-05).** V-0 below records the
measured evidence this phase was created from — the 2026-08-04 mission-
conformance audit, the 24-config `(k, β, prior)` sweep, and the multiplicity
baseline that reset the phase's acceptance bar. V-1 through V-7 are
placeholders to be filled as T5.1–T5.7 land, mirroring Phases 0–4's
convention of recording negative results and surprises alongside successes.

## Exit Criteria

- [ ] (T5.1, FR-1) Per-jet ELBO normalization is `N`-invariant; the `(k, β)`
      operating point re-established afterwards; checkpoint-selection
      criterion no longer reconstruction F1 alone
- [ ] (T5.2, FR-2) `log_var` reaches the downstream classifier; the
      `(z̃, A_z)` vs `(z̃, A_z, log_var)` ablation measured and reported
- [ ] (T5.3, FR-3) Occupancy-aware pooled posterior implemented and measured
      against both today's behavior (0.7034) and the concatenation control
      (0.7683)
- [ ] (T5.4, FR-4) Learned mixture prior implemented, with free bits/warm-up,
      evaluated at a `β` where the prior can express itself
- [ ] (T5.5, FR-5) Graph-MRF `λ` swept jointly with `β`; a documented answer
      to whether a graph prior can help rather than merely survive
- [ ] (T5.6, FR-6) Variational `A_z` with learned, jet-specific topology;
      deterministic path still selectable
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
(**2.35×**).

**Interpretation.** Particle multiplicity is the classic quark/gluon
discriminant, and fixed-`M` pooling destroys it by construction — every jet
becomes `M=4` nodes regardless of size, and `LatentGraphPooling`
column-normalizes (`w = s / col_sum`), explicitly dividing occupancy out.
Some leaks back through the degree-normalized GCN
(`max |corr(z̃ dim, N)| = 0.67`, mean 0.385) but not enough. **The full
GVLS→QGNN pipeline currently performs worse than counting particles.**

The same `N`-dependence appears inside the objective: reconstruction is
mean-reduced over `N²` pairs while the KL is divided by `M`, so
`β_eff = β · N²/M ≈ 551β` at the mean — which also explains Part B's
collapse at `β=1.0` (`β_eff ≈ 551`) and why `β=0.001` (`β_eff ≈ 0.55`) sits
near a proper ELBO. Because `N` is strongly class-correlated, the
regularization strength is itself a function of the label. This is FR-1.

---

## V-1: Consistent per-jet ELBO normalization (T5.1) ⬜ Not started

## V-2: Variational output reaches the downstream task (T5.2) ⬜ Not started

## V-3: Occupancy-aware pooled posterior (T5.3) ⬜ Not started

## V-4: Learned mixture prior (T5.4) ⬜ Not started

## V-5: Graph-MRF `λ` sweep (T5.5) ⬜ Not started

## V-6: Variational latent graph `A_z` (T5.6) ⬜ Not started

## V-7: Final QGNN run and `README.md` (T5.7) ⬜ Not started
