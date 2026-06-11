# Phase 1 — Validation

## Exit Criteria

---

## V-1: Encoder ⬜

| Check | Pass condition | Result |
|---|---|---|
| Output shapes | `mu`, `log_var`, `z` all have shape `(N, latent_dim)` | ⬜ |
| Gradient flow | `loss.backward()` produces non-None grad on `encoder.layer1.lin.weight` | ⬜ |
| Eval-mode determinism | `z == mu` when `model.eval()` | ⬜ |
| log_σ² clamping | Input log_var=100 maps to clamped value ≤ 10 | ⬜ |

---

## V-2: Latent Graph Inference ⬜

| Check | Pass condition | Result |
|---|---|---|
| Output shape | A_z has shape `(N, N)` for all three methods | ⬜ |
| Value range | All entries in [0, 1] | ⬜ |
| Zero diagonal | `A_z.diagonal()` is all zeros | ⬜ |
| Symmetry | `A_z == A_z.T` (within float tolerance) | ⬜ |
| Sparsification | At most k non-zero entries per row | ⬜ |
| Gradient flow | Gradients reach learner parameters for attention, fgp, and nri | ⬜ |

---

## V-3: Full Model and Latent Message Passing ⬜

| Check | Pass condition | Result |
|---|---|---|
| Output shapes | All five outputs have correct shapes on a synthetic graph | ⬜ |
| mp_rounds=0 | z̃ == z (message passing is a no-op) | ⬜ |
| Gradient end-to-end | Grad reaches `encoder.layer1` after backward through z̃ | ⬜ |

---

## V-4: ELBO Loss ⬜

| Check | Pass condition | Result |
|---|---|---|
| Isotropic KL baseline | `kl_isotropic(mu=0, log_var=0)` returns 0.0 | ⬜ |
| Isotropic KL positivity | `kl_isotropic(mu=1, log_var=0)` > 0 | ⬜ |
| Graph-MRF KL finite | `kl_graph_mrf` returns finite scalar on valid A_z | ⬜ |
| NaN guard | `RuntimeError` raised when loss is NaN | ⬜ |

---

## V-5: Training Run ⬜

| Check | Pass condition | Result |
|---|---|---|
| No NaN losses | Zero NaN/Inf losses across all epochs on Cora (default config) | ⬜ |
| ELBO decreasing | ELBO at epoch 50 < ELBO at epoch 1 | ⬜ |
| A_z ≠ input adjacency | `|density(A_z) − density(input adj)| ≥ 0.05` | ⬜ |
| Val AUC sanity | Val AUC > 0.6 after 200 epochs | ⬜ |
| W&B logging | `train/elbo`, `train/kl`, `train/recon`, `val/auc`, `val/ap`, `latent/density` all logged | ⬜ |
| Checkpoint saved | `checkpoints/best.pt` exists after training | ⬜ |
| `pytest tests/` | All tests pass (including new Phase 1 tests) | ⬜ |
| `ruff check src/` | Zero violations | ⬜ |
