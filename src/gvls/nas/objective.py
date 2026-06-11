from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from gvls.data.splits import EdgeSplit
from gvls.eval.metrics import auc_ap
from gvls.losses.elbo import elbo
from gvls.models.encoder import GVLSEncoder
from gvls.models.gvls import GVLS
from gvls.models.latent_graph import LatentGraphLearner
from gvls.nas.search_space import suggest_config


def make_objective(
    data: Data,
    split: EdgeSplit,
    cfg: DictConfig,
    device: torch.device,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective closure for one dataset.

    Expensive setup (dense adjacency, pos_weight) is done once here and shared
    across all trials.  Each trial recreates the model and optimizer from scratch.

    cfg must expose cfg.nas.epochs_per_trial and cfg.nas.include_nri.
    """
    # ── shared precomputation ────────────────────────────────────────────────
    x = data.x.to(device)
    train_ei = split.train_edge_index.to(device)
    N = split.n_nodes
    in_channels = int(x.size(1))

    adj_true = torch.zeros(N, N, device=device)
    adj_true[train_ei[0], train_ei[1]] = 1.0
    n_edges = float(adj_true.sum().item())
    pw = (N * N - n_edges) / n_edges      # positive-edge weight (VGAE convention)

    val_pos = split.val_pos   # kept on CPU; moved inside closure
    val_neg = split.val_neg
    n_val = val_pos.size(1)

    epochs: int = cfg.nas.epochs_per_trial
    mid: int = epochs // 2
    include_nri: bool = cfg.nas.include_nri

    # ── closure ──────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(trial.number * 42)

        tc = suggest_config(trial, include_nri=include_nri)

        encoder = GVLSEncoder(in_channels, tc["hidden_dim"], tc["latent_dim"])
        lgl = LatentGraphLearner(tc["latent_dim"], method=tc["graph_method"], k=tc["k"])
        model = GVLS(encoder, lgl, latent_dim=tc["latent_dim"], mp_rounds=tc["mp_rounds"])
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

        best_val_auc = 0.0

        for epoch in range(1, epochs + 1):
            # ── train step ───────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            mu, log_var, _z, A_z, z_tilde = model(x, train_ei)
            recon_logits = z_tilde @ z_tilde.T
            loss = elbo(
                recon_logits, adj_true, mu, log_var, A_z,
                beta=tc["beta"], lambda_=tc["lambda_"], prior=tc["prior"],
                pos_weight=pw,
            )
            loss.backward()
            optimizer.step()

            # ── eval step ────────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                _, _, _, _, z_eval = model(x, train_ei)
                scores = z_eval @ z_eval.T
                vp = val_pos.to(device)
                vn = val_neg.to(device)
                labels = np.concatenate([np.ones(n_val), np.zeros(n_val)])
                sc = torch.cat([scores[vp[0], vp[1]], scores[vn[0], vn[1]]]).cpu().numpy()
                val_auc, _ = auc_ap(labels, sc)

            best_val_auc = max(best_val_auc, val_auc)

            # ── pruning checkpoint at halfway ─────────────────────────────────
            if epoch == mid:
                trial.report(val_auc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_val_auc

    return objective
