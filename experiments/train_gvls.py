"""Train GVLS on a Planetoid dataset.

Usage:
    # default (Cora, 80% train, 200 epochs):
    python experiments/train_gvls.py

    # override dataset, split ratio, or model hyperparameters:
    python experiments/train_gvls.py data=citeseer train.split_ratio=0.4
    python experiments/train_gvls.py model.graph_method=fgp model.prior=graph_mrf

    # sweep:
    python experiments/train_gvls.py -m data=cora,citeseer,pubmed
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from gvls.data import load_planetoid, split_edges
from gvls.eval import auc_ap
from gvls.losses.elbo import elbo, kl_isotropic, kl_graph_mrf
from gvls.models.encoder import GVLSEncoder
from gvls.models.gvls import GVLS
from gvls.models.latent_graph import LatentGraphLearner


@hydra.main(version_base=None, config_path="../configs", config_name="train_gvls_config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = (
        f"{cfg.data.name}-{cfg.model.graph_method}-{cfg.model.prior}"
        f"-r{cfg.train.split_ratio}-s{cfg.train.seed}"
    )
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading {cfg.data.name}...")
    data = load_planetoid(cfg.data.name)
    split = split_edges(data, train_ratio=cfg.train.split_ratio, seed=cfg.train.seed)

    x = data.x.to(device)                                    # (N, F)
    train_ei = split.train_edge_index.to(device)             # (2, 2·E_train)
    N = split.n_nodes
    in_channels = x.size(1)

    # Dense binary adjacency for reconstruction loss (both directions)
    adj_true = torch.zeros(N, N, device=device)
    adj_true[train_ei[0], train_ei[1]] = 1.0

    # Positive-edge weight to counter the extreme class imbalance in sparse graphs
    # (VGAE convention: upweight positives so they contribute equally to the loss)
    n_edges = adj_true.sum().item()
    pos_weight = float((N * N - n_edges) / n_edges)

    n_val  = split.val_pos.size(1)
    n_test = split.test_pos.size(1)
    print(f"  nodes={N}  train_edges={train_ei.size(1)//2}  val={n_val}  test={n_test}")

    # ── model ─────────────────────────────────────────────────────────────────
    encoder = GVLSEncoder(
        in_channels=in_channels,
        hidden_channels=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
    )
    lgl = LatentGraphLearner(
        latent_dim=cfg.model.latent_dim,
        method=cfg.model.graph_method,
        k=cfg.model.k,
    )
    model = GVLS(encoder, lgl, latent_dim=cfg.model.latent_dim, mp_rounds=cfg.model.mp_rounds)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        optimizer.zero_grad()

        mu, log_var, _z, A_z, z_tilde = model(x, train_ei)
        recon_logits = z_tilde @ z_tilde.T

        # Compute KL separately for logging
        if cfg.model.prior == "isotropic":
            kl_val = kl_isotropic(mu, log_var)
        else:
            kl_val = kl_graph_mrf(mu, log_var, A_z, lambda_=cfg.model.lambda_)

        recon_loss = F.binary_cross_entropy_with_logits(
            recon_logits, adj_true,
            pos_weight=torch.tensor(pos_weight, device=device),
            reduction="mean",
        )
        loss = elbo(
            recon_logits, adj_true, mu, log_var, A_z,
            beta=cfg.model.beta, lambda_=cfg.model.lambda_, prior=cfg.model.prior,
            pos_weight=pos_weight,
        )

        loss.backward()
        optimizer.step()

        # ── evaluation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            _, _, _, A_z_eval, z_tilde_eval = model(x, train_ei)
            scores_all = z_tilde_eval @ z_tilde_eval.T  # (N, N)

            def _score_pairs(pos: torch.Tensor, neg: torch.Tensor) -> tuple[float, float]:
                pos = pos.to(device)
                neg = neg.to(device)
                labels = np.concatenate([np.ones(pos.size(1)), np.zeros(neg.size(1))])
                sc = torch.cat([
                    scores_all[pos[0], pos[1]],
                    scores_all[neg[0], neg[1]],
                ]).cpu().numpy()
                return auc_ap(labels, sc)

            val_auc, val_ap = _score_pairs(split.val_pos, split.val_neg)
            density = (A_z_eval > 0).float().mean().item()

        log = {
            "train/elbo":   loss.item(),
            "train/recon":  recon_loss.item(),
            "train/kl":     kl_val.item(),
            "val/auc":      val_auc,
            "val/ap":       val_ap,
            "latent/density": density,
        }
        wandb.log(log, step=epoch)

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  epoch={epoch:4d}  loss={loss.item():.4f}"
                f"  val_auc={val_auc:.4f}  density={density:.4f}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/best.pt")

    # ── final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load("checkpoints/best.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        _, _, _, _, z_tilde_test = model(x, train_ei)
        scores_all = z_tilde_test @ z_tilde_test.T
        test_pos = split.test_pos.to(device)
        test_neg = split.test_neg.to(device)
        labels = np.concatenate([np.ones(n_test), np.zeros(n_test)])
        sc = torch.cat([
            scores_all[test_pos[0], test_pos[1]],
            scores_all[test_neg[0], test_neg[1]],
        ]).cpu().numpy()
        test_auc, test_ap = auc_ap(labels, sc)

    wandb.log({"test/auc": test_auc, "test/ap": test_ap})
    print(f"\n  best val_auc={best_val_auc:.4f}")
    print(f"  test auc={test_auc:.4f}  ap={test_ap:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
