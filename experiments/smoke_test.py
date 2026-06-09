"""Smoke test — validates the full data + eval + logging stack.

Loads a Planetoid dataset, splits edges, runs a dummy random predictor,
and logs all metrics to W&B (offline by default).

Usage:
    # single run (Cora, 80% train, seed 42):
    python experiments/smoke_test.py

    # override dataset and split ratio:
    python experiments/smoke_test.py data=citeseer train.split_ratio=0.4

    # sweep all datasets × all ratios:
    python experiments/smoke_test.py -m \
        data=cora,citeseer,pubmed \
        train.split_ratio=0.2,0.4,0.8
"""

import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from gvls.data import load_planetoid, split_edges
from gvls.eval import auc_ap, bits_per_edge


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_name = f"{cfg.data.name}-r{cfg.train.split_ratio}-s{cfg.train.seed}"

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

    n_train = split.train_edge_index.size(1) // 2
    n_val   = split.val_pos.size(1)
    n_test  = split.test_pos.size(1)

    print(f"  nodes={split.n_nodes}  train={n_train}  val={n_val}  test={n_test}")

    # ── dummy predictor ───────────────────────────────────────────────────────
    # Random scores from a fixed seed — gives AUC ≈ 0.5, confirming metric
    # plumbing works before any real model is implemented.
    rng = np.random.default_rng(cfg.train.seed)

    val_labels  = np.concatenate([np.ones(n_val),  np.zeros(n_val)])
    test_labels = np.concatenate([np.ones(n_test), np.zeros(n_test)])
    val_scores  = rng.random(2 * n_val)
    test_scores = rng.random(2 * n_test)

    val_auc,  val_ap  = auc_ap(val_labels,  val_scores)
    test_auc, test_ap = auc_ap(test_labels, test_scores)

    # bits_per_edge: logit=0 everywhere → 1.0 bit/edge (sanity baseline)
    bpe = bits_per_edge(
        np.concatenate([np.ones(n_test), np.zeros(n_test)]),
        np.zeros(2 * n_test),
    )

    # ── logging ───────────────────────────────────────────────────────────────
    metrics = {
        "n_nodes":      split.n_nodes,
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       n_test,
        "val/auc":      val_auc,
        "val/ap":       val_ap,
        "test/auc":     test_auc,
        "test/ap":      test_ap,
        "bits_per_edge": bpe,
    }
    wandb.log(metrics)

    print(
        f"  val  auc={val_auc:.4f}  ap={val_ap:.4f}\n"
        f"  test auc={test_auc:.4f}  ap={test_ap:.4f}  bpe={bpe:.4f}"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
