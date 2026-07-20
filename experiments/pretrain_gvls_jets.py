"""Per-jet GVLS pretraining sweep over fixed M (T4.3).

For each M in cfg.experiment.m_grid, trains a fresh PooledGVLS unsupervised
(ELBO only) over the jet pretraining split, iterating jets one at a time with
gradient-accumulated minibatches (T4.2), then evaluates average per-jet
reconstruction_f1/bits_per_edge on a held-out split. (hidden_dim, latent_dim,
k, graph_method, prior, mp_rounds, lr, beta, lambda_) come from
configs/train/jet_pretrain.yaml (a Phase 2/3-derived starting point, not
re-tuned via NAS for jets -- see plan.md T4.3). Unlike Phases 0-3, no labels
are used here: T4.3 is purely unsupervised pretraining, since labels are only
introduced in T4.5's QGNN classifier stage.

Usage:
    # full sweep (M in {4,6,8}):
    python experiments/pretrain_gvls_jets.py

    # quick check with a smaller grid / jet count / epoch budget:
    python experiments/pretrain_gvls_jets.py \
        experiment.m_grid=[4,6] data.num_jets=2000 train.epochs=10
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.compression.jet_sweep import (
    JET_RESULT_FIELDS,
    run_jet_compression_sweep,
    select_compression_optimal_m,
)
from gvls.compression.sweep import write_results_csv
from gvls.data.jets import NUM_FEATURES, load_qg_jets, split_jets


@hydra.main(version_base=None, config_path="../configs", config_name="jet_pretrain_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {cfg.data.num_jets} qg_jets (seed={cfg.data.seed})...")
    jets = load_qg_jets(
        num_jets=int(cfg.data.num_jets),
        k_graph_cap=int(cfg.data.k_graph_cap),
        seed=int(cfg.data.seed),
    )
    split = split_jets(
        jets,
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )
    # T4.3 is unsupervised (ELBO only, no labels) -- val jets serve as the
    # held-out set for compression-fidelity evaluation; test is untouched,
    # reserved for T4.5/T4.6's QGNN classifier evaluation.
    train_jets, eval_jets = split.train, split.val
    print(
        f"  train={len(train_jets)}  eval(val)={len(eval_jets)}  "
        f"test(untouched)={len(split.test)}"
    )

    base_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    m_grid = [int(m) for m in cfg.experiment.m_grid]
    print(
        f"Starting config: hidden_dim={base_cfg['hidden_dim']} latent_dim={base_cfg['latent_dim']} "
        f"k={base_cfg['k']} graph_method={base_cfg['graph_method']} prior={base_cfg['prior']} "
        f"mp_rounds={base_cfg['mp_rounds']} lr={base_cfg['lr']} beta={base_cfg['beta']} "
        f"epochs={base_cfg['epochs']} batch_size={base_cfg['batch_size']}"
    )
    print(f"\nJet compression sweep: M in {m_grid}\n")

    rows: list[dict] = []
    for m in m_grid:
        run_name = f"qg_jets-M{m}"
        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=run_name,
            group="jet-compression-sweep",
            config={"num_clusters": m, **base_cfg, "num_jets": int(cfg.data.num_jets)},
            reinit=True,
        )

        [row] = run_jet_compression_sweep(
            train_jets,
            eval_jets,
            m_grid=[m],
            in_channels=NUM_FEATURES,
            latent_dim=int(base_cfg["latent_dim"]),
            k=int(base_cfg["k"]),
            base_cfg=base_cfg,
            epochs=int(base_cfg["epochs"]),
            seed=int(base_cfg["seed"]),
            device=device,
            batch_size=int(base_cfg["batch_size"]),
            entropy_weight=float(cfg.experiment.entropy_weight),
            aux_link_weight=float(cfg.experiment.aux_link_weight),
            f1_negative_ratio=float(cfg.experiment.f1_negative_ratio),
        )
        rows.append(row)

        wandb.log(row)
        wandb.finish()

        print(
            f"  M={m:<3} avg_f1={row['avg_reconstruction_f1']:.4f} "
            f"avg_bpe={row['avg_bits_per_edge']:.4f} "
            f"avg_node_ratio={row['avg_node_compression_ratio']:.4f}"
        )

    csv_path = "results/compression/qg_jets_pooling.csv"
    write_results_csv(rows, csv_path, fieldnames=JET_RESULT_FIELDS)
    print(f"\nResults written to {csv_path}")

    best = select_compression_optimal_m(rows, tolerance=float(cfg.experiment.f1_tolerance))
    print(
        f"\nCompression-optimal M={best['num_clusters']} "
        f"(avg_f1={best['avg_reconstruction_f1']:.4f}, tolerance={cfg.experiment.f1_tolerance})"
    )


if __name__ == "__main__":
    main()
