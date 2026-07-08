"""Rate-distortion sweep for graph compression (T3.3).

For each (latent_dim, k) grid point, trains a fresh GVLS model on the full
input graph (no held-out split -- see gvls.data.splits.full_graph_split) and
measures how compressed (z_tilde, A_z) is relative to the input graph (X, A):
dim_compression_ratio (d/F), edge_compression_ratio (|A_z|/|E|), and
reconstruction fidelity (F1, bits-per-edge). All other hyperparameters are
held fixed to the dataset's Phase 2 NAS-best config (configs/best/{name}.yaml)
-- only latent_dim and k vary, independent of the AUC-optimal k that NAS chose
(see specs/phase3/plan.md, Design Decisions #2).

Usage:
    # full sweep on Cora (36 grid points, 200 epochs each):
    python experiments/compression_sweep.py data=cora

    # quick check with a smaller grid / epoch budget:
    python experiments/compression_sweep.py \
        experiment.latent_dim=[8,16] experiment.k=[2,5] train.epochs=10
"""

import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from gvls.compression import (
    evaluate_compression,
    select_compression_optimal,
    train_gvls_full_graph,
    write_results_csv,
)
from gvls.data import full_graph_split, load_planetoid


@hydra.main(version_base=None, config_path="../configs", config_name="compression_sweep_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.data.name.lower()

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading {cfg.data.name}...")
    data = load_planetoid(cfg.data.name)
    split = full_graph_split(data, seed=cfg.train.seed)

    x = data.x.to(device)
    train_ei = split.train_edge_index.to(device)
    n_nodes = split.n_nodes
    in_channels = int(x.size(1))

    adj_true = torch.zeros(n_nodes, n_nodes, device=device)
    adj_true[train_ei[0], train_ei[1]] = 1.0
    num_input_edges = int(train_ei.size(1) // 2)
    pos_weight = float((n_nodes * n_nodes - train_ei.size(1)) / train_ei.size(1))

    src, dst = train_ei[0], train_ei[1]
    pos_edge_index = train_ei[:, src < dst].cpu()

    print(f"  nodes={n_nodes}  features={in_channels}  edges={num_input_edges}")

    # ── fixed hyperparameters (Phase 2 NAS-best) ────────────────────────────────
    best_cfg_path = f"configs/best/{dataset_name}.yaml"
    base_cfg = OmegaConf.load(best_cfg_path)
    print(
        f"Fixed hyperparameters from {best_cfg_path}: graph_method={base_cfg.graph_method} "
        f"prior={base_cfg.prior} mp_rounds={base_cfg.mp_rounds} hidden_dim={base_cfg.hidden_dim} "
        f"lr={base_cfg.lr} beta={base_cfg.beta} lambda_={base_cfg.lambda_}"
    )

    latent_dims = list(cfg.experiment.latent_dim)
    ks = list(cfg.experiment.k)
    epochs = int(cfg.train.epochs)
    seed = int(cfg.train.seed)

    print(
        f"\nRate-distortion sweep: {len(latent_dims)}x{len(ks)} = "
        f"{len(latent_dims) * len(ks)} grid points, {epochs} epochs each\n"
    )

    # ── grid sweep ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for latent_dim in latent_dims:
        for k in ks:
            latent_dim, k = int(latent_dim), int(k)
            run_name = f"{dataset_name}-d{latent_dim}-k{k}"
            wandb.init(
                project=cfg.wandb.project,
                mode=cfg.wandb.mode,
                name=run_name,
                group=f"compression-sweep-{dataset_name}",
                config={
                    "latent_dim": latent_dim,
                    "k": k,
                    **OmegaConf.to_container(base_cfg, resolve=True),
                },
                reinit=True,
            )

            model = train_gvls_full_graph(
                x, train_ei, adj_true, pos_weight, in_channels,
                latent_dim, k, base_cfg, epochs, seed, device,
            )
            metrics = evaluate_compression(
                model, x, train_ei, adj_true, pos_edge_index, n_nodes,
                in_channels, num_input_edges, latent_dim, k,
                f1_negative_ratio=float(cfg.experiment.f1_negative_ratio),
                dense_pair_limit=int(cfg.experiment.dense_pair_limit),
                bpe_sample_size=int(cfg.experiment.bpe_sample_size),
                seed=seed, device=device,
            )
            rows.append({"dataset": dataset_name, **metrics})

            wandb.log(metrics)
            wandb.finish()

            print(
                f"  d={latent_dim:<4} k={k:<3} "
                f"dim_ratio={metrics['dim_compression_ratio']:.4f} "
                f"edge_ratio={metrics['edge_compression_ratio']:.4f} "
                f"f1={metrics['reconstruction_f1']:.4f} "
                f"bpe={metrics['bits_per_edge']:.4f}"
            )

    # ── results ──────────────────────────────────────────────────────────────
    csv_path = f"results/compression/{dataset_name}.csv"
    write_results_csv(rows, csv_path)
    print(f"\nResults written to {csv_path}")

    best, floor_met = select_compression_optimal(rows, float(cfg.experiment.fidelity_floor))
    if not floor_met:
        print(
            f"\nWARNING: no grid point met the fidelity floor "
            f"(F1 >= {cfg.experiment.fidelity_floor}); falling back to the "
            f"highest-F1 point found (F1={best['reconstruction_f1']:.4f})."
        )

    compression_cfg = {
        "name": "gvls",
        "latent_dim": int(best["latent_dim"]),
        "hidden_dim": int(base_cfg.hidden_dim),
        "mp_rounds": int(base_cfg.mp_rounds),
        "graph_method": str(base_cfg.graph_method),
        "prior": str(base_cfg.prior),
        "k": int(best["k"]),
        "beta": float(base_cfg.beta),
        "lambda_": float(base_cfg.lambda_),
        "lr": float(base_cfg.lr),
    }
    os.makedirs("configs/compression", exist_ok=True)
    out_path = f"configs/compression/{dataset_name}.yaml"
    OmegaConf.save(OmegaConf.create(compression_cfg), out_path)
    print(
        f"\nCompression-optimal config: d={best['latent_dim']} k={best['k']} "
        f"(F1={best['reconstruction_f1']:.4f}, "
        f"dim_ratio={best['dim_compression_ratio']:.4f}, "
        f"edge_ratio={best['edge_compression_ratio']:.4f}) saved to {out_path}"
    )


if __name__ == "__main__":
    main()
