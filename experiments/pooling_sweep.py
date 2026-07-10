"""Node-count pooling sweep for graph compression (T3.6).

For each `pool_ratio` grid point, trains a fresh PooledGVLS model on the full
input graph (no held-out split -- see gvls.data.splits.full_graph_split) with
(latent_dim, k, hidden_dim, mp_rounds, graph_method, prior, beta, lambda_, lr)
held fixed to the dataset's T3.3 compression-optimal config
(configs/compression/{name}.yaml) -- only the pooled node count
M = round(pool_ratio * N) varies. This isolates the node-count compression
axis from the (d, k) axes T3.3 already swept (see specs/phase3/plan.md, T3.6).

Usage:
    # full sweep on Cora (4 grid points, 200 epochs each):
    python experiments/pooling_sweep.py data=cora

    # quick check with a smaller grid / epoch budget:
    python experiments/pooling_sweep.py \
        experiment.pool_ratio=[0.5,0.25] train.epochs=10
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.compression.pooling_sweep import (
    RESULT_FIELDS,
    evaluate_pooled_compression,
    train_pooled_gvls_full_graph,
)
from gvls.compression.sweep import write_results_csv
from gvls.data import full_graph_split, load_planetoid


@hydra.main(version_base=None, config_path="../configs", config_name="pooling_sweep_config")
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

    # ── fixed hyperparameters (T3.3 compression-optimal) ────────────────────
    compression_cfg_path = f"configs/compression/{dataset_name}.yaml"
    base_cfg = OmegaConf.load(compression_cfg_path)
    latent_dim = int(base_cfg.latent_dim)
    k = int(base_cfg.k)
    print(
        f"Fixed hyperparameters from {compression_cfg_path}: latent_dim={latent_dim} k={k} "
        f"graph_method={base_cfg.graph_method} prior={base_cfg.prior} "
        f"mp_rounds={base_cfg.mp_rounds} hidden_dim={base_cfg.hidden_dim} "
        f"lr={base_cfg.lr} beta={base_cfg.beta} lambda_={base_cfg.lambda_}"
    )

    pool_ratios = [float(pr) for pr in cfg.experiment.pool_ratio]
    epochs = int(cfg.train.epochs)
    seed = int(cfg.train.seed)

    print(f"\nNode-count pooling sweep: {len(pool_ratios)} grid points, {epochs} epochs each\n")

    # ── grid sweep ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for pool_ratio in pool_ratios:
        num_clusters = max(2, round(pool_ratio * n_nodes))
        run_name = f"{dataset_name}-pool{pool_ratio}"
        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=run_name,
            group=f"compression-pooling-sweep-{dataset_name}",
            config={
                "pool_ratio": pool_ratio,
                "num_clusters": num_clusters,
                **OmegaConf.to_container(base_cfg, resolve=True),
            },
            reinit=True,
        )

        model = train_pooled_gvls_full_graph(
            x, train_ei, adj_true, pos_weight, in_channels,
            latent_dim, k, num_clusters, base_cfg, epochs, seed, device,
            entropy_weight=float(cfg.experiment.entropy_weight),
            aux_link_weight=float(cfg.experiment.aux_link_weight),
        )
        metrics = evaluate_pooled_compression(
            model, x, train_ei, adj_true, pos_edge_index, n_nodes, num_clusters,
            in_channels, num_input_edges, latent_dim, k, pool_ratio,
            f1_negative_ratio=float(cfg.experiment.f1_negative_ratio),
            dense_pair_limit=int(cfg.experiment.dense_pair_limit),
            bpe_sample_size=int(cfg.experiment.bpe_sample_size),
            seed=seed, device=device,
        )
        rows.append({"dataset": dataset_name, **metrics})

        wandb.log(metrics)
        wandb.finish()

        print(
            f"  pool_ratio={pool_ratio:<6} M={num_clusters:<5} "
            f"node_ratio={metrics['node_compression_ratio']:.4f} "
            f"f1={metrics['reconstruction_f1']:.4f} "
            f"bpe={metrics['bits_per_edge']:.4f}"
        )

    # ── results ──────────────────────────────────────────────────────────────
    csv_path = f"results/compression/{dataset_name}_pooling.csv"
    write_results_csv(rows, csv_path, fieldnames=RESULT_FIELDS)
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
