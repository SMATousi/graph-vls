"""Train and persist a production PooledGVLS checkpoint at a fixed M.

T4.3's sweep (experiments/pretrain_gvls_jets.py) trains a fresh PooledGVLS per
M in a grid purely to compare compression fidelity across M -- it never
saves a checkpoint. T4.5 needs one frozen, persisted model at the
compression-optimal M (M=4 was selected in specs/phase4/validation.md V-3) to
extract (z_tilde, A_z) from for the QGNN classifier. This script is that
prerequisite: it trains that one production model and saves it via
save_gvls_checkpoint.

Usage:
    python experiments/pretrain_gvls_jets_final.py
    python experiments/pretrain_gvls_jets_final.py train.m=6 train.epochs=200
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.compression.jet_sweep import save_gvls_checkpoint, train_pooled_gvls_on_jets
from gvls.data.jets import NUM_FEATURES, load_split_from_config


@hydra.main(version_base=None, config_path="../configs", config_name="jet_pretrain_final_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    protocol = data_cfg.get("protocol", "balanced")
    print(f"Loading qg_jets (protocol={protocol}, seed={cfg.data.seed})...")
    split = load_split_from_config(data_cfg)
    print(f"  train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")

    base_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    m = int(base_cfg["m"])
    k = min(int(base_cfg["k"]), m - 1)
    print(
        f"Production config: M={m} latent_dim={base_cfg['latent_dim']} k={k} "
        f"hidden_dim={base_cfg['hidden_dim']} graph_method={base_cfg['graph_method']} "
        f"prior={base_cfg['prior']} epochs={base_cfg['epochs']}"
    )

    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name or f"qg_jets-gvls-final-M{m}",
        group=cfg.wandb.group or "jet-gvls-final",
        tags=list(cfg.wandb.tags),
        config={**base_cfg, "data": data_cfg},
    )

    last_metrics: dict = {}

    def _on_epoch_end(epoch: int, metrics: dict) -> None:
        last_metrics.update(metrics)
        wandb.log(metrics, step=epoch)

    model = train_pooled_gvls_on_jets(
        split.train,
        in_channels=NUM_FEATURES,
        latent_dim=int(base_cfg["latent_dim"]),
        k=k,
        num_clusters=m,
        base_cfg=base_cfg,
        epochs=int(base_cfg["epochs"]),
        seed=int(base_cfg["seed"]),
        device=device,
        batch_size=int(base_cfg["batch_size"]),
        progress_desc=f"pretrain GVLS (production, M={m})",
        eval_jets=split.val,
        on_epoch_end=_on_epoch_end,
        # T5.1: falls back to the pre-Phase-5 criterion if a config predating
        # specs/phase5/ is used, rather than silently changing its behaviour.
        selection_metric=str(base_cfg.get("selection_metric", "reconstruction_f1")),
    )

    config = {
        "in_channels": NUM_FEATURES,
        "latent_dim": int(base_cfg["latent_dim"]),
        "k": k,
        "num_clusters": m,
        "base_cfg": base_cfg,
    }
    checkpoint_path = str(cfg.checkpoint_path)
    save_gvls_checkpoint(model, config, checkpoint_path)
    print(f"\nSaved production GVLS checkpoint (M={m}) to {checkpoint_path}")

    artifact = wandb.Artifact(
        name=f"gvls-jets-m{m}",
        type="model",
        metadata={
            "m": m,
            "latent_dim": int(base_cfg["latent_dim"]),
            "k": k,
            **{key: val for key, val in last_metrics.items() if key != "epoch"},
        },
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact, aliases=["latest"])
    wandb.finish()


if __name__ == "__main__":
    main()
