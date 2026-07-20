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
from gvls.data.jets import NUM_FEATURES, load_qg_jets, split_jets


@hydra.main(version_base=None, config_path="../configs", config_name="jet_pretrain_final_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        name=f"qg_jets-gvls-final-M{m}",
        group="jet-gvls-final",
        config={**base_cfg, "num_jets": int(cfg.data.num_jets)},
    )

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
    )
    wandb.finish()

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


if __name__ == "__main__":
    main()
