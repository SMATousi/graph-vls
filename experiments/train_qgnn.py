"""Two-stage supervised QGNN training on frozen GVLS features (T4.5).

Loads the frozen, pretrained PooledGVLS checkpoint (from
experiments/pretrain_gvls_jets_final.py), extracts (z_tilde, A_z) once for
every jet in the train/val split (no further gradient updates to GVLS --
plan.md Design Decision 8), then trains QGNNClassifier's circuit parameters
(theta, b_i, the readout rotation -- src/gvls/models/qgnn.py) supervised on
the quark/gluon label via Adam, gradient-accumulated over minibatches of
jets. Tracks the full metric suite (accuracy, AUC, AP, macro-F1, precision,
recall, confusion matrix) on the validation split every epoch and
checkpoints whichever epoch had the best validation accuracy. Test-set
evaluation is a separate step (experiments/evaluate_qgnn.py).

Usage:
    python experiments/train_qgnn.py
    python experiments/train_qgnn.py train.epochs=100 train.num_layers=2
    python experiments/train_qgnn.py gvls_checkpoint_path=checkpoints/gvls_jets_m6.pt \
        qgnn_checkpoint_path=checkpoints/qgnn_jets_m6.pt
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.compression.jet_sweep import load_gvls_checkpoint
from gvls.data.jets import load_qg_jets, split_jets
from gvls.qgnn_training import extract_latent_features, save_qgnn_checkpoint, train_qgnn_classifier


@hydra.main(version_base=None, config_path="../configs", config_name="qgnn_train_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading frozen GVLS checkpoint from {cfg.gvls_checkpoint_path}...")
    gvls_model, gvls_config = load_gvls_checkpoint(str(cfg.gvls_checkpoint_path), device)
    m = int(gvls_config["num_clusters"])
    d = int(gvls_config["latent_dim"])
    print(f"  M={m}  d={d}")

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
    print(f"  train={len(split.train)}  val={len(split.val)}  test(unused here)={len(split.test)}")

    print("Extracting frozen (z_tilde, A_z) features (no gradient)...")
    train_features = extract_latent_features(gvls_model, split.train, device)
    val_features = extract_latent_features(gvls_model, split.val, device)

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    print(
        f"QGNN config: M={m} num_layers={train_cfg['num_layers']} lr={train_cfg['lr']} "
        f"epochs={train_cfg['epochs']} batch_size={train_cfg['batch_size']}"
    )

    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=f"qgnn-M{m}",
        group="qgnn-jet-classification",
        config={"m": m, "d": d, **train_cfg},
    )

    result = train_qgnn_classifier(
        train_features,
        val_features,
        m=m,
        d=d,
        num_layers=int(train_cfg["num_layers"]),
        lr=float(train_cfg["lr"]),
        epochs=int(train_cfg["epochs"]),
        seed=int(train_cfg["seed"]),
        device=device,
        batch_size=int(train_cfg["batch_size"]),
    )

    for row in result.history:
        wandb.log(row)
    wandb.finish()

    best = result.best_val_metrics
    print(
        f"\nBest epoch={result.best_epoch}  val_accuracy={best['accuracy']:.4f}  "
        f"val_auc={best['auc']:.4f}  val_macro_f1={best['macro_f1']:.4f}"
    )

    config = {"m": m, "d": d, "num_layers": int(train_cfg["num_layers"])}
    save_qgnn_checkpoint(result.best_state_dict, config, str(cfg.qgnn_checkpoint_path))
    print(f"Saved best QGNN checkpoint to {cfg.qgnn_checkpoint_path}")


if __name__ == "__main__":
    main()
