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
    # T4.10: literature-comparability run (Lorentz-EQGNN, plan.md Design
    # Decision 12) -- 800-jet subset, AdamW/lr=1e-3/batch=16 already the
    # config defaults (configs/train/qgnn_classifier.yaml):
    python experiments/train_qgnn.py data.num_jets=800
"""

import time

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
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    optimizer_name = str(train_cfg.get("optimizer", "adam"))  # pre-T4.10 configs lack this key
    print(
        f"QGNN config: M={m} num_layers={train_cfg['num_layers']} "
        f"optimizer={optimizer_name} lr={train_cfg['lr']} "
        f"epochs={train_cfg['epochs']} batch_size={train_cfg['batch_size']} "
        f"gradient_method={train_cfg['gradient_method']}"
    )

    # T4.10 (validation.md V-10): data_cfg (notably num_jets) is logged so
    # runs at different dataset sizes (e.g. the 800-jet Lorentz-EQGNN
    # comparability subset vs. the 20000-jet target run) are distinguishable
    # in the W&B config/UI, not just by checkpoint inspection. wandb.name/
    # group/tags default to the pre-T4.10 behavior unless overridden.
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name or f"qgnn-M{m}",
        group=cfg.wandb.group or "qgnn-jet-classification",
        tags=list(cfg.wandb.tags),
        config={"m": m, "d": d, "data": data_cfg, **train_cfg},
    )

    training_start = time.perf_counter()
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
        on_epoch_end=lambda epoch, metrics: wandb.log(metrics, step=epoch),
        gradient_method=str(train_cfg["gradient_method"]),
        spsa_epsilon=float(train_cfg["spsa_epsilon"]),
        spsa_batch_size=int(train_cfg["spsa_batch_size"]),
        optimizer=optimizer_name,
    )
    training_time_s = time.perf_counter() - training_start

    best = result.best_val_metrics
    train_accuracy = result.best_train_metrics["accuracy"]
    print(
        f"\nBest epoch={result.best_epoch}  val_accuracy={best['accuracy']:.4f}  "
        f"val_auc={best['auc']:.4f}  val_macro_f1={best['macro_f1']:.4f}"
    )
    print(f"train_accuracy={train_accuracy:.4f}  training_time_s={training_time_s:.2f}")
    # NFR-5 (T4.10): this wall-clock number is our own hardware's, not
    # matched to the literature table's -- report plainly, don't imply parity.
    wandb.log({"train_accuracy": train_accuracy, "training_time_s": training_time_s})

    config = {
        "m": m,
        "d": d,
        "num_layers": int(train_cfg["num_layers"]),
        "gradient_method": str(train_cfg["gradient_method"]),
        "optimizer": optimizer_name,
        "train_accuracy": train_accuracy,
        "training_time_s": training_time_s,
    }
    save_qgnn_checkpoint(result.best_state_dict, config, str(cfg.qgnn_checkpoint_path))
    print(f"Saved best QGNN checkpoint to {cfg.qgnn_checkpoint_path}")

    artifact = wandb.Artifact(
        name=f"qgnn-jets-m{m}",
        type="model",
        metadata={
            "m": m,
            "d": d,
            "num_layers": int(train_cfg["num_layers"]),
            "optimizer": optimizer_name,
            "best_epoch": result.best_epoch,
            "train_accuracy": train_accuracy,
            "training_time_s": training_time_s,
            **{f"best_val_{key}": val for key, val in best.items() if key != "confusion_matrix"},
        },
    )
    artifact.add_file(str(cfg.qgnn_checkpoint_path))
    wandb.log_artifact(artifact, aliases=["best"])
    wandb.finish()


if __name__ == "__main__":
    main()
