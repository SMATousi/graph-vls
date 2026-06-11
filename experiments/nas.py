"""NAS entry point: run hyperparameter search on a Planetoid dataset.

Usage:
    # 50-trial search on Cora (default):
    python experiments/nas.py

    # Different dataset:
    python experiments/nas.py data=citeseer

    # Fewer trials for a quick smoke-test:
    python experiments/nas.py nas.n_trials=5 nas.epochs_per_trial=20
"""

import os

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from gvls.data import load_planetoid, split_edges
from gvls.nas.objective import make_objective

optuna.logging.set_verbosity(optuna.logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="nas_config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg.data.name.lower()

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading {cfg.data.name}...")
    data = load_planetoid(cfg.data.name)
    split = split_edges(data, train_ratio=cfg.train.split_ratio, seed=cfg.train.seed)
    N = split.n_nodes
    print(f"  nodes={N}  train_edges={split.train_edge_index.size(1)//2}"
          f"  val={split.val_pos.size(1)}  test={split.test_pos.size(1)}")

    # ── study ─────────────────────────────────────────────────────────────────
    os.makedirs(cfg.nas.study_dir, exist_ok=True)
    db_path = os.path.join(cfg.nas.study_dir, f"{dataset_name}.db")

    study = optuna.create_study(
        study_name=f"gvls-{dataset_name}",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.train.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        load_if_exists=True,
    )

    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nStudy: gvls-{dataset_name} ({already_done} trials already complete)")
    print(f"Running up to {cfg.nas.n_trials} more trials "
          f"({cfg.nas.epochs_per_trial} epochs each, timeout={cfg.nas.timeout}s)...\n")

    # ── optimize ──────────────────────────────────────────────────────────────
    objective = make_objective(data, split, cfg, device)
    study.optimize(
        objective,
        n_trials=cfg.nas.n_trials,
        timeout=cfg.nas.timeout,
        show_progress_bar=False,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n{'='*68}")
    print(f"  NAS complete — {len(completed)} completed, "
          f"{len(pruned)} pruned, {len(failed)} failed")
    print(f"{'='*68}")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]  # type: ignore[arg-type]
    header = (
        f"{'Rank':<6}{'Trial':<8}{'Val AUC':<10}"
        f"{'Method':<12}{'Prior':<14}{'ld':<6}{'hd':<6}{'beta'}"
    )
    print(header)
    print("-" * len(header))
    for rank, t in enumerate(top5, 1):
        p = t.params
        print(
            f"{rank:<6}{t.number:<8}{t.value:<10.4f}"
            f"{p.get('graph_method','?'):<12}{p.get('prior','?'):<14}"
            f"{p.get('latent_dim','?'):<6}{p.get('hidden_dim','?'):<6}"
            f"{p.get('beta', 0.0):.2e}"
        )

    if not completed:
        print("\nNo completed trials — nothing to save.")
        return

    # ── save best config ──────────────────────────────────────────────────────
    best_trial = study.best_trial
    p = best_trial.params
    best_cfg = {
        "name": "gvls",
        "latent_dim": p["latent_dim"],
        "hidden_dim": p["hidden_dim"],
        "mp_rounds": p["mp_rounds"],
        "graph_method": p["graph_method"],
        "prior": p["prior"],
        "k": p["k"],
        "beta": p["beta"],
        "lambda_": p.get("lambda_", 1.0),  # fixed to 1.0 when prior=isotropic
        "lr": p["lr"],
    }

    os.makedirs("configs/best", exist_ok=True)
    out_path = f"configs/best/{dataset_name}.yaml"
    OmegaConf.save(OmegaConf.create(best_cfg), out_path)
    print(f"\nBest config (trial #{best_trial.number}, val_auc={study.best_value:.4f}) "
          f"saved to {out_path}")

    # ── W&B summary run ───────────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        name=f"nas-{dataset_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    log_payload: dict = {
        "best_val_auc": study.best_value,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
    }
    for k, v in best_cfg.items():
        if k != "name":
            log_payload[f"best/{k}"] = v
    wandb.log(log_payload)
    wandb.finish()


if __name__ == "__main__":
    main()
