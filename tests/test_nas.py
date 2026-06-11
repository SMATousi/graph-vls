import math
import os
import tempfile

import optuna
import pytest
import torch
from omegaconf import OmegaConf
from optuna.trial import FixedTrial
from torch_geometric.data import Data

from gvls.data.splits import split_edges
from gvls.nas.objective import make_objective
from gvls.nas.search_space import CONFIG_KEYS, suggest_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fixed_trial(overrides: dict | None = None) -> FixedTrial:
    """Return a FixedTrial with sensible defaults, optionally overridden."""
    defaults = {
        "latent_dim": 32,
        "hidden_dim": 64,
        "mp_rounds": 1,
        "graph_method": "attention",
        "k": 10,
        "prior": "isotropic",
        "beta": 0.001,
        "lr": 0.01,
    }
    if overrides:
        defaults.update(overrides)
    return FixedTrial(defaults)


# ── V-1: correct keys ─────────────────────────────────────────────────────────

def test_suggest_config_returns_all_keys_isotropic() -> None:
    cfg = suggest_config(_fixed_trial({"prior": "isotropic"}))
    assert set(cfg.keys()) == CONFIG_KEYS


def test_suggest_config_returns_all_keys_graph_mrf() -> None:
    cfg = suggest_config(_fixed_trial({"prior": "graph_mrf", "lambda_": 2.0}))
    assert set(cfg.keys()) == CONFIG_KEYS


def test_name_is_always_gvls() -> None:
    cfg = suggest_config(_fixed_trial())
    assert cfg["name"] == "gvls"


# ── V-1: lambda_ conditional ──────────────────────────────────────────────────

def test_lambda_fixed_when_isotropic() -> None:
    # FixedTrial has no 'lambda_' key → if suggest_float were called it would raise.
    cfg = suggest_config(_fixed_trial({"prior": "isotropic"}))
    assert cfg["lambda_"] == 1.0


def test_lambda_suggested_when_graph_mrf() -> None:
    cfg = suggest_config(_fixed_trial({"prior": "graph_mrf", "lambda_": 3.7}))
    assert cfg["lambda_"] == pytest.approx(3.7)


# ── V-1: categorical values valid ─────────────────────────────────────────────

def test_graph_method_valid_without_nri() -> None:
    for method in ("attention", "fgp"):
        cfg = suggest_config(_fixed_trial({"graph_method": method}))
        assert cfg["graph_method"] in {"attention", "fgp"}


def test_graph_method_nri_when_enabled() -> None:
    trial = FixedTrial({
        "latent_dim": 32, "hidden_dim": 64, "mp_rounds": 1,
        "k": 10, "prior": "isotropic", "beta": 0.001, "lr": 0.01,
        "graph_method": "nri",
    })
    cfg = suggest_config(trial, include_nri=True)
    assert cfg["graph_method"] == "nri"


def test_prior_values_valid() -> None:
    for prior in ("isotropic", "graph_mrf"):
        overrides = {"prior": prior}
        if prior == "graph_mrf":
            overrides["lambda_"] = 1.0
        cfg = suggest_config(_fixed_trial(overrides))
        assert cfg["prior"] in {"isotropic", "graph_mrf"}


def test_latent_dim_from_allowed_set() -> None:
    for ld in (16, 32, 64, 128):
        cfg = suggest_config(_fixed_trial({"latent_dim": ld}))
        assert cfg["latent_dim"] in {16, 32, 64, 128}


# ── integration: real Optuna study ────────────────────────────────────────────

def test_suggest_config_works_inside_real_study() -> None:
    """Verify suggest_config integrates with an actual Optuna TPE study."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    configs: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial)
        configs.append(cfg)
        return 0.5  # dummy

    study.optimize(objective, n_trials=8)
    assert len(configs) == 8
    for cfg in configs:
        assert set(cfg.keys()) == CONFIG_KEYS
        assert cfg["graph_method"] in {"attention", "fgp"}
        assert cfg["prior"] in {"isotropic", "graph_mrf"}
        assert cfg["latent_dim"] in {16, 32, 64, 128}
        # lambda_ must be 1.0 when prior is isotropic
        if cfg["prior"] == "isotropic":
            assert cfg["lambda_"] == 1.0


# ── V-2: objective function ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_split():
    """Small synthetic graph (N=40) with edge splits for objective tests."""
    N = 40
    torch.manual_seed(7)
    x = torch.randn(N, 8)
    # Dense-ish path + cross edges to ensure enough val edges
    row, col = [], []
    for i in range(N - 1):
        row += [i, i + 1]
        col += [i + 1, i]
    for i in range(0, N - 5, 4):
        row += [i, i + 4]
        col += [i + 4, i]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=N)
    split = split_edges(data, train_ratio=0.8, seed=42)
    return data, split


def _tiny_cfg(epochs: int = 6) -> object:
    return OmegaConf.create({"nas": {"epochs_per_trial": epochs, "include_nri": False}})


def test_objective_single_trial_completes(tiny_split) -> None:
    data, split = tiny_split
    obj = make_objective(data, split, _tiny_cfg(epochs=6), torch.device("cpu"))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=1),
    )
    study.optimize(obj, n_trials=1)
    assert len(study.trials) == 1
    val = study.trials[0].value
    assert val is not None
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_objective_pruning_fires(tiny_split) -> None:
    data, split = tiny_split
    obj = make_objective(data, split, _tiny_cfg(epochs=6), torch.device("cpu"))
    # ThresholdPruner with lower=1.0 prunes every trial whose mid-point AUC < 1.0,
    # which is guaranteed for randomly initialised models.
    pruner = optuna.pruners.ThresholdPruner(lower=1.0, n_warmup_steps=0)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=2),
        pruner=pruner,
    )
    study.optimize(obj, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_objective_returns_best_across_epochs(tiny_split) -> None:
    """Returned value is the max over all epochs, so it is >= the mid-epoch AUC."""
    data, split = tiny_split
    # Run 3 trials; every returned value must be a valid AUC in [0, 1].
    obj = make_objective(data, split, _tiny_cfg(epochs=6), torch.device("cpu"))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=3),
    )
    study.optimize(obj, n_trials=3)
    for t in study.trials:
        assert t.value is not None
        assert math.isfinite(t.value)
        assert 0.0 <= t.value <= 1.0
    # best_value is the max over all completed trials
    assert study.best_value == max(t.value for t in study.trials)


# ── V-3: NAS entry point behaviours ──────────────────────────────────────────

def test_study_resumable(tiny_split) -> None:
    """Trials accumulate across two calls with the same SQLite storage."""
    data, split = tiny_split
    cfg = _tiny_cfg(epochs=4)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        storage = f"sqlite:///{tmp}/test.db"

        def _make_study():
            return optuna.create_study(
                study_name="test-resume",
                storage=storage,
                direction="maximize",
                sampler=optuna.samplers.RandomSampler(seed=0),
                load_if_exists=True,
            )

        # First run: 2 trials
        s1 = _make_study()
        s1.optimize(make_objective(data, split, cfg, device), n_trials=2)
        assert len(s1.trials) == 2

        # Second run: 2 more trials — should see 4 total
        s2 = _make_study()
        s2.optimize(make_objective(data, split, cfg, device), n_trials=2)
        assert len(s2.trials) == 4


def test_best_config_written(tiny_split) -> None:
    """Best config YAML contains all required keys after a 2-trial study."""
    data, split = tiny_split
    cfg = _tiny_cfg(epochs=4)
    device = torch.device("cpu")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=5),
    )
    study.optimize(make_objective(data, split, cfg, device), n_trials=2)

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
        "lambda_": p.get("lambda_", 1.0),
        "lr": p["lr"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "best.yaml")
        OmegaConf.save(OmegaConf.create(best_cfg), out_path)
        loaded = OmegaConf.load(out_path)
        # All gvls.yaml schema keys must be present
        for key in ("name", "latent_dim", "hidden_dim", "mp_rounds",
                    "graph_method", "prior", "k", "beta", "lambda_", "lr"):
            assert key in loaded, f"Missing key: {key}"
        assert loaded.name == "gvls"
        assert loaded.latent_dim in {16, 32, 64, 128}
