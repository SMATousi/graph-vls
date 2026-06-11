from __future__ import annotations

import optuna

_LATENT_DIMS = [16, 32, 64, 128]
_HIDDEN_DIMS = [32, 64, 128, 256]
_K_VALUES = [5, 10, 20, 50]
_GRAPH_METHODS_BASE = ["attention", "fgp"]
_PRIORS = ["isotropic", "graph_mrf"]

# Keys present in every config returned by suggest_config.
CONFIG_KEYS = frozenset(
    {"name", "latent_dim", "hidden_dim", "mp_rounds", "graph_method",
     "prior", "k", "beta", "lambda_", "lr"}
)


def suggest_config(trial: optuna.Trial, include_nri: bool = False) -> dict:
    """Suggest a GVLS config for one Optuna trial.

    Returns a flat dict with the same schema as configs/model/gvls.yaml plus
    the training parameter 'lr'.  All values are drawn from trial.suggest_*
    except 'name' (always 'gvls') and 'lambda_' when prior='isotropic' (fixed 1.0).

    include_nri: if True, adds 'nri' to graph_method choices.  Must be consistent
    across all trials in the same study — changing it mid-study breaks the TPE
    sampler's internal model.
    """
    latent_dim: int = trial.suggest_categorical("latent_dim", _LATENT_DIMS)  # type: ignore[assignment]
    hidden_dim: int = trial.suggest_categorical("hidden_dim", _HIDDEN_DIMS)  # type: ignore[assignment]
    mp_rounds: int = trial.suggest_int("mp_rounds", 0, 2)
    k: int = trial.suggest_categorical("k", _K_VALUES)  # type: ignore[assignment]

    methods = _GRAPH_METHODS_BASE + (["nri"] if include_nri else [])
    graph_method: str = trial.suggest_categorical("graph_method", methods)  # type: ignore[assignment]

    prior: str = trial.suggest_categorical("prior", _PRIORS)  # type: ignore[assignment]
    beta: float = trial.suggest_float("beta", 1e-5, 0.1, log=True)
    lr: float = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

    # lambda_ is only a meaningful hyperparameter under the graph_mrf prior;
    # fix it to 1.0 for isotropic so the TPE sampler doesn't waste budget on it.
    lambda_: float = (
        trial.suggest_float("lambda_", 0.1, 10.0, log=True)
        if prior == "graph_mrf"
        else 1.0
    )

    return {
        "name": "gvls",
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "mp_rounds": mp_rounds,
        "graph_method": graph_method,
        "prior": prior,
        "k": k,
        "beta": beta,
        "lambda_": lambda_,
        "lr": lr,
    }
