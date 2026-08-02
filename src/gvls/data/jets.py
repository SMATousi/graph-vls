"""Pythia8 quark/gluon jet loading and per-jet k-NN graph construction (T4.1).

Label convention: 0 = quark, 1 = gluon (energyflow's own `qg_jets` convention,
kept as-is rather than remapped).
"""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch_geometric.data import Data

# `Data`-compatible per-jet graph: x (N, NUM_FEATURES), edge_index (2, E), y (1,).
JetGraph: TypeAlias = Data

# Fixed PDG-ID vocabulary for the particle species qg_jets actually contains
# (photon, e+/-, mu+/-, pi+/-, K+/-, K_L, p, pbar, n, nbar) -- 14 species, matching
# what's observed in the dataset -- plus a trailing "unknown" bucket so an
# unexpected pdgid (a system-boundary detail we don't control) degrades
# gracefully instead of raising.
PDGIDS: tuple[int, ...] = (
    22, 11, -11, 13, -13, 211, -211, 321, -321, 130, 2212, -2212, 2112, -2112,
)
NUM_FEATURES = 3 + len(PDGIDS) + 1  # log_pt, y, phi, one_hot(pdgid ∪ unknown)

DEFAULT_K_GRAPH_CAP = 8


def _knn_edge_index(
    y: npt.NDArray[np.floating[Any]], phi: npt.NDArray[np.floating[Any]], k_graph_cap: int
) -> Tensor:
    """Undirected k-NN graph over (y, phi), periodic in phi, union-symmetrized.

    Each node connects to its k_graph nearest neighbors by angular distance
    ΔR = sqrt(Δy² + Δφ²); an edge (i, j) survives if either i lists j or j
    lists i among its nearest neighbors (union, not mutual intersection --
    mutual intersection can empty small/sparse graphs, the same failure mode
    documented for the latent-graph learner in specs/phase1/plan.md).
    """
    n = y.shape[0]
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(k_graph_cap, n - 1)
    dy = y[:, None] - y[None, :]
    dphi = phi[:, None] - phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(dy**2 + dphi**2)
    np.fill_diagonal(dist, np.inf)

    nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # (n, k), unordered

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in nn_idx[i]:
            j = int(j)
            edges.add((i, j) if i < j else (j, i))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    undirected = torch.tensor(sorted(edges), dtype=torch.long).t()  # (2, E)
    return torch.cat([undirected, undirected.flip(0)], dim=1)


def _one_hot_pdgid(pdgid: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    table = {pid: idx for idx, pid in enumerate(PDGIDS)}
    unknown_idx = len(PDGIDS)
    idx = np.array([table.get(int(p), unknown_idx) for p in pdgid], dtype=np.int64)
    onehot = np.zeros((len(pdgid), len(PDGIDS) + 1), dtype=np.float32)
    onehot[np.arange(len(pdgid)), idx] = 1.0
    return onehot


def build_jet_graph(
    particles: npt.NDArray[np.floating[Any]], label: int, k_graph_cap: int = DEFAULT_K_GRAPH_CAP
) -> JetGraph:
    """Build one JetGraph from a jet's raw (n_particles, 4) (pT, y, φ, pdgid) array."""
    pt, y, phi, pdgid = (particles[:, i] for i in range(4))
    edge_index = _knn_edge_index(y, phi, k_graph_cap)

    x = np.concatenate(
        [np.log(pt)[:, None], y[:, None], phi[:, None], _one_hot_pdgid(pdgid)],
        axis=1,
    ).astype(np.float32)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=particles.shape[0],
    )


def load_qg_jets(
    num_jets: int = 20_000,
    k_graph_cap: int = DEFAULT_K_GRAPH_CAP,
    seed: int = 42,
    raw_multiplier: float = 1.3,
    cache_dir: str | None = None,
) -> list[JetGraph]:
    """Load a class-balanced subset of Pythia8 quark/gluon jets as JetGraphs.

    Downloads/reads via `energyflow.qg_jets.load` (cached after first use).
    `num_jets` must be even so an exact 50/50 quark/gluon split is possible.
    """
    import energyflow as ef

    if num_jets <= 0 or num_jets % 2 != 0:
        raise ValueError(f"num_jets must be a positive even number, got {num_jets}")
    per_class = num_jets // 2

    raw_num = min(int(num_jets * raw_multiplier) + 100, 2_000_000)
    load_kwargs: dict[str, object] = {"num_data": raw_num, "pad": False}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    raw_x, raw_y = ef.qg_jets.load(**load_kwargs)
    raw_y = raw_y.astype(np.int64)

    rng = np.random.default_rng(seed)
    quark_idx = np.flatnonzero(raw_y == 0)
    gluon_idx = np.flatnonzero(raw_y == 1)
    if len(quark_idx) < per_class or len(gluon_idx) < per_class:
        raise ValueError(
            f"first {raw_num} loaded jets contain only {len(quark_idx)} quark / "
            f"{len(gluon_idx)} gluon jets, not enough for a balanced {num_jets}-jet "
            "subset -- increase raw_multiplier or num_data"
        )
    selected = np.concatenate(
        [
            rng.choice(quark_idx, size=per_class, replace=False),
            rng.choice(gluon_idx, size=per_class, replace=False),
        ]
    )
    rng.shuffle(selected)

    return [build_jet_graph(raw_x[i], int(raw_y[i]), k_graph_cap) for i in selected]


@dataclass
class JetSplit:
    train: list[JetGraph]
    val: list[JetGraph]
    test: list[JetGraph]


def split_jets(
    graphs: list[JetGraph], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> JetSplit:
    """Deterministic train/val/test split over a list of JetGraphs."""
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1) and val_ratio in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave a test split")

    n = len(graphs)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return JetSplit(
        train=[graphs[i] for i in train_idx],
        val=[graphs[i] for i in val_idx],
        test=[graphs[i] for i in test_idx],
    )


def load_qg_jets_fixed_pool(
    num_jets: int,
    min_particles: int = 1,
    k_graph_cap: int = DEFAULT_K_GRAPH_CAP,
    seed: int = 42,
    raw_multiplier: float = 1.3,
    cache_dir: str | None = None,
) -> list[JetGraph]:
    """Draw `num_jets` jets uniformly at random, with no forced class balance.

    Mirrors the Lorentz-EQGNN literature protocol (arXiv:2411.01641 Section
    IV.A.1): filter to jets with >= `min_particles` particles, then draw a
    uniform random sample of `num_jets` -- unlike `load_qg_jets`'s exact
    50/50 per-class draw, class proportions here are whatever the random
    draw yields (the paper reports 4982/658/583 quark jets across its
    10000/1250/1250 split, not an exact 50/50). `seed` controls the draw
    order, which callers (e.g. `load_qg_jets_lorentz_protocol`) rely on for
    a deterministic positional train/val/test slice.
    """
    import energyflow as ef

    if num_jets <= 0:
        raise ValueError(f"num_jets must be positive, got {num_jets}")

    raw_num = min(int(num_jets * raw_multiplier) + 100, 2_000_000)
    load_kwargs: dict[str, object] = {"num_data": raw_num, "pad": False}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    raw_x, raw_y = ef.qg_jets.load(**load_kwargs)
    raw_y = raw_y.astype(np.int64)

    if min_particles > 1:
        keep = np.array([particles.shape[0] >= min_particles for particles in raw_x])
        raw_x = [particles for particles, k in zip(raw_x, keep) if k]
        raw_y = raw_y[keep]

    if len(raw_x) < num_jets:
        raise ValueError(
            f"filtered pool has only {len(raw_x)} jets (min_particles={min_particles}), "
            f"fewer than the requested {num_jets} -- increase raw_multiplier or num_data"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(raw_x), size=num_jets, replace=False)

    return [build_jet_graph(raw_x[i], int(raw_y[i]), k_graph_cap) for i in selected]


def load_qg_jets_lorentz_protocol(
    num_train: int = 10_000,
    num_val: int = 1_250,
    num_test: int = 1_250,
    min_particles: int = 10,
    k_graph_cap: int = DEFAULT_K_GRAPH_CAP,
    seed: int = 42,
    raw_multiplier: float = 1.3,
    cache_dir: str | None = None,
) -> JetSplit:
    """Reproduce the Lorentz-EQGNN paper's train/val/test draw exactly (T4.10).

    arXiv:2411.01641 Section IV.A.1: "we randomly picked N=12,500 jets,
    allocating the first 10,000 for training, the subsequent 1,250 for
    validation, and the final 1,250 for testing" from jets with >= 10
    particles -- a single random draw of `num_train + num_val + num_test`
    jets, sliced positionally into train/val/test by draw order. This is
    NOT `load_qg_jets` + `split_jets`'s combination (exact 50/50 per-class
    draw, then a second independent random permutation for the split).
    """
    total = num_train + num_val + num_test
    graphs = load_qg_jets_fixed_pool(
        num_jets=total,
        min_particles=min_particles,
        k_graph_cap=k_graph_cap,
        seed=seed,
        raw_multiplier=raw_multiplier,
        cache_dir=cache_dir,
    )
    return JetSplit(
        train=graphs[:num_train],
        val=graphs[num_train : num_train + num_val],
        test=graphs[num_train + num_val :],
    )


def subsample_train(split: JetSplit, n: int, seed: int) -> JetSplit:
    """Randomly subsample `split.train` down to `n` jets; val/test untouched.

    Mirrors Table II's "800 (subset)" row (arXiv:2411.01641): only the
    training set shrinks for the data-scarcity comparison -- validation and
    test stay at their full, fixed size, so test accuracy is measured on the
    same held-out jets regardless of training-set size.
    """
    if n > len(split.train):
        raise ValueError(f"n={n} exceeds available training jets ({len(split.train)})")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(split.train), size=n, replace=False)
    return JetSplit(
        train=[split.train[i] for i in idx],
        val=split.val,
        test=split.test,
    )


def subsample_train_balanced(split: JetSplit, n_per_class: int, seed: int) -> JetSplit:
    """Randomly subsample `split.train` to `n_per_class` jets per label; val/test untouched.

    Used for the T4.10 repeated-training-subset comparison (validation.md
    V-10, user-directed): each of several trials draws a *different*
    class-balanced training subset (e.g. 400 quark + 400 gluon) from the
    same fixed training pool, rather than either (a) reusing one identical
    800-jet subset across all trials (varies only the QGNN's own training
    seed, not the data) or (b) true k-fold CV (would also require
    re-partitioning val/test and likely retraining the frozen GVLS encoder
    per fold). `seed` should differ per trial to get a different draw;
    `split`'s own val/test (and the pool `split.train` is drawn from) should
    come from one fixed outer draw shared by every trial, so only the
    training subset composition varies -- see `load_split_from_config`'s
    `train_subset_seed` vs. `seed` distinction.
    """
    labels = np.array([g.y.item() for g in split.train])
    rng = np.random.default_rng(seed)
    idx_parts = []
    for label in (0, 1):
        label_idx = np.flatnonzero(labels == label)
        if len(label_idx) < n_per_class:
            raise ValueError(
                f"class {label} has only {len(label_idx)} training jets, fewer than "
                f"the requested {n_per_class} per class"
            )
        idx_parts.append(rng.choice(label_idx, size=n_per_class, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return JetSplit(
        train=[split.train[i] for i in idx],
        val=split.val,
        test=split.test,
    )


def load_split_from_config(data_cfg: dict[str, Any]) -> JetSplit:
    """Dispatch to the balanced-draw or Lorentz-protocol jet loader.

    `data_cfg["protocol"]` selects between `"balanced"` (default --
    `load_qg_jets` + `split_jets`, this project's original exact-50/50-draw
    convention) and `"lorentz"` (T4.10, `load_qg_jets_lorentz_protocol` +
    optional `subsample_train`/`subsample_train_balanced`, matching
    arXiv:2411.01641's protocol exactly -- see
    `configs/data/qg_jets_lorentz.yaml`). Centralized here so the three
    jet-pipeline experiment scripts (pretrain/train/evaluate) share one
    dispatch point instead of three copies of this branch.

    Under `"lorentz"`, `data_cfg["seed"]` seeds the outer
    `load_qg_jets_lorentz_protocol` draw (the 10000/1250/1250-jet
    train/val/test partition) -- keep this fixed across a multi-trial repeat
    sweep so every trial shares the same val/test. `train_subset`'s own draw
    is seeded separately by `data_cfg["train_subset_seed"]` (falls back to
    `data_cfg["seed"]` if absent, i.e. the pre-T4.10-repeats default of one
    fixed training subset for every trial) -- vary this per trial instead to
    get a different training subset per trial while val/test stay identical
    (validation.md V-10's repeated-training-subset comparison, as opposed to
    true k-fold CV or a single fixed-subset seed-only repeat).
    `train_subset_balanced` (default `False`) selects `subsample_train_balanced`
    (exact per-class balance, `train_subset` must be even) over
    `subsample_train` (uniform draw, inherits the pool's own class
    proportions).
    """
    protocol = data_cfg.get("protocol", "balanced")
    if protocol == "balanced":
        graphs = load_qg_jets(
            num_jets=int(data_cfg["num_jets"]),
            k_graph_cap=int(data_cfg["k_graph_cap"]),
            seed=int(data_cfg["seed"]),
        )
        return split_jets(
            graphs,
            train_ratio=float(data_cfg["train_ratio"]),
            val_ratio=float(data_cfg["val_ratio"]),
            seed=int(data_cfg["seed"]),
        )
    elif protocol == "lorentz":
        split = load_qg_jets_lorentz_protocol(
            num_train=int(data_cfg["num_train"]),
            num_val=int(data_cfg["num_val"]),
            num_test=int(data_cfg["num_test"]),
            min_particles=int(data_cfg.get("min_particles", 10)),
            k_graph_cap=int(data_cfg["k_graph_cap"]),
            seed=int(data_cfg["seed"]),
        )
        train_subset = data_cfg.get("train_subset")
        if train_subset is not None:
            subset_seed_raw = data_cfg.get("train_subset_seed")
            subset_seed = (
                int(subset_seed_raw) if subset_seed_raw is not None else int(data_cfg["seed"])
            )
            if data_cfg.get("train_subset_balanced", False):
                n_per_class, remainder = divmod(int(train_subset), 2)
                if remainder != 0:
                    raise ValueError(
                        f"train_subset must be even for train_subset_balanced, got {train_subset}"
                    )
                split = subsample_train_balanced(split, n_per_class=n_per_class, seed=subset_seed)
            else:
                split = subsample_train(split, n=int(train_subset), seed=subset_seed)
        return split
    else:
        raise ValueError(f"unknown data protocol: {protocol!r}")
