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
