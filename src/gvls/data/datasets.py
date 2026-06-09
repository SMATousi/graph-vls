from pathlib import Path

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, TUDataset

PLANETOID_NAMES = ("Cora", "CiteSeer", "PubMed")
TU_NAMES = ("MUTAG", "PROTEINS", "IMDB-B")

_ROOT = Path(__file__).parent.parent.parent.parent / "data"


def load_planetoid(name: str, root: str | None = None) -> Data:
    if name not in PLANETOID_NAMES:
        raise ValueError(f"name must be one of {PLANETOID_NAMES}, got {name!r}")
    root_path = str(root or _ROOT)
    dataset = Planetoid(root=root_path, name=name)
    return dataset[0]


def load_tu_dataset(name: str, root: str | None = None) -> Dataset:
    if name not in TU_NAMES:
        raise ValueError(f"name must be one of {TU_NAMES}, got {name!r}")
    root_path = str(root or _ROOT)
    return TUDataset(root=root_path, name=name)
