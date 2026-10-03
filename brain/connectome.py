"""Loading a compact *real-data-derived* MANC circuit, not a whole-brain model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from scipy import sparse

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "manc_front_leg_v1"


@dataclass(frozen=True)
class ConnectomeCircuit:
    neuron_ids: tuple[str, ...]
    roles: tuple[str, ...]
    cell_types: tuple[str, ...]
    weights: sparse.csr_matrix  # [post, pre], normalized synapse-count coupling

    @property
    def size(self) -> int:
        return len(self.neuron_ids)

    def indices(self, role: str) -> np.ndarray:
        return np.asarray([i for i, value in enumerate(self.roles) if value == role], dtype=int)


def load_manc_front_leg_circuit(data_dir: Path = DATA_DIR) -> ConnectomeCircuit:
    """Load the vendored compact subgraph extracted from public MANC v1.2.1."""
    neurons_path, edges_path = data_dir / "neurons.csv", data_dir / "edges.csv"
    if not neurons_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing circuit files in {data_dir}. See data/README.md.")
    with neurons_path.open(newline="") as handle:
        neurons = list(csv.DictReader(handle))
    index = {row["manc_121_id"]: i for i, row in enumerate(neurons)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    with edges_path.open(newline="") as handle:
        for edge in csv.DictReader(handle):
            rows.append(index[edge["post"]])
            cols.append(index[edge["pre"]])
            values.append(float(edge["weight"]))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(neurons), len(neurons))).tocsr()
    return ConnectomeCircuit(
        tuple(row["manc_121_id"] for row in neurons),
        tuple(row["role"] for row in neurons),
        tuple(row["cell_type"] for row in neurons),
        matrix,
    )
