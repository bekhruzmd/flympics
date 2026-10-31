"""Experiment-only lesions of the compact MANC circuit.

Ablation is a modeling intervention, not a biological measurement. A removed
neuron loses every incoming and outgoing edge, and its modeled rate is clamped
to 0, including any artificial sensory stimulation. The vendored circuit is
never changed, and surviving weights are not renormalized.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
from scipy import sparse

from .connectome import ConnectomeCircuit


def ablate(circuit: ConnectomeCircuit, neurons: Iterable[int] = (),
           edges: Iterable[tuple[int, int]] = ()) -> ConnectomeCircuit:
    """Return a lesioned copy. `neurons` are indices; `edges` are (pre, post) index pairs."""
    removed = sorted({int(i) for i in neurons} | set(circuit.ablated))
    if any(not 0 <= i < circuit.size for i in removed):
        raise ValueError(f"Neuron indices must be between 0 and {circuit.size - 1}.")
    cut = {(int(post), int(pre)) for pre, post in edges}
    for post, pre in cut:
        if not (0 <= pre < circuit.size and 0 <= post < circuit.size) or circuit.weights[post, pre] == 0:
            raise ValueError(f"No edge {pre} -> {post} in the circuit.")
    coo = circuit.weights.tocoo()
    lost = np.isin(coo.row, removed) | np.isin(coo.col, removed)
    lost |= np.fromiter(((r, c) in cut for r, c in zip(coo.row, coo.col)), dtype=bool, count=coo.nnz)
    weights = sparse.csr_matrix((coo.data[~lost], (coo.row[~lost], coo.col[~lost])), shape=coo.shape)
    return replace(circuit, weights=weights, ablated=tuple(removed))


def top_drivers(circuit: ConnectomeCircuit, target_role: str, k: int,
                source_role: str = "intrinsic") -> np.ndarray:
    """Indices of the k `source_role` neurons with most summed weight into `target_role`."""
    sources, targets = circuit.indices(source_role), circuit.indices(target_role)
    drive = np.asarray(circuit.weights[targets][:, sources].sum(axis=0)).ravel()
    order = np.argsort(-drive, kind="stable")[:k]
    return sources[order]
