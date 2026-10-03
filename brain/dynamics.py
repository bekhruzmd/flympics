"""Modeled rate dynamics; MANC supplies connectivity, not these dynamics."""
from __future__ import annotations

import numpy as np

from .connectome import ConnectomeCircuit


class RateNetwork:
    """Sparse leaky rate network with bounded nonlinearity.

    MANC synapse counts are structurally real. The time constant, gain, input scale,
    nonlinearity, and unsigned coupling are explicit engineering assumptions.
    """
    def __init__(self, circuit: ConnectomeCircuit, leak: float = 0.35, gain: float = 3.0):
        self.circuit, self.leak, self.gain = circuit, leak, gain
        self.state = np.zeros(circuit.size, dtype=float)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, stimulus: np.ndarray, substeps: int = 2) -> np.ndarray:
        if stimulus.shape != self.state.shape:
            raise ValueError(f"Expected stimulus shape {self.state.shape}, got {stimulus.shape}")
        for _ in range(substeps):
            drive = self.gain * (self.circuit.weights @ self.state) + stimulus
            target = np.tanh(drive)
            self.state += self.leak * (target - self.state)
        return self.state.copy()
