"""Modeled rate dynamics; MANC supplies connectivity, not these dynamics."""
from __future__ import annotations

import numpy as np

from .connectome import ConnectomeCircuit


class RateNetwork:
    """Sparse leaky rate network with bounded nonlinearity.

    MANC synapse counts are structurally real. The time constant, gain, input scale,
    nonlinearity, and unsigned coupling are explicit engineering assumptions.

    noise, if positive, adds independent Gaussian noise with that standard deviation to
    every neuron's drive (before the tanh) on every substep. It is off by default, and with
    noise == 0 no random numbers are drawn, so results are bit-for-bit the noiseless ones.
    reset(seed) reseeds the noise, so a seeded run is reproducible.
    """
    def __init__(self, circuit: ConnectomeCircuit, leak: float = 0.35, gain: float = 3.0,
                 noise: float = 0.0, seed: int | None = None):
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("noise must be a finite standard deviation >= 0.")
        self.circuit, self.leak, self.gain, self.noise = circuit, leak, gain, float(noise)
        self.state = np.zeros(circuit.size, dtype=float)
        self.silenced = np.asarray(circuit.ablated, dtype=int)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.state.fill(0.0)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def step(self, stimulus: np.ndarray, substeps: int = 2) -> np.ndarray:
        if stimulus.shape != self.state.shape:
            raise ValueError(f"Expected stimulus shape {self.state.shape}, got {stimulus.shape}")
        for _ in range(substeps):
            drive = self.gain * (self.circuit.weights @ self.state) + stimulus
            if self.noise:
                drive = drive + self.rng.normal(0.0, self.noise, drive.shape)
            target = np.tanh(drive)
            self.state += self.leak * (target - self.state)
            # Ablated neurons stay silent even when the encoder stimulates them.
            self.state[self.silenced] = 0.0
        return self.state.copy()
