"""Artificial task-observation to biological-population stimulation adapter."""
from __future__ import annotations

import numpy as np

from .connectome import ConnectomeCircuit


class SensoryEncoder:
    """Artificially stimulates selected front-leg proprioceptor nodes.

    Distance and velocity are virtual-sprint variables, not measurements known to a
    fly neuron. This component is intentionally replaceable.
    """
    def __init__(self, circuit: ConnectomeCircuit, sprint_distance: float = 100.0):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.sprint_distance = sprint_distance

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        distance_fraction = np.clip(observation["distance_to_finish"] / self.sprint_distance, 0.0, 1.0)
        velocity_fraction = np.clip(observation["velocity"] / 8.0, 0.0, 1.0)
        # Artificial tonic/proprioceptive-like drive, deliberately not a biological claim.
        level = 0.45 + 0.90 * distance_fraction + 0.25 * velocity_fraction
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = level
        return vector
