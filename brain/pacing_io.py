"""400 m Pacing observation/action adapters.

Deliberately simple and untuned: a starting point for calibration/training, not a final policy.
"""
from __future__ import annotations

import numpy as np

from events.pacing import ACCELERATE, COAST
from .connectome import ConnectomeCircuit
from .event_io import BandDecoder, FeatureEncoder

# Column order of E, and the `pre` names in models/pacing_encoder_init.csv.
PACING_INPUTS = ("distance", "velocity", "energy")
# Sprint's (0.90 distance, 0.25 velocity), plus 0.45 for energy budget.
PACING_INITIAL_WEIGHTS = (0.90, 0.25, 0.45)
PACING_ACTIONS = (COAST, ACCELERATE)
PACING_INITIAL_THRESHOLDS = (0.015,)


class PacingEncoder(FeatureEncoder):
    """Learned per-neuron weights on [d / 400, v / 8, energy / 100], plus a 0.45 tonic drive.

    Distance to finish, velocity, and energy budget are virtual-race variables,
    not measurements known to biological fly leg neurons.
    """
    INPUTS = PACING_INPUTS

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray,
                 distance: float = 400.0, max_velocity: float = 8.0,
                 energy_budget: float = 100.0):
        super().__init__(circuit, weights)
        self.distance = distance
        self.max_velocity = max_velocity
        self.energy_budget = energy_budget

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.clip([
            observation["distance_to_finish"] / self.distance,
            observation["velocity"] / self.max_velocity,
            observation["energy"] / self.energy_budget,
        ], 0.0, 1.0)


class PacingDecoder(BandDecoder):
    """Up to the threshold: COAST; above it: ACCELERATE."""
    def __init__(self, circuit: ConnectomeCircuit, thresholds=PACING_INITIAL_THRESHOLDS):
        super().__init__(circuit, PACING_ACTIONS, thresholds)


def initial_pacing_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets PACING_INITIAL_WEIGHTS."""
    return np.tile(PACING_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
