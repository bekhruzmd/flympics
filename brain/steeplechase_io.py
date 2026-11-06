"""Steeplechase observation/action adapters.

The untrained values are deliberately simple placeholders; training replaces them.
"""
from __future__ import annotations

import numpy as np

from events.steeplechase import ACCELERATE, COAST, JUMP
from .connectome import ConnectomeCircuit
from .event_io import BandDecoder, FeatureEncoder

# Column order of E, and the `pre` names in the steeplechase encoder CSVs.
STEEPLECHASE_INPUTS = ("barrier", "water", "velocity", "stamina", "finish")
STEEPLECHASE_INITIAL_WEIGHTS = (0.90, 0.45, 0.25, 0.45, 0.45)
# Bands of extensor - flexor, lowest first; the same order as the Hurdles decoder.
STEEPLECHASE_ACTIONS = (COAST, ACCELERATE, JUMP)
STEEPLECHASE_INITIAL_THRESHOLDS = (0.015, 0.30)


class SteeplechaseEncoder(FeatureEncoder):
    """Learned per-neuron weights on [d_barrier / 40, water, v / 8, stamina, d_finish / 600].

    All features are clipped to [0, 1], plus a 0.45 tonic drive. They are virtual-race
    variables, not signals measured in a fly.
    """
    INPUTS = STEEPLECHASE_INPUTS

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray, sensor_range: float = 40.0,
                 max_velocity: float = 8.0, distance: float = 600.0):
        super().__init__(circuit, weights)
        self.sensor_range, self.max_velocity, self.distance = sensor_range, max_velocity, distance

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.clip([
            observation["distance_to_barrier"] / self.sensor_range,
            observation["water_next"],
            observation["velocity"] / self.max_velocity,
            observation["stamina"],
            observation["distance_to_finish"] / self.distance,
        ], 0.0, 1.0)


class SteeplechaseDecoder(BandDecoder):
    """Up to the first threshold: COAST; up to the second: ACCELERATE; above: JUMP."""
    def __init__(self, circuit: ConnectomeCircuit, thresholds=STEEPLECHASE_INITIAL_THRESHOLDS):
        super().__init__(circuit, STEEPLECHASE_ACTIONS, thresholds)


def initial_steeplechase_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets STEEPLECHASE_INITIAL_WEIGHTS."""
    return np.tile(STEEPLECHASE_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
