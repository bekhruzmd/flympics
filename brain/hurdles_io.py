"""Hurdles observation/action adapters; same form as the Sprint encoder and decoder.

Deliberately simple and untuned: a starting point for later training, not a policy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from events.hurdles import ACCELERATE, COAST, JUMP
from .connectome import ConnectomeCircuit
from .event_io import BandDecoder, FeatureEncoder

# Column order of E, and the `pre` names in models/hurdles_encoder_init.csv.
HURDLES_INPUTS = ("next_hurdle", "second_hurdle", "velocity")
# Sprint's (0.90 distance, 0.25 velocity), with the distance weight split across both hurdles.
HURDLES_INITIAL_WEIGHTS = (0.90, 0.45, 0.25)
HURDLES_ACTIONS = (COAST, ACCELERATE, JUMP)
HURDLES_INITIAL_THRESHOLDS = (0.015, 0.30)


class HurdlesEncoder(FeatureEncoder):
    """Learned per-neuron weights on [d1 / 40, d2 / 40, v / 8], plus Sprint's 0.45 tonic drive.

    Hurdle distances and velocity are virtual-race variables, not measurements known to a
    fly neuron.
    """
    INPUTS = HURDLES_INPUTS

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray,
                 sensor_range: float = 40.0, max_velocity: float = 8.0):
        super().__init__(circuit, weights)
        self.sensor_range = sensor_range
        self.max_velocity = max_velocity

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.clip([
            observation["distance_to_next_hurdle"] / self.sensor_range,
            observation["distance_to_second_hurdle"] / self.sensor_range,
            observation["velocity"] / self.max_velocity,
        ], 0.0, 1.0)


class HurdlesDecoder(BandDecoder):
    """Sprint's extensor - flexor readout, split into three bands by two thresholds.

    Up to coast_threshold: COAST; up to jump_threshold: ACCELERATE; above it: JUMP. The
    order follows increasing extensor drive, and the lower band keeps Sprint's meaning of
    "above the threshold means push". Both thresholds are placeholders for training.
    """
    def __init__(self, circuit: ConnectomeCircuit,
                 coast_threshold: float | Sequence[float] = 0.015,
                 jump_threshold: float | None = None):
        if jump_threshold is None and isinstance(coast_threshold, (tuple, list, np.ndarray)):
            thresholds = tuple(float(t) for t in coast_threshold)
        else:
            if jump_threshold is None:
                jump_threshold = 0.30
            thresholds = (float(coast_threshold), float(jump_threshold))
        if thresholds[0] >= thresholds[1]:
            raise ValueError("coast_threshold must be below jump_threshold.")
        super().__init__(circuit, HURDLES_ACTIONS, thresholds)

    @property
    def coast_threshold(self) -> float:
        return self.thresholds[0]

    @property
    def jump_threshold(self) -> float:
        return self.thresholds[1]


def initial_hurdles_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets HURDLES_INITIAL_WEIGHTS."""
    return np.tile(HURDLES_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
