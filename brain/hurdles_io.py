"""Hurdles observation/action adapters; same form as the Sprint encoder and decoder.

Deliberately simple and untuned: a starting point for later training, not a policy.
"""
from __future__ import annotations

import numpy as np

from events.hurdles import ACCELERATE, COAST, JUMP
from .connectome import ConnectomeCircuit
from .motor import MotorDecoder

# Column order of E, and the `pre` names in models/hurdles_encoder_init.csv.
HURDLES_INPUTS = ("next_hurdle", "second_hurdle", "velocity")
# Sprint's (0.90 distance, 0.25 velocity), with the distance weight split across both hurdles.
HURDLES_INITIAL_WEIGHTS = (0.90, 0.45, 0.25)


class HurdlesEncoder:
    """Learned per-neuron weights on [d1 / 40, d2 / 40, v / 8], plus Sprint's 0.45 tonic drive.

    Hurdle distances and velocity are virtual-race variables, not measurements known to a
    fly neuron.
    """
    tonic = 0.45

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray,
                 sensor_range: float = 40.0, max_velocity: float = 8.0):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.sensor_range = sensor_range
        self.max_velocity = max_velocity
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape != (len(self.indices), len(HURDLES_INPUTS)):
            raise ValueError(f"Expected encoder shape {(len(self.indices), len(HURDLES_INPUTS))}, "
                             f"got {self.weights.shape}")

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.clip([
            observation["distance_to_next_hurdle"] / self.sensor_range,
            observation["distance_to_second_hurdle"] / self.sensor_range,
            observation["velocity"] / self.max_velocity,
        ], 0.0, 1.0)

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        return self.encode_features(self.features(observation))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Stimulus from already-normalized features; training can call this directly."""
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = self.tonic + self.weights @ features
        return vector


class HurdlesDecoder(MotorDecoder):
    """Sprint's extensor - flexor readout, split into three bands by two thresholds.

    Up to coast_threshold: COAST; up to jump_threshold: ACCELERATE; above it: JUMP. The
    order follows increasing extensor drive, and the lower band keeps Sprint's meaning of
    "above the threshold means push". Both thresholds are placeholders for training.
    """
    def __init__(self, circuit: ConnectomeCircuit, coast_threshold: float = 0.015,
                 jump_threshold: float = 0.30):
        if not coast_threshold < jump_threshold:
            raise ValueError("coast_threshold must be below jump_threshold.")
        super().__init__(circuit, threshold=coast_threshold)
        self.jump_threshold = jump_threshold

    @property
    def coast_threshold(self) -> float:
        return self.threshold

    def decode(self, activity: np.ndarray) -> tuple[str, dict[str, float]]:
        _, motor = super().decode(activity)
        difference = motor["difference"]
        if difference > self.jump_threshold:
            return JUMP, motor
        return (ACCELERATE if difference > self.coast_threshold else COAST), motor


def initial_hurdles_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets HURDLES_INITIAL_WEIGHTS."""
    return np.tile(HURDLES_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
