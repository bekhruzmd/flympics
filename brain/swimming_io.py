"""Swimming observation/action adapters; same form as the Sprint and Hurdles ones.

Deliberately simple and untuned: a starting point for later training, not a policy.
"""
from __future__ import annotations

import numpy as np

from events.sprint import ACCELERATE
from events.swimming import PULL, RECOVER
from .connectome import ConnectomeCircuit
from .event_io import BandDecoder
from .motor import MotorDecoder

# Column order of E, and the `pre` names in models/swimming_encoder_init.csv.
SWIMMING_INPUTS = ("wall", "velocity", "stroke_phase")
# Sprint's (0.90 distance, 0.25 velocity), plus an untuned 0.45 for the arm position.
SWIMMING_INITIAL_WEIGHTS = (0.90, 0.25, 0.45)
# For evolution training: band 0 (difference at or below the threshold) recovers, band 1 pulls,
# the same rule as SwimmingDecoder.
SWIMMING_ACTIONS = (RECOVER, PULL)


class SwimmingEncoder:
    """Learned per-neuron weights on [d_wall / 50, v / 4, stroke_phase], plus a 0.45 tonic drive.

    Wall distance and velocity are virtual-race variables. stroke_phase stands in for
    limb-position (proprioceptive) input; none of them are measured fly neuron signals.
    """
    tonic = 0.45

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray,
                 pool_length: float = 50.0, hand_speed: float = 4.0):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.pool_length = pool_length
        self.hand_speed = hand_speed
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape != (len(self.indices), len(SWIMMING_INPUTS)):
            raise ValueError(f"Expected encoder shape {(len(self.indices), len(SWIMMING_INPUTS))}, "
                             f"got {self.weights.shape}")

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.clip([
            observation["distance_to_wall"] / self.pool_length,
            observation["velocity"] / self.hand_speed,
            observation["stroke_phase"],
        ], 0.0, 1.0)

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        return self.encode_features(self.features(observation))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Stimulus from already-normalized features; training can call this directly."""
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = self.tonic + self.weights @ features
        return vector


class SwimmingDecoder(MotorDecoder):
    """Sprint's extensor - flexor > threshold readout, read as PULL (power stroke) or RECOVER."""

    def decode(self, activity: np.ndarray) -> tuple[str, dict[str, float]]:
        action, motor = super().decode(activity)
        return (PULL if action == ACCELERATE else RECOVER), motor


class SwimmingBandDecoder(BandDecoder):
    """SwimmingDecoder's rule as a one-threshold band decoder, for trained policies."""

    def __init__(self, circuit: ConnectomeCircuit, thresholds):
        super().__init__(circuit, SWIMMING_ACTIONS, thresholds)


def initial_swimming_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets SWIMMING_INITIAL_WEIGHTS."""
    return np.tile(SWIMMING_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
