"""Shared encoder/decoder pieces for events added after Swimming.

Same form as the Sprint, Hurdles and Swimming adapters: a tonic drive plus learned
per-neuron weights on normalized observation features, and the extensor - flexor
readout split into bands.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .connectome import ConnectomeCircuit
from .motor import MotorDecoder


class FeatureEncoder:
    """stimulus[sensory] = tonic + E @ features(observation); every other neuron gets 0.

    Subclasses set INPUTS (column names of E) and implement features().
    """
    INPUTS: tuple[str, ...] = ()
    tonic = 0.45

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.weights = np.array(weights, dtype=float)
        expected = (len(self.indices), len(self.INPUTS))
        if self.weights.shape != expected:
            raise ValueError(f"Expected encoder shape {expected}, got {self.weights.shape}")

    def features(self, observation: dict[str, float]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        return self.encode_features(self.features(observation))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Stimulus from already-normalized features; training can call this directly."""
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = self.tonic + self.weights @ features
        return vector


class BandDecoder(MotorDecoder):
    """extensor - flexor, split into len(actions) bands by sorted thresholds.

    actions are listed in order of increasing difference: the difference falls in band i
    when thresholds[i-1] < difference <= thresholds[i]. Thresholds may be negative.
    """
    def __init__(self, circuit: ConnectomeCircuit, actions: Sequence[str], thresholds: Sequence[float]):
        thresholds = [float(t) for t in thresholds]
        if len(thresholds) != len(actions) - 1:
            raise ValueError(f"{len(actions)} actions need {len(actions) - 1} thresholds.")
        if any(not math.isfinite(t) or not -1 <= t <= 1 for t in thresholds):
            raise ValueError("Thresholds must be finite and between -1 and 1.")
        if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("Thresholds must be strictly increasing.")
        super().__init__(circuit, threshold=thresholds[0])
        self.actions = tuple(actions)
        self.thresholds = tuple(thresholds)

    def band(self, difference: float) -> int:
        return int(np.searchsorted(self.thresholds, difference, side="left"))

    def decode(self, activity: np.ndarray) -> tuple[str, dict[str, float]]:
        _, motor = super().decode(activity)
        return self.actions[self.band(motor["difference"])], motor
