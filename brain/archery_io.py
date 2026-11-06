"""Archery observation/action adapters.

The untrained values are deliberately simple placeholders; training replaces them.
"""
from __future__ import annotations

import numpy as np

from events.archery import LEFT, RELEASE, RIGHT
from .connectome import ConnectomeCircuit
from .event_io import BandDecoder, FeatureEncoder

# Column order of E, and the `pre` names in the archery encoder CSVs.
ARCHERY_INPUTS = ("aim", "wind", "draw")
ARCHERY_INITIAL_WEIGHTS = (0.90, 0.25, 0.45)
# Bands of extensor - flexor, lowest first.
ARCHERY_ACTIONS = (LEFT, RELEASE, RIGHT)
ARCHERY_INITIAL_THRESHOLDS = (-0.015, 0.015)


class ArcheryEncoder(FeatureEncoder):
    """Learned per-neuron weights on [aim / 0.5, wind / 4, draw], plus a 0.45 tonic drive.

    aim and wind are signed and clipped to [-1, 1]; draw is 0 (undrawn) to 1 (drawn).
    They are virtual-range variables, not signals measured in a fly.
    """
    INPUTS = ARCHERY_INPUTS

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray,
                 aim_scale: float = 0.5, wind_scale: float = 4.0):
        super().__init__(circuit, weights)
        self.aim_scale, self.wind_scale = aim_scale, wind_scale

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.array([
            np.clip(observation["aim_offset"] / self.aim_scale, -1.0, 1.0),
            np.clip(observation["wind"] / self.wind_scale, -1.0, 1.0),
            np.clip(observation["draw"], 0.0, 1.0),
        ])


class ArcheryDecoder(BandDecoder):
    """Flexor-dominant: aim LEFT; balanced: RELEASE; extensor-dominant: aim RIGHT.

    RELEASE before the bow is drawn is ignored by the event, so it also means "hold".
    """
    def __init__(self, circuit: ConnectomeCircuit, thresholds=ARCHERY_INITIAL_THRESHOLDS):
        super().__init__(circuit, ARCHERY_ACTIONS, thresholds)


def initial_archery_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Every sensory neuron gets ARCHERY_INITIAL_WEIGHTS."""
    return np.tile(ARCHERY_INITIAL_WEIGHTS, (len(circuit.indices("sensory")), 1))
