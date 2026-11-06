"""Sprint input from a drawn ground-motion calibration of FlyVis.

The eye-to-leg link is modeled: a lookup of steady T4/T5 activity drives the
Sprint circuit's sensory neurons. Running the controller only needs NumPy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from events.sprint import ACCELERATE, COAST
from .event_io import BandDecoder
from .sensory import LearnedSensoryEncoder

SPEED_CALIBRATION = Path(__file__).resolve().parents[1] / "models" / "flyvis_speed_eye.json"
SPRINT_EYE_INPUTS = ("distance", "eye")
SPRINT_EYE_ACTIONS = (COAST, ACCELERATE)


class FlyVisSpeedEye:
    """Interpolate horizontal motion activity, zeroed at rest and scaled by its span."""

    def __init__(self, path: Path = SPEED_CALIBRATION):
        path = Path(path)
        table = json.loads(path.read_text())
        self.speeds = np.asarray(table["speed_bl_per_s"], dtype=float)
        self.signal = np.asarray(table["signal"], dtype=float)
        if (self.speeds.ndim != 1 or len(self.speeds) < 2 or self.speeds[0] != 0.0
                or np.any(np.diff(self.speeds) <= 0) or self.signal.shape != self.speeds.shape
                or not np.isfinite(self.speeds).all() or not np.isfinite(self.signal).all()):
            raise ValueError(f"{path}: expected finite signals at increasing speeds starting at zero.")
        self.centre = self.signal[0]
        self.span = np.max(np.abs(self.signal - self.centre))
        if self.span == 0.0:
            raise ValueError(f"{path}: the eye signal is constant.")

    def __call__(self, speed: float) -> float:
        return float((np.interp(speed, self.speeds, self.signal) - self.centre) / self.span)


def blindfolded_eye(_speed: float) -> float:
    return 0.0


class SprintEyeEncoder(LearnedSensoryEncoder):
    """Keep distance to finish and use visual activity for the second feature."""

    def __init__(self, circuit, weights: np.ndarray, eye: Callable[[float], float] | None = None):
        super().__init__(circuit, weights)
        self.eye = eye if eye is not None else FlyVisSpeedEye()

    def features(self, observation: dict[str, float]) -> np.ndarray:
        # One Sprint velocity unit maps to one body length/s in the drawn scene.
        distance = min(1.0, max(0.0, observation["distance_to_finish"] / self.sprint_distance))
        return np.array([distance, self.eye(observation["velocity"])])


class SprintEyeDecoder(BandDecoder):
    def __init__(self, circuit, thresholds):
        super().__init__(circuit, SPRINT_EYE_ACTIONS, thresholds)
