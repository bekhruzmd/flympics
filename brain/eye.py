"""A FlyVis-calibrated eye for the Archery fly.

The 135-neuron MANC circuit has no eyes. This module gives the Archery fly a wind sense
measured from FlyVis, a model of the fly's real motion-vision system: for each wind
strength, models/flyvis_wind_eye.json stores the steady response of its direction-selective
T4/T5 cells to grass drifting at that speed (written by
`python -m experiments.vision_archery --calibrate` in the optional vision environment).

The eye-to-leg link is a modeling choice: the calibrated signal drives the same input
neurons the plain Archery encoder uses. FlyVis itself is not run here, so this module
needs only NumPy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .archery_io import ArcheryEncoder

EYE_CALIBRATION = Path(__file__).resolve().parents[1] / "models" / "flyvis_wind_eye.json"
ARCHERY_EYE_INPUTS = ("aim", "eye", "draw")


class FlyVisWindEye:
    """Wind -> calibrated T4/T5 signal, centred on no wind and scaled to about [-1, 1]."""

    def __init__(self, path: Path = EYE_CALIBRATION):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing eye calibration {path}; see docs/vision.md.")
        table = json.loads(path.read_text())
        self.wind = np.asarray(table["wind"], dtype=float)
        self.signal = np.asarray(table["signal"], dtype=float)
        if np.any(np.diff(self.wind) <= 0):
            raise ValueError(f"{path}: wind levels must be increasing.")
        self.centre = float(np.interp(0.0, self.wind, self.signal))
        self.span = float(np.max(np.abs(self.signal - self.centre)))

    def raw(self, wind: float) -> float:
        """The T4/T5 signal FlyVis gives for this wind (clamped at the calibrated range)."""
        return float(np.interp(wind, self.wind, self.signal))

    def __call__(self, wind: float) -> float:
        return (self.raw(wind) - self.centre) / self.span


class ArcheryEyeEncoder(ArcheryEncoder):
    """Archery encoder that senses wind through the FlyVis eye instead of reading it.

    Inputs: aim (where the bow arm points, a leg-proprioceptive sense), eye (the FlyVis
    motion signal), draw (how far the bow is drawn).
    """
    INPUTS = ARCHERY_EYE_INPUTS

    def __init__(self, circuit, weights: np.ndarray, eye: FlyVisWindEye | None = None, aim_scale: float = 0.5):
        super().__init__(circuit, weights, aim_scale=aim_scale)
        self.eye = eye if eye is not None else FlyVisWindEye()

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.array([
            np.clip(observation["aim_offset"] / self.aim_scale, -1.0, 1.0),
            np.clip(self.eye(observation["wind"]), -1.0, 1.0),
            np.clip(observation["draw"], 0.0, 1.0),
        ])


class BlindfoldedEye(FlyVisWindEye):
    """The same eye, blindfolded: it always reports no wind. For showing the fly needs it."""

    def __call__(self, wind: float) -> float:
        return 0.0


class SwappedEye(FlyVisWindEye):
    """The same eye with left and right swapped."""

    def __call__(self, wind: float) -> float:
        return super().__call__(-wind)
