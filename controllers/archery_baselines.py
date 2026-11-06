"""Transparent non-neural Archery baselines; each one tests a way to game the event."""
from __future__ import annotations

import numpy as np

from events.archery import HOLD, LEFT, LEGAL_ACTIONS, RELEASE, RIGHT, Archery
from .base import Controller


class AlwaysController(Controller):
    """Repeats one action: RELEASE shoots at the centre ignoring wind, HOLD never shoots."""
    def __init__(self, action: str) -> None:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal archery action {action!r}")
        self.action = action

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return self.action


class RandomController(Controller):
    """LEFT, RIGHT or HOLD at random, releasing with probability p_release per step."""
    def __init__(self, p_release: float = 0.1) -> None:
        if not 0 <= p_release <= 1:
            raise ValueError("Release probability must be between 0 and 1.")
        self.p_release = p_release

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        if self.rng.random() < self.p_release:
            return RELEASE
        return (LEFT, RIGHT, HOLD)[int(self.rng.integers(3))]


class ArcheryRuleController(Controller):
    """Hand-written rule: aim at -drift_per_wind * wind, release once within tolerance.

    It knows the event's drift constant, which a learned controller would not.
    """
    def __init__(self, tolerance: float = 0.025, event: Archery | None = None) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        self.tolerance = tolerance
        self.drift = (event if event is not None else Archery()).drift_per_wind

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        error = observation["aim_offset"] + self.drift * observation["wind"]
        if error > self.tolerance:
            return LEFT
        if error < -self.tolerance:
            return RIGHT
        return RELEASE if observation["draw"] >= 1.0 else HOLD
