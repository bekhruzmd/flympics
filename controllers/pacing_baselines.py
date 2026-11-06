"""Transparent non-neural 400 m Pacing baselines."""
from __future__ import annotations

import numpy as np

from events.pacing import ACCELERATE, COAST
from .base import Controller


class AlwaysAccelerateController(Controller):
    """Sprints continuously without pacing; burns through energy early."""

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return ACCELERATE


class AlwaysCoastController(Controller):
    """Never accelerates; stays at the starting line."""

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return COAST


class RandomPacingController(Controller):
    """Accelerates with probability p per step; untimed and uncoordinated."""

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probability p must be between 0 and 1.")
        self.p = p
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        return ACCELERATE if self.rng.random() < self.p else COAST


class PacingRuleController(Controller):
    """Transparent paced baseline: accelerates only to sustain a target cruise speed.

    By holding speed around target_velocity (default 3.3 m/s), aerodynamic drag is
    kept low and the energy budget lasts across the full 400 m distance.
    """

    def __init__(self, target_velocity: float = 3.3) -> None:
        if target_velocity <= 0.0:
            raise ValueError("target_velocity must be positive.")
        self.target_velocity = target_velocity

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return ACCELERATE if observation["velocity"] < self.target_velocity else COAST


class SprintPacingController(Controller):
    """Applies Sprint's high target velocity (7.0 m/s) to 400 m; exhausts energy."""

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return ACCELERATE if observation["velocity"] < 7.0 else COAST
