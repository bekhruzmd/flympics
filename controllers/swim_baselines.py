"""Transparent non-neural Swimming baselines; each one tests a way to game the event."""
from __future__ import annotations

import numpy as np

from events.swimming import PULL, RECOVER
from .base import Controller


class AlwaysController(Controller):
    """Repeats one action: PULL stalls after one stroke, RECOVER only glides."""
    def __init__(self, action: str) -> None:
        if action not in (PULL, RECOVER):
            raise ValueError(f"Illegal swimming action {action!r}")
        self.action = action

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return self.action


class ChoppyController(Controller):
    """Switches action on every step, starting with PULL: grip never builds."""
    def reset(self, seed: int | None = None) -> None:
        self._pull = False

    def act(self, observation: dict[str, float]) -> str:
        self._pull = not self._pull
        return PULL if self._pull else RECOVER


class RandomController(Controller):
    """Pulls with probability p per step, otherwise recovers; untimed."""
    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            raise ValueError("Pull probability must be between 0 and 1.")
        self.p = p

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        return PULL if self.rng.random() < self.p else RECOVER


class SwimRuleController(Controller):
    """Base controller: full strokes, with a glide after the dive and the turn.

    It pulls until the arm reaches the end of its range (stroke_phase 1), then recovers
    until the arm is forward again (stroke_phase 0). After the dive and after the turn it
    glides while its speed is above v_break. v_break=None never glides.
    """
    def __init__(self, v_break: float | None = 2.6) -> None:
        if v_break is not None and v_break < 0:
            raise ValueError("v_break must be non-negative or None.")
        self.v_break = v_break

    def reset(self, seed: int | None = None) -> None:
        self._pulling = True
        self._gliding = self.v_break is not None
        self._last_wall: float | None = None

    def act(self, observation: dict[str, float]) -> str:
        wall, phase = observation["distance_to_wall"], observation["stroke_phase"]
        # The wall distance jumps up after the turn: start a new glide.
        if self._last_wall is not None and wall > self._last_wall and self.v_break is not None:
            self._gliding = True
        self._last_wall = wall
        if self._gliding:
            if observation["velocity"] > self.v_break:
                self._pulling = False
                return RECOVER  # brings the arm forward, then glides
            self._gliding, self._pulling = False, True
        if self._pulling and phase >= 1.0:
            self._pulling = False
        elif not self._pulling and phase <= 0.0:
            self._pulling = True
        return PULL if self._pulling else RECOVER
