"""A deliberately small, deterministic 100 m sprint environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import OlympicEvent

COAST = "COAST"
ACCELERATE = "ACCELERATE"
LEGAL_ACTIONS = frozenset((COAST, ACCELERATE))


@dataclass
class Sprint(OlympicEvent):
    """One-dimensional 100 m sprint; this is a game model, not fly biomechanics."""

    distance: float = 100.0
    dt: float = 0.1
    max_time: float = 60.0
    acceleration: float = 1.25
    drag: float = 0.16

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        self.position = 0.0
        self.velocity = 0.0
        self.time = 0.0
        self.steps = 0
        self._finished = False
        return self.observe()

    def observe(self) -> Mapping[str, float]:
        return {
            "distance_to_finish": max(0.0, self.distance - self.position),
            "velocity": self.velocity,
        }

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal sprint action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished sprint; call reset().")
        force = self.acceleration if action == ACCELERATE else 0.0
        # Semi-implicit Euler: fixed dt makes runs independent of drawing/frame rate.
        self.velocity = max(0.0, self.velocity + self.dt * (force - self.drag * self.velocity))
        self.position = min(self.distance, self.position + self.dt * self.velocity)
        self.time += self.dt
        self.steps += 1
        self._finished = self.position >= self.distance or self.time >= self.max_time
        return self.observe()

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Finished time is better; unfinished trials receive a distance-sensitive penalty."""
        return self.time if self.position >= self.distance else self.max_time + (self.distance - self.position)
