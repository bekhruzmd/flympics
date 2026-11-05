"""A deterministic 100 m swimming race in a 50 m pool with PULL/RECOVER actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import OlympicEvent

PULL = "PULL"
RECOVER = "RECOVER"
LEGAL_ACTIONS = frozenset((PULL, RECOVER))


@dataclass
class Swimming(OlympicEvent):
    """One-dimensional 100 m swim with one turn; this is a game model, not fly biomechanics.

    The race is fixed: the seed passed to reset() changes nothing.
    """

    distance: float = 100.0
    pool_length: float = 50.0
    dt: float = 0.05
    max_time: float = 120.0
    pull_time: float = 0.6
    recover_time: float = 0.4
    thrust: float = 5.0
    grip_time: float = 0.25
    hand_speed: float = 4.0
    swim_drag: float = 0.08
    glide_drag: float = 0.02
    linear_drag: float = 0.15
    recovery_drag: float = 0.5
    dive_speed: float = 4.0
    push_off: float = 3.5
    turn_keep: float = 0.2

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        # The arm moves in whole steps; integer units keep a exactly 0 or 1 at the ends.
        self._pull_steps = max(1, round(self.pull_time / self.dt))
        self._recover_steps = max(1, round(self.recover_time / self.dt))
        self._arm_full = self._pull_steps * self._recover_steps
        self._arm = 0
        self._pull_run = 0
        self.position = 0.0
        self.velocity = self.dive_speed
        self.time = 0.0
        self.steps = 0
        self.strokes = 0
        self.turns = 0
        self.grip = 0.0
        self.gliding = False
        self._finished = False
        return self.observe()

    @property
    def stroke_phase(self) -> float:
        return self._arm / self._arm_full

    @property
    def lap(self) -> int:
        return self.turns + 1

    def _next_wall(self) -> float:
        return min(self.distance, self.pool_length * (self.turns + 1))

    def observe(self) -> Mapping[str, float]:
        return {
            "distance_to_wall": max(0.0, self._next_wall() - self.position),
            "velocity": self.velocity,
            "stroke_phase": self.stroke_phase,
        }

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal swimming action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished race; call reset().")
        v = self.velocity
        force, body_drag, extra = 0.0, self.swim_drag, 0.0
        self.grip, self.gliding = 0.0, False
        if action == PULL:
            if self._arm < self._arm_full:
                if self._pull_run == 0:
                    self.strokes += 1
                self._pull_run += 1
                self.grip = min(1.0, self._pull_run * self.dt / self.grip_time)
                force = self.thrust * self.grip * max(0.0, 1.0 - v / self.hand_speed)
                self._arm = min(self._arm_full, self._arm + self._recover_steps)
            # At a = 1 the arm has no range left: no thrust, normal drag.
        else:
            self._pull_run = 0
            if self._arm > 0:
                extra = self.recovery_drag * v
                self._arm = max(0, self._arm - self._pull_steps)
            else:
                body_drag, self.gliding = self.glide_drag, True
        # Semi-implicit Euler, as in Sprint.
        drag = body_drag * v * v + self.linear_drag * v + extra
        self.velocity = max(0.0, v + self.dt * (force - drag))
        self.position = min(self.distance, self.position + self.dt * self.velocity)
        if self.position >= self._next_wall() and self.position < self.distance:
            self.turns += 1
            self.velocity = self.turn_keep * self.velocity + self.push_off
        self.steps += 1
        self.time = self.steps * self.dt
        self._finished = self.position >= self.distance or self.time >= self.max_time
        return self.observe()

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Finished time is better; unfinished trials receive a distance-sensitive penalty."""
        return self.time if self.position >= self.distance else self.max_time + (self.distance - self.position)
