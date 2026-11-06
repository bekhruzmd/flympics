"""A deterministic 400 m pacing race with a limited energy budget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import OlympicEvent

COAST = "COAST"
ACCELERATE = "ACCELERATE"
LEGAL_ACTIONS = frozenset((COAST, ACCELERATE))


@dataclass
class Pacing400m(OlympicEvent):
    """One-dimensional 400 m race with a limited energy budget; game model, not fly biomechanics."""

    distance: float = 400.0
    dt: float = 0.1
    max_time: float = 240.0
    acceleration: float = 1.25
    drag: float = 0.10
    drag_quad: float = 0.03
    energy_budget: float = 100.0
    burn_rate: float = 1.5
    fatigue_accel: float = 0.05
    recovery_rate: float = 0.0

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        self.position = 0.0
        self.velocity = 0.0
        self.energy = self.energy_budget
        self.time = 0.0
        self.steps = 0
        self.accelerations = 0
        self.coasts = 0
        self.exhausted_steps = 0
        self._finished = False
        return self.observe()

    def observe(self) -> Mapping[str, float]:
        return {
            "distance_to_finish": max(0.0, self.distance - self.position),
            "velocity": self.velocity,
            "energy": self.energy,
        }

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal pacing action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished race; call reset().")

        if action == ACCELERATE:
            self.accelerations += 1
            if self.energy > 0.0:
                cost = self.burn_rate * self.dt
                actual_cost = min(self.energy, cost)
                self.energy -= actual_cost
                if self.energy < 1e-12:
                    self.energy = 0.0
                # Scale force if energy depleted mid-step
                force = self.acceleration * (actual_cost / cost) if cost > 0 else self.acceleration
                if self.energy == 0.0:
                    self.exhausted_steps += 1
            else:
                force = self.fatigue_accel
                self.exhausted_steps += 1
        else:
            self.coasts += 1
            force = 0.0
            if self.recovery_rate > 0.0:
                self.energy = min(self.energy_budget, self.energy + self.recovery_rate * self.dt)

        # Semi-implicit Euler with linear and quadratic drag
        drag_force = self.drag * self.velocity + self.drag_quad * self.velocity * self.velocity
        self.velocity = max(0.0, self.velocity + self.dt * (force - drag_force))
        self.position = min(self.distance, self.position + self.dt * self.velocity)
        self.steps += 1
        self.time = round(self.steps * self.dt, 8)
        self._finished = self.position >= self.distance or self.time >= self.max_time
        return self.observe()

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Finished time is better; unfinished trials receive a distance-sensitive penalty."""
        return self.time if self.position >= self.distance else self.max_time + (self.distance - self.position)


# Alias for convenience
Pacing = Pacing400m
