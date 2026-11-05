"""Transparent non-neural Hurdles baselines; each one tests a way to game the event."""
from __future__ import annotations

import math

import numpy as np

from events.hurdles import ACCELERATE, COAST, JUMP, LEGAL_ACTIONS, Hurdles
from .base import Controller


class AlwaysController(Controller):
    """Repeats one action: ACCELERATE tests the hit cost, COAST and JUMP never get going."""
    def __init__(self, action: str) -> None:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal hurdles action {action!r}")
        self.action = action

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return self.action


class RandomController(Controller):
    """Jumps with probability p_jump and coasts with p_coast per step, otherwise accelerates."""
    def __init__(self, p_jump: float = 0.01, p_coast: float = 0.0) -> None:
        if not (0 <= p_jump <= 1 and 0 <= p_coast <= 1 and p_jump + p_coast <= 1):
            raise ValueError("Probabilities must be between 0 and 1 and sum to at most 1.")
        self.p_jump, self.p_coast = p_jump, p_coast

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        draw = self.rng.random()
        if draw < self.p_jump:
            return JUMP
        return COAST if draw < self.p_jump + self.p_coast else ACCELERATE


class HurdleRuleController(Controller):
    """Hand-written rule: brake to safe_speed before each hurdle, then jump mid-window.

    With coast=False it never brakes, so it takes off too fast. It knows the event's
    physics constants, which a learned controller would not.
    """
    def __init__(self, coast: bool = True, event: Hurdles | None = None) -> None:
        self.coast = coast
        self.event = event if event is not None else Hurdles()
        self.dt, self.drag = self.event.dt, self.event.drag
        self.air_time = round(self.event.air_time / self.event.dt) * self.event.dt
        # The arc is above the hurdle between these fractions of the jump.
        half_width = math.sqrt(1 - self.event.hurdle_height / self.event.jump_height) / 2
        self.window = (0.5 - half_width, 0.5 + half_width)

    def reset(self, seed: int | None = None) -> None:
        pass

    def take_off_distance(self, velocity: float) -> float:
        """Distance before the hurdle at the middle of the window, for a take-off at this speed."""
        air_speed = velocity * self.event.takeoff_keep_at(velocity) * (1 - self.drag * self.air_time / 2)
        return sum(self.window) / 2 * air_speed * self.air_time

    def act(self, observation: dict[str, float]) -> str:
        distance, velocity = observation["distance_to_next_hurdle"], observation["velocity"]
        # Jumps on the first step within half a step of the target, so the average take-off lands on it.
        take_off = self.take_off_distance(velocity) + velocity * self.dt / 2
        if distance <= take_off:
            return JUMP
        # Coasting from v down to safe_speed takes (v - safe_speed) / drag metres.
        if self.coast and velocity > self.event.safe_speed:
            if distance - take_off <= (velocity - self.event.safe_speed) / self.drag:
                return COAST
        return ACCELERATE
