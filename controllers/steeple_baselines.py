"""Transparent non-neural Steeplechase baselines; each one tests a way to game the event."""
from __future__ import annotations

import math

import numpy as np

from events.steeplechase import ACCELERATE, COAST, JUMP, LEGAL_ACTIONS, Steeplechase
from .base import Controller


class AlwaysController(Controller):
    """Repeats one action: ACCELERATE hits every barrier and burns out, COAST never starts."""
    def __init__(self, action: str) -> None:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal steeplechase action {action!r}")
        self.action = action

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return self.action


class RandomController(Controller):
    """Jumps with probability p_jump and coasts with p_coast per step, otherwise accelerates."""
    def __init__(self, p_jump: float = 0.02, p_coast: float = 0.3) -> None:
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


class SteepleRuleController(Controller):
    """Hand-written rule: hold a pace, speed up for water jumps, kick at the end.

    It holds `pace` m/s, raises it to `water_speed` over the last `water_run_up` metres
    before a water barrier, and accelerates flat out over the last `kick` metres. It jumps
    so the barrier passes under the top of the arc, or, for water, as late as still lands
    past the pit. It knows the event's physics constants, which a learned controller would not.
    """
    def __init__(self, pace: float = 5.0, water_speed: float = 6.2, water_run_up: float = 30.0,
                 kick: float = 80.0, event: Steeplechase | None = None) -> None:
        self.pace, self.water_speed = pace, water_speed
        self.water_run_up, self.kick = water_run_up, kick
        self.event = event if event is not None else Steeplechase()
        e = self.event
        self.air_steps = max(1, round(e.air_time / e.dt))
        # Fractions of the flight during which the arc is above the barrier.
        half_width = math.sqrt(1 - e.barrier_height / e.jump_height) / 2
        self.window = (0.5 - half_width, 0.5 + half_width)

    def reset(self, seed: int | None = None) -> None:
        pass

    def flight(self, velocity: float) -> list[float]:
        """Distance travelled after each airborne step if the fly jumps now."""
        e, v, x, out = self.event, velocity * self.event.takeoff_keep, 0.0, []
        for _ in range(self.air_steps):
            v = max(0.0, v + e.dt * (-e.drag * v))
            x += e.dt * v
            out.append(x)
        return out

    def crossing(self, d: float, velocity: float) -> float | None:
        """When a jump taken now passes a barrier d metres ahead, as a fraction of the flight."""
        if d <= 0:
            return None
        positions = [0.0, *self.flight(velocity)]
        for k in range(1, len(positions)):
            if positions[k] >= d:
                step = (d - positions[k - 1]) / (positions[k] - positions[k - 1])
                return (k - 1 + step) / self.air_steps
        return None

    def act(self, observation: dict[str, float]) -> str:
        d, v = observation["distance_to_barrier"], observation["velocity"]
        water = observation["water_next"] > 0.5
        e = self.event
        if d < e.sensor_range and v > 0:
            frac = self.crossing(d, v)
            if frac is not None and self.window[0] + 0.02 <= frac <= self.window[1] - 0.02:
                if water:
                    # Jumping later lands further past the pit, so take off at the last step
                    # that still clears the barrier.
                    later = self.crossing(d - e.dt * v, v)
                    if later is None or later < self.window[0] + 0.02:
                        return JUMP
                elif frac <= 0.5:
                    return JUMP
        target = self.pace
        if water and d < self.water_run_up:
            target = self.water_speed
        if observation["distance_to_finish"] < self.kick:
            target = math.inf
        return ACCELERATE if v < target else COAST
