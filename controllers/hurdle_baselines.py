"""Transparent non-neural Hurdles baselines; each one tests a way to game the event."""
from __future__ import annotations

import math

import numpy as np

from events.hurdles import JUMP, RUN, Hurdles
from .base import Controller


class AlwaysController(Controller):
    """Repeats one action: RUN tests the hit cost, JUMP tests the take-off cost."""
    def __init__(self, action: str) -> None:
        if action not in (RUN, JUMP):
            raise ValueError(f"Illegal hurdles action {action!r}")
        self.action = action

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return self.action


class RandomJumpController(Controller):
    """Jumps with probability p per step: about one jump per hurdle at p=0.01, untimed."""
    def __init__(self, p: float = 0.01) -> None:
        if not 0 <= p <= 1:
            raise ValueError("Jump probability must be between 0 and 1.")
        self.p = p

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        return JUMP if self.rng.random() < self.p else RUN


class HurdleRuleController(Controller):
    """Hand-written timing rule; look_ahead=1 ignores the second hurdle, so it fails some pairs.

    It knows the event's physics constants, which a learned controller would not.
    """
    def __init__(self, look_ahead: int = 2, event: Hurdles | None = None) -> None:
        if look_ahead not in (1, 2):
            raise ValueError("look_ahead must be 1 or 2.")
        self.look_ahead = look_ahead
        event = event if event is not None else Hurdles()
        self.dt = event.dt
        self.air_time = round(event.air_time / event.dt) * event.dt
        self.air_speed_keep = event.takeoff_keep * (1 - event.drag * self.air_time / 2)
        # The arc is above the hurdle between these fractions of the jump.
        half_width = math.sqrt(1 - event.hurdle_height / event.jump_height) / 2
        self.window = (0.5 - half_width, 0.5 + half_width)

    def reset(self, seed: int | None = None) -> None:
        pass

    def take_off_distance(self, observation: dict[str, float]) -> float:
        """Distance before the next hurdle at the middle of the take-off window."""
        length = observation["velocity"] * self.air_speed_keep * self.air_time
        near, far = self.window[0] * length, self.window[1] * length
        gap = observation["distance_to_second_hurdle"] - observation["distance_to_next_hurdle"]
        if self.look_ahead == 2 and gap < far - near:
            far -= gap  # one jump must clear both hurdles of a pair
        return (near + far) / 2

    def act(self, observation: dict[str, float]) -> str:
        # Jumps on the first step within half a step of the target, so the average take-off lands on it.
        half_step = observation["velocity"] * self.dt / 2
        return JUMP if observation["distance_to_next_hurdle"] <= self.take_off_distance(observation) + half_step else RUN
