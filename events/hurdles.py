"""A deterministic 200 m hurdles environment with ACCELERATE/COAST/JUMP actions."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

from .base import OlympicEvent

# Same strings as Sprint's actions.
ACCELERATE = "ACCELERATE"
COAST = "COAST"
JUMP = "JUMP"
LEGAL_ACTIONS = frozenset((ACCELERATE, COAST, JUMP))

# Layout rules: every gap is wider than one jump can clear, so each hurdle needs its own jump.
MIN_RUN_UP = 15.0
MIN_RUN_OUT = 12.0
MIN_GAP = 8.0
GAP = (8.0, 30.0)

# Fixed scoreboard course: short gaps that leave no room to speed up, and long gaps that
# end in a braking stretch.
BENCHMARK_LAYOUT = (15.0, 24.0, 52.0, 61.0, 70.5, 98.0, 107.0, 135.0, 144.0, 158.0, 186.0)


def validate_layout(hurdles: Sequence[float], distance: float) -> None:
    """Raise ValueError unless the run-up, run-out and every gap are long enough."""
    if not hurdles:
        raise ValueError("A hurdles layout needs at least one hurdle.")
    if hurdles[0] < MIN_RUN_UP:
        raise ValueError(f"First hurdle at {hurdles[0]} m; the run-up must be at least {MIN_RUN_UP} m.")
    if distance - hurdles[-1] < MIN_RUN_OUT:
        raise ValueError(f"Last hurdle at {hurdles[-1]} m; the run-out must be at least {MIN_RUN_OUT} m.")
    for i, (a, b) in enumerate(zip(hurdles, hurdles[1:]), start=1):
        if b - a < MIN_GAP:
            raise ValueError(f"Gap of {b - a:.2f} m after hurdle {i - 1}; every gap must be at least {MIN_GAP} m.")


def generate_layout(seed: int, distance: float = 200.0) -> tuple[float, ...]:
    """Seeded course with every gap drawn uniformly from GAP."""
    rng = random.Random(seed)
    hurdles, position = [], MIN_RUN_UP
    while position <= distance - MIN_RUN_OUT:
        hurdles.append(round(position, 3))
        position += rng.uniform(*GAP)
    layout = tuple(hurdles)
    validate_layout(layout, distance)
    return layout


@dataclass
class Hurdles(OlympicEvent):
    """One-dimensional 200 m hurdles; this is a game model, not fly biomechanics."""

    distance: float = 200.0
    dt: float = 0.05
    max_time: float = 90.0
    acceleration: float = 1.25
    drag: float = 0.16
    air_time: float = 0.8
    jump_height: float = 1.0
    hurdle_height: float = 0.5
    takeoff_keep: float = 0.90
    safe_speed: float = 4.5
    overspeed_cost: float = 0.40
    min_takeoff_keep: float = 0.20
    stumble_keep: float = 0.30
    sensor_range: float = 40.0
    layout: tuple[float, ...] | None = None
    seeded: bool = False

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        if self.layout is not None:
            hurdles = tuple(self.layout)
        elif self.seeded:
            if seed is None:
                raise ValueError("Seeded hurdle layouts need a seed.")
            hurdles = generate_layout(seed, self.distance)
        else:
            hurdles = BENCHMARK_LAYOUT
        validate_layout(hurdles, self.distance)
        self.hurdles = hurdles
        # Jump length in whole steps keeps landing exact for any dt that divides air_time.
        self._air_steps = max(1, round(self.air_time / self.dt))
        self.position = 0.0
        self.velocity = 0.0
        self.time = 0.0
        self.steps = 0
        self._air: int | None = None
        self._next = 0
        self.hits = 0
        self.jumps = 0
        self.clears = 0
        self.coasts = 0
        self.last_takeoff_keep: float | None = None
        self._finished = False
        return self.observe()

    def observe(self) -> Mapping[str, float]:
        return {
            "distance_to_next_hurdle": self._sensed(self._next),
            "distance_to_second_hurdle": self._sensed(self._next + 1),
            "velocity": self.velocity,
        }

    def _sensed(self, index: int) -> float:
        # Missing hurdles read as the cap: "nothing nearby".
        if index >= len(self.hurdles):
            return self.sensor_range
        return min(self.hurdles[index] - self.position, self.sensor_range)

    @property
    def airborne(self) -> bool:
        return self._air is not None

    @property
    def height(self) -> float:
        return 0.0 if self._air is None else self._arc(self._air * self.dt)

    def _arc(self, t: float) -> float:
        total = self._air_steps * self.dt
        return max(0.0, 4.0 * self.jump_height * t * (total - t) / total**2)

    def takeoff_keep_at(self, velocity: float) -> float:
        """Share of speed kept when jumping at this speed; over safe_speed costs extra."""
        if velocity <= self.safe_speed:
            return self.takeoff_keep
        return max(self.min_takeoff_keep, self.takeoff_keep - self.overspeed_cost * (velocity - self.safe_speed))

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal hurdles action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished race; call reset().")
        # Actions in the air are ignored, so JUMP spam is legal but slow.
        on_ground = self._air is None
        if action == JUMP and on_ground:
            self._air = 0
            self.last_takeoff_keep = self.takeoff_keep_at(self.velocity)
            self.velocity *= self.last_takeoff_keep
            self.jumps += 1
        elif action == COAST and on_ground:
            self.coasts += 1
        # Only ACCELERATE on the ground pushes; coasting and flying both lose speed to drag alone.
        force = self.acceleration if action == ACCELERATE and on_ground else 0.0
        # Semi-implicit Euler, as in Sprint.
        self.velocity = max(0.0, self.velocity + self.dt * (force - self.drag * self.velocity))
        start = self.position
        self.position = min(self.distance, self.position + self.dt * self.velocity)
        while self._next < len(self.hurdles) and self.hurdles[self._next] <= self.position:
            self._cross(self.hurdles[self._next], start)
            self._next += 1
        if self._air is not None:
            self._air += 1
            if self._air >= self._air_steps:
                self._air = None
        self.steps += 1
        # Multiplying, not summing, so 1800 steps of 0.05 s reach max_time exactly.
        self.time = self.steps * self.dt
        self._finished = self.position >= self.distance or self.time >= self.max_time
        return self.observe()

    def _cross(self, hurdle: float, start: float) -> None:
        cleared = False
        if self._air is not None:
            # Interpolate when inside this step the fly passed the hurdle.
            fraction = (hurdle - start) / (self.position - start)
            cleared = self._arc((self._air + fraction) * self.dt) >= self.hurdle_height
        if cleared:
            self.clears += 1
        else:
            self.hits += 1
            self.velocity *= self.stumble_keep

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Finished time is better; unfinished trials receive a distance-sensitive penalty."""
        return self.time if self.position >= self.distance else self.max_time + (self.distance - self.position)
