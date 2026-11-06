"""A deterministic, seeded archery end: adjust the aim against a gusty crosswind, then release."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .base import OlympicEvent

LEFT = "LEFT"
RIGHT = "RIGHT"
HOLD = "HOLD"
RELEASE = "RELEASE"
LEGAL_ACTIONS = frozenset((LEFT, RIGHT, HOLD, RELEASE))


def _clip(value: float, limit: float) -> float:
    return min(limit, max(-limit, value))


@dataclass
class Archery(OlympicEvent):
    """One end of arrows at a 10-ring target; a game model, not fly (or human) biomechanics.

    Only horizontal aim matters. Each arrow starts aimed at the centre with the bow
    undrawn. The crosswind is a per-arrow base wind plus a gust (an Ornstein-Uhlenbeck
    process), so the right aim keeps moving. An arrow lands at

        landing = aim + drift_per_wind * wind_at_release

    RELEASE is ignored until the bow is fully drawn. After steady_time the aim starts to
    wander (fatigue). An arrow not released within shot_clock scores 0.

    The seed fixes the wind and the sway; the same seed always gives the same end.
    score() is total points, so **higher is better** (unlike the race events).
    """

    arrows: int = 6
    dt: float = 0.1
    shot_clock: float = 10.0
    draw_time: float = 1.0
    aim_rate: float = 0.4
    max_aim: float = 0.8
    drift_per_wind: float = 0.12
    wind_sd: float = 2.0
    wind_max: float = 4.0
    gust_sd: float = 0.8
    gust_time: float = 1.5
    steady_time: float = 4.0
    sway_rate: float = 0.03
    release_sd: float = 0.02
    ring_width: float = 0.061
    rings: int = 10

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self._draw_steps = max(1, round(self.draw_time / self.dt))
        self._clock_steps = max(self._draw_steps + 1, round(self.shot_clock / self.dt))
        self._steady_steps = round(self.steady_time / self.dt)
        self.time = 0.0
        self.steps = 0
        self.arrow = 0          # index of the arrow being shot
        self.points: list[int] = []
        self.landings: list[float | None] = []   # None: timed out
        self.release_times: list[float] = []
        self.last_landing: float | None = None
        self._finished = False
        self._new_arrow()
        return self.observe()

    def _new_arrow(self) -> None:
        self.aim = 0.0
        self.base_wind = _clip(float(self.rng.normal(0.0, self.wind_sd)), self.wind_max)
        self.gust = float(self.rng.normal(0.0, self.gust_sd))
        self.arrow_steps = 0

    @property
    def wind(self) -> float:
        return _clip(self.base_wind + self.gust, self.wind_max)

    @property
    def draw(self) -> float:
        return min(1.0, self.arrow_steps / self._draw_steps)

    @property
    def hold_time(self) -> float:
        return self.arrow_steps * self.dt

    @property
    def total_points(self) -> int:
        return sum(self.points)

    def landing_at(self, aim: float, wind: float) -> float:
        return aim + self.drift_per_wind * wind

    def points_for(self, landing: float | None) -> int:
        if landing is None:
            return 0
        return max(0, self.rings - int(math.floor(abs(landing) / self.ring_width)))

    def observe(self) -> Mapping[str, float]:
        return {
            "aim_offset": self.aim,
            "wind": self.wind,
            "draw": self.draw,
            "hold_time": self.hold_time,
            "arrows_left": float(self.arrows - self.arrow),
        }

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal archery action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished end; call reset().")
        self.last_landing = None
        released = action == RELEASE and self.draw >= 1.0
        if released:
            jitter = self.release_sd * float(self.rng.normal())
            self._score_arrow(self.landing_at(self.aim, self.wind) + jitter)
        else:
            if action == LEFT:
                self.aim -= self.aim_rate * self.dt
            elif action == RIGHT:
                self.aim += self.aim_rate * self.dt
            if self.arrow_steps >= self._steady_steps:
                self.aim += self.sway_rate * math.sqrt(self.dt) * float(self.rng.normal())
            self.aim = _clip(self.aim, self.max_aim)
            # The gust relaxes towards 0 with correlation time gust_time (exact OU update).
            decay = math.exp(-self.dt / self.gust_time)
            self.gust = decay * self.gust + self.gust_sd * math.sqrt(1 - decay**2) * float(self.rng.normal())
            self.arrow_steps += 1
            if self.arrow_steps >= self._clock_steps:
                self._score_arrow(None)
        self.steps += 1
        self.time = self.steps * self.dt
        return self.observe()

    def _score_arrow(self, landing: float | None) -> None:
        self.last_landing = landing
        self.landings.append(landing)
        self.points.append(self.points_for(landing))
        self.release_times.append(self.hold_time)
        self.arrow += 1
        if self.arrow >= self.arrows:
            self._finished = True
        else:
            self._new_arrow()

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Total points, 0 to 10 per arrow. Higher is better."""
        return float(self.total_points)
