"""A deterministic 600 m steeplechase: barriers, water pits and a stamina budget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import OlympicEvent

# Same strings as the Hurdles actions.
ACCELERATE = "ACCELERATE"
COAST = "COAST"
JUMP = "JUMP"
LEGAL_ACTIONS = frozenset((ACCELERATE, COAST, JUMP))

# 12 barriers every 50 m; every third one is a water jump with a pit behind it.
BARRIERS = tuple(25.0 + 50.0 * i for i in range(12))
WATER = frozenset((2, 5, 8, 11))


@dataclass
class Steeplechase(OlympicEvent):
    """One-dimensional 600 m steeplechase; a game model, not fly biomechanics.

    Running and jumping work as in Hurdles: only ACCELERATE on the ground pushes, a jump
    lasts air_time and clears a barrier when the arc is above barrier_height as it passes.
    A hit keeps stumble_keep of the speed.

    Water jumps have a pit of pit_length metres right after the barrier. Touching the
    ground inside it (landing short, or running through after a hit) keeps only
    water_keep of the speed. Clearing it needs a long jump, so a fast take-off.

    Stamina runs from 1 (fresh) to 0. Accelerating drains it, faster at high speed;
    coasting and flying let it recover. The push scales from min_power (empty) to full
    (fresh), so running flat out early leaves the fly crawling. Jumps, hits and splashes
    cost stamina too.

    The course is fixed: the seed passed to reset() changes nothing.
    """

    distance: float = 600.0
    dt: float = 0.1
    max_time: float = 240.0
    acceleration: float = 1.25
    drag: float = 0.16
    air_time: float = 0.8
    jump_height: float = 1.0
    barrier_height: float = 0.5
    takeoff_keep: float = 0.92
    stumble_keep: float = 0.30
    pit_length: float = 3.0
    water_keep: float = 0.50
    cruise_speed: float = 5.0
    drain: float = 0.008
    recovery: float = 0.02
    min_power: float = 0.25
    jump_cost: float = 0.01
    hit_cost: float = 0.04
    water_cost: float = 0.04
    sensor_range: float = 40.0

    def reset(self, seed: int | None = None) -> Mapping[str, float]:
        self.barriers = BARRIERS
        self.water = WATER
        self._air_steps = max(1, round(self.air_time / self.dt))
        self.position = 0.0
        self.velocity = 0.0
        self.stamina = 1.0
        self.time = 0.0
        self.steps = 0
        self._air: int | None = None
        self._next = 0           # next barrier not yet passed
        self._pit = 0            # next water pit not yet resolved (index into sorted WATER)
        self._pits = tuple(self.barriers[i] for i in sorted(self.water))
        self.hits = 0
        self.clears = 0
        self.jumps = 0
        self.splashes = 0
        self.dry_pits = 0
        self.min_stamina = 1.0
        self._finished = False
        return self.observe()

    def observe(self) -> Mapping[str, float]:
        if self._next < len(self.barriers):
            to_barrier = min(self.barriers[self._next] - self.position, self.sensor_range)
            water = float(self._next in self.water)
        else:
            to_barrier, water = self.sensor_range, 0.0
        return {
            "distance_to_barrier": to_barrier,
            "water_next": water,
            "velocity": self.velocity,
            "stamina": self.stamina,
            "distance_to_finish": max(0.0, self.distance - self.position),
        }

    @property
    def airborne(self) -> bool:
        return self._air is not None

    @property
    def height(self) -> float:
        return 0.0 if self._air is None else self._arc(self._air * self.dt)

    def _arc(self, t: float) -> float:
        total = self._air_steps * self.dt
        return max(0.0, 4.0 * self.jump_height * t * (total - t) / total**2)

    def power(self) -> float:
        return self.min_power + (1.0 - self.min_power) * self.stamina

    def _spend(self, amount: float) -> None:
        self.stamina = min(1.0, max(0.0, self.stamina - amount))
        self.min_stamina = min(self.min_stamina, self.stamina)

    def step(self, action: str) -> Mapping[str, float]:
        if action not in LEGAL_ACTIONS:
            raise ValueError(f"Illegal steeplechase action {action!r}; choose from {sorted(LEGAL_ACTIONS)}")
        if self._finished:
            raise RuntimeError("Cannot step a finished race; call reset().")
        on_ground = self._air is None
        if action == JUMP and on_ground:
            self._air = 0
            self.velocity *= self.takeoff_keep
            self.jumps += 1
            self._spend(self.jump_cost)
        pushing = action == ACCELERATE and on_ground
        if pushing:
            self._spend(self.drain * self.dt * (self.velocity / self.cruise_speed) ** 2)
        else:
            self._spend(-self.recovery * self.dt)
        force = self.acceleration * self.power() if pushing else 0.0
        # Semi-implicit Euler, as in Sprint and Hurdles.
        self.velocity = max(0.0, self.velocity + self.dt * (force - self.drag * self.velocity))
        start = self.position
        self.position = min(self.distance, self.position + self.dt * self.velocity)
        while self._next < len(self.barriers) and self.barriers[self._next] <= self.position:
            self._cross(self.barriers[self._next], start)
            self._next += 1
        if self._air is not None:
            self._air += 1
            if self._air >= self._air_steps:
                self._air = None
        if self._air is None:
            self._check_pit()
        self.steps += 1
        self.time = self.steps * self.dt
        self._finished = self.position >= self.distance or self.time >= self.max_time
        return self.observe()

    def _cross(self, barrier: float, start: float) -> None:
        cleared = False
        if self._air is not None:
            fraction = (barrier - start) / (self.position - start)
            cleared = self._arc((self._air + fraction) * self.dt) >= self.barrier_height
        if cleared:
            self.clears += 1
        else:
            self.hits += 1
            self.velocity *= self.stumble_keep
            self._spend(self.hit_cost)

    def _check_pit(self) -> None:
        """On the ground: resolve the next pit once the fly is past its barrier."""
        while self._pit < len(self._pits) and self.position >= self._pits[self._pit]:
            if self.position < self._pits[self._pit] + self.pit_length:
                self.splashes += 1
                self.velocity *= self.water_keep
                self._spend(self.water_cost)
            else:
                self.dry_pits += 1
            self._pit += 1

    def is_finished(self) -> bool:
        return self._finished

    def score(self) -> float:
        """Finished time is better; unfinished trials receive a distance-sensitive penalty."""
        return self.time if self.position >= self.distance else self.max_time + (self.distance - self.position)
