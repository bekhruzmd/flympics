from __future__ import annotations

from events.sprint import ACCELERATE, COAST
from .base import Controller


class RuleController(Controller):
    """Transparent non-neural baseline: accelerate until a modest target speed."""
    target_velocity = 7.0

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: dict[str, float]) -> str:
        return ACCELERATE if observation["velocity"] < self.target_velocity else COAST
