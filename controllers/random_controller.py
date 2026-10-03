from __future__ import annotations

import numpy as np

from events.sprint import LEGAL_ACTIONS
from .base import Controller


class RandomController(Controller):
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, float]) -> str:
        return str(self.rng.choice(sorted(LEGAL_ACTIONS)))
