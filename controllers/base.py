from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class Controller(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> None: ...

    @abstractmethod
    def act(self, observation: Mapping[str, float]) -> str: ...
