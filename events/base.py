"""Stable interface shared by all Flympics events."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class OlympicEvent(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> Mapping[str, float]: ...

    @abstractmethod
    def observe(self) -> Mapping[str, float]: ...

    @abstractmethod
    def step(self, action: str) -> Mapping[str, float]: ...

    @abstractmethod
    def is_finished(self) -> bool: ...

    @abstractmethod
    def score(self) -> float: ...
