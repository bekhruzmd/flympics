"""The only controller that couples the event loop to the neural model."""
from __future__ import annotations

from typing import Mapping
from pathlib import Path
import json
import math

from controllers.base import Controller
from .connectome import load_manc_front_leg_circuit
from .dynamics import RateNetwork
from .motor import MotorDecoder
from .sensory import SensoryEncoder

SPRINT_POLICY = Path(__file__).resolve().parents[1] / "models" / "sprint.json"


class FlyBrainController(Controller):
    """A connectome-derived neural controller, not a simulated complete fly brain."""
    def __init__(self, motor_threshold: float = 0.015) -> None:
        if not math.isfinite(motor_threshold) or not 0 <= motor_threshold <= 1:
            raise ValueError("Motor threshold must be finite and between 0 and 1.")
        self.circuit = load_manc_front_leg_circuit()
        self.encoder = SensoryEncoder(self.circuit)
        self.network = RateNetwork(self.circuit)
        self.decoder = MotorDecoder(self.circuit, threshold=motor_threshold)
        self.last_summary: dict[str, float] = {}

    @classmethod
    def trained(cls, path: Path = SPRINT_POLICY) -> FlyBrainController:
        if not path.exists():
            raise FileNotFoundError(f"Missing sprint policy: {path}. Run python -m experiments.train_sprint")
        policy = json.loads(path.read_text())
        if policy.get("version") != 1 or policy.get("event") != "sprint":
            raise ValueError("Expected a version 1 sprint policy.")
        return cls(motor_threshold=float(policy["motor_threshold"]))

    def reset(self, seed: int | None = None) -> None:
        self.network.reset()
        self.last_summary = {}

    def act(self, observation: Mapping[str, float]) -> str:
        stimulus = self.encoder.encode(dict(observation))
        activity = self.network.step(stimulus)
        action, motor = self.decoder.decode(activity)
        self.last_summary = {
            "sensory_mean": float(activity[self.encoder.indices].mean()),
            "intrinsic_mean": float(activity[self.circuit.indices("intrinsic")].mean()),
            "motor_extensor": motor["extensor"],
            "motor_flexor": motor["flexor"],
            "motor_difference": motor["difference"],
        }
        return action
