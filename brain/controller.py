"""The only controller that couples the event loop to the neural model."""
from __future__ import annotations

from typing import Mapping
from pathlib import Path
import json
import math

import numpy as np

from controllers.base import Controller
from .connectome import ConnectomeCircuit, load_manc_front_leg_circuit
from .dynamics import RateNetwork
from .motor import MotorDecoder
from .sensory import LearnedSensoryEncoder, SensoryEncoder, load_encoder_weights

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SPRINT_POLICY = MODELS_DIR / "sprint.json"
SPRINT_ENCODER = MODELS_DIR / "sprint_encoder.csv"
SPRINT_ENCODER_INIT = MODELS_DIR / "sprint_encoder_init.csv"


class FlyBrainController(Controller):
    """A connectome-derived neural controller, not a simulated complete fly brain."""
    def __init__(self, motor_threshold: float = 0.015,
                 encoder_weights: Path | np.ndarray | None = None,
                 circuit: ConnectomeCircuit | None = None) -> None:
        if not math.isfinite(motor_threshold) or not 0 <= motor_threshold <= 1:
            raise ValueError("Motor threshold must be finite and between 0 and 1.")
        self.circuit = circuit if circuit is not None else load_manc_front_leg_circuit()
        if encoder_weights is None:
            self.encoder = SensoryEncoder(self.circuit)
        else:
            if not isinstance(encoder_weights, np.ndarray):
                encoder_weights = load_encoder_weights(Path(encoder_weights), self.circuit)
            self.encoder = LearnedSensoryEncoder(self.circuit, encoder_weights)
        self.network = RateNetwork(self.circuit)
        self.decoder = MotorDecoder(self.circuit, threshold=motor_threshold)
        self.last_summary: dict[str, float] = {}

    @classmethod
    def trained(cls, path: Path = SPRINT_POLICY, circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        if not path.exists():
            raise FileNotFoundError(f"Missing sprint policy: {path}. Run python -m experiments.train_sprint")
        policy = json.loads(path.read_text())
        if policy.get("version") != 1 or policy.get("event") != "sprint":
            raise ValueError("Expected a version 1 sprint policy.")
        return cls(motor_threshold=float(policy["motor_threshold"]), circuit=circuit)

    @classmethod
    def always_accelerate(cls, path: Path = SPRINT_ENCODER) -> FlyBrainController:
        """Learned encoder weights with the default (untrained) 0.015 decoder threshold."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing encoder weights: {path}. Run python -m experiments.train_always_accelerate")
        return cls(encoder_weights=path)

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
