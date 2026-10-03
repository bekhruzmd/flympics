"""Artificial output-population to game-action adapter."""
from __future__ import annotations

import numpy as np

from events.sprint import ACCELERATE, COAST
from .connectome import ConnectomeCircuit


class MotorDecoder:
    """Uses front-leg extensor/flexor MN activity as an artificial game readout."""
    def __init__(self, circuit: ConnectomeCircuit, threshold: float = 0.015):
        self.extensor = circuit.indices("motor_extensor")
        self.flexor = circuit.indices("motor_flexor")
        if not len(self.extensor) or not len(self.flexor):
            raise ValueError("Circuit needs extensor and flexor motor populations.")
        self.threshold = threshold

    def decode(self, activity: np.ndarray) -> tuple[str, dict[str, float]]:
        extensor = float(np.mean(activity[self.extensor]))
        flexor = float(np.mean(activity[self.flexor]))
        # This comparison is a virtual-sprint command convention, not a motor claim.
        action = ACCELERATE if extensor - flexor > self.threshold else COAST
        return action, {"extensor": extensor, "flexor": flexor, "difference": extensor - flexor}
