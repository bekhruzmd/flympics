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
from .eye import ARCHERY_EYE_INPUTS, ArcheryEyeEncoder
from .archery_io import ARCHERY_INITIAL_THRESHOLDS, ARCHERY_INPUTS, ArcheryDecoder, ArcheryEncoder
from .hurdles_io import (HURDLES_INITIAL_THRESHOLDS, HURDLES_INPUTS, HurdlesDecoder,
                         HurdlesEncoder)
from .motor import MotorDecoder
from .swimming_io import SWIMMING_INPUTS, SwimmingDecoder, SwimmingEncoder
from .sensory import LearnedSensoryEncoder, SensoryEncoder, load_encoder_weights
from .speed_eye import SPRINT_EYE_INPUTS, SprintEyeDecoder, SprintEyeEncoder
from .steeplechase_io import (STEEPLECHASE_INITIAL_THRESHOLDS, STEEPLECHASE_INPUTS,
                              SteeplechaseDecoder, SteeplechaseEncoder)
from .pacing_io import (PACING_INITIAL_THRESHOLDS, PACING_INPUTS,
                        PacingDecoder, PacingEncoder)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SPRINT_POLICY = MODELS_DIR / "sprint.json"
SPRINT_EYE_POLICY = MODELS_DIR / "sprint_eye.json"
SPRINT_ENCODER = MODELS_DIR / "sprint_encoder.csv"
SPRINT_ENCODER_INIT = MODELS_DIR / "sprint_encoder_init.csv"
HURDLES_ENCODER_INIT = MODELS_DIR / "hurdles_encoder_init.csv"
HURDLES_POLICY = MODELS_DIR / "hurdles_es.json"
HURDLES_GAIN1_POLICY = MODELS_DIR / "hurdles_es_gain1.json"
SWIMMING_ENCODER_INIT = MODELS_DIR / "swimming_encoder_init.csv"
ARCHERY_ENCODER_INIT = MODELS_DIR / "archery_encoder_init.csv"
ARCHERY_POLICY = MODELS_DIR / "archery.json"
STEEPLECHASE_ENCODER_INIT = MODELS_DIR / "steeplechase_encoder_init.csv"
STEEPLECHASE_POLICY = MODELS_DIR / "steeplechase.json"
PACING_ENCODER_INIT = MODELS_DIR / "pacing_encoder_init.csv"
PACING_POLICY = MODELS_DIR / "pacing.json"

# Event name -> (encoder inputs, encoder class, decoder class) for the band-decoder events.
BAND_EVENTS = {
    "sprint_eye": (SPRINT_EYE_INPUTS, SprintEyeEncoder, SprintEyeDecoder),
    "archery": (ARCHERY_INPUTS, ArcheryEncoder, ArcheryDecoder),
    # Archery with the wind sensed through the FlyVis-calibrated eye (brain/eye.py).
    "archery_eye": (ARCHERY_EYE_INPUTS, ArcheryEyeEncoder, ArcheryDecoder),
    "steeplechase": (STEEPLECHASE_INPUTS, SteeplechaseEncoder, SteeplechaseDecoder),
    "pacing": (PACING_INPUTS, PacingEncoder, PacingDecoder),
    "hurdles": (HURDLES_INPUTS, HurdlesEncoder, HurdlesDecoder),
}


class FlyBrainController(Controller):
    """A connectome-derived neural controller, not a simulated complete fly brain.

    By default it is wired for Sprint; another event passes its own encoder and decoder,
    built on the same circuit.
    """
    def __init__(self, motor_threshold: float = 0.015,
                 encoder_weights: Path | np.ndarray | None = None,
                 circuit: ConnectomeCircuit | None = None, *,
                 encoder=None, decoder: MotorDecoder | None = None, noise: float = 0.0,
                 gain: float | None = None) -> None:
        if not math.isfinite(motor_threshold) or not 0 <= motor_threshold <= 1:
            raise ValueError("Motor threshold must be finite and between 0 and 1.")
        if encoder is not None and encoder_weights is not None:
            raise ValueError("Pass either encoder or encoder_weights, not both.")
        self.circuit = circuit if circuit is not None else load_manc_front_leg_circuit()
        if encoder is not None:
            self.encoder = encoder
        elif encoder_weights is None:
            self.encoder = SensoryEncoder(self.circuit)
        else:
            if not isinstance(encoder_weights, np.ndarray):
                encoder_weights = load_encoder_weights(Path(encoder_weights), self.circuit)
            self.encoder = LearnedSensoryEncoder(self.circuit, encoder_weights)
        if gain is not None:
            self.network = RateNetwork(self.circuit, gain=gain, noise=noise)
        else:
            self.network = RateNetwork(self.circuit, noise=noise)
        self.decoder = decoder if decoder is not None else MotorDecoder(self.circuit, threshold=motor_threshold)
        self.last_summary: dict[str, float] = {}

    @classmethod
    def hurdles(cls, path: Path = HURDLES_ENCODER_INIT, coast_threshold: float = 0.015,
                jump_threshold: float = 0.30, circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        """Untrained Hurdles fly: starting encoder weights, Sprint's threshold for COAST/ACCELERATE."""
        for threshold in (coast_threshold, jump_threshold):
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Motor thresholds must be finite and between 0 and 1.")
        circuit = circuit if circuit is not None else load_manc_front_leg_circuit()
        weights = load_encoder_weights(Path(path), circuit, HURDLES_INPUTS)
        return cls(circuit=circuit, encoder=HurdlesEncoder(circuit, weights),
                   decoder=HurdlesDecoder(circuit, coast_threshold, jump_threshold))

    @classmethod
    def hurdles_trained(cls, path: Path = HURDLES_POLICY,
                        circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        """Trained Hurdles fly from policy JSON."""
        return cls.from_policy(path, circuit)

    @classmethod
    def swimming(cls, path: Path = SWIMMING_ENCODER_INIT, motor_threshold: float = 0.015,
                 circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        """Untrained Swimming fly: starting encoder weights and Sprint's default threshold."""
        if not math.isfinite(motor_threshold) or not 0 <= motor_threshold <= 1:
            raise ValueError("Motor threshold must be finite and between 0 and 1.")
        circuit = circuit if circuit is not None else load_manc_front_leg_circuit()
        weights = load_encoder_weights(Path(path), circuit, SWIMMING_INPUTS)
        return cls(circuit=circuit, encoder=SwimmingEncoder(circuit, weights),
                   decoder=SwimmingDecoder(circuit, threshold=motor_threshold))

    @classmethod
    def band_event(cls, event: str, weights: Path | np.ndarray, thresholds,
                   circuit: ConnectomeCircuit | None = None, noise: float = 0.0,
                   gain: float | None = None) -> FlyBrainController:
        """Build a band-decoder controller from encoder weights (array or CSV) and thresholds."""
        if event not in BAND_EVENTS:
            raise ValueError(f"Unknown band event {event!r}; choose from {sorted(BAND_EVENTS)}")
        inputs, encoder, decoder = BAND_EVENTS[event]
        circuit = circuit if circuit is not None else load_manc_front_leg_circuit()
        if not isinstance(weights, np.ndarray):
            weights = load_encoder_weights(Path(weights), circuit, inputs)
        return cls(circuit=circuit, encoder=encoder(circuit, weights), decoder=decoder(circuit, thresholds),
                   noise=noise, gain=gain)

    @classmethod
    def archery(cls, path: Path = ARCHERY_ENCODER_INIT, thresholds=ARCHERY_INITIAL_THRESHOLDS,
                circuit: ConnectomeCircuit | None = None, noise: float = 0.0) -> FlyBrainController:
        """Untrained Archery fly: starting encoder weights and placeholder thresholds."""
        return cls.band_event("archery", path, thresholds, circuit, noise)

    @classmethod
    def steeplechase(cls, path: Path = STEEPLECHASE_ENCODER_INIT, thresholds=STEEPLECHASE_INITIAL_THRESHOLDS,
                     circuit: ConnectomeCircuit | None = None, noise: float = 0.0) -> FlyBrainController:
        """Untrained Steeplechase fly: starting encoder weights, Hurdles' placeholder thresholds."""
        return cls.band_event("steeplechase", path, thresholds, circuit, noise)

    @classmethod
    def pacing(cls, path: Path = PACING_ENCODER_INIT, thresholds=PACING_INITIAL_THRESHOLDS,
               circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        """Untrained 400 m Pacing fly: starting encoder weights, default threshold."""
        return cls.band_event("pacing", path, thresholds, circuit)

    @classmethod
    def pacing_trained(cls, path: Path = PACING_POLICY,
                       circuit: ConnectomeCircuit | None = None) -> FlyBrainController:
        """Trained 400 m Pacing fly from policy JSON."""
        return cls.from_policy(path, circuit)

    @classmethod
    def from_policy(cls, path: Path, circuit: ConnectomeCircuit | None = None,
                    noise: float = 0.0) -> FlyBrainController:
        """Load a trained band-decoder controller from its policy JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing policy: {path}. Run the event's training script.")
        policy = json.loads(path.read_text())
        if policy.get("version") != 1 or policy.get("event") not in BAND_EVENTS:
            raise ValueError(f"{path}: expected a version 1 band policy for {sorted(BAND_EVENTS)}.")
        gain = float(policy["gain"]) if "gain" in policy else None
        return cls.band_event(policy["event"], path.parent / policy["encoder"], policy["thresholds"], circuit,
                              noise, gain=gain)

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
        """Silence the network; a seed also reseeds its noise (only drawn when noise > 0)."""
        self.network.reset(seed)
        self.last_summary = {}

    def act(self, observation: Mapping[str, float]) -> str:
        stimulus = self.encoder.encode(dict(observation))
        activity = self.network.step(stimulus)
        action, motor = self.decoder.decode(activity)
        population_indices = {
            "sensory": self.encoder.indices,
            "intrinsic": self.circuit.indices("intrinsic"),
            "extensor": self.circuit.indices("motor_extensor"),
            "flexor": self.circuit.indices("motor_flexor"),
        }
        self.last_summary = {
            "sensory_mean": float(activity[self.encoder.indices].mean()),
            "intrinsic_mean": float(activity[self.circuit.indices("intrinsic")].mean()),
            "motor_extensor": motor["extensor"],
            "motor_flexor": motor["flexor"],
            "motor_difference": motor["difference"],
            **{
                f"{group}_over_095_pct": float(np.mean(activity[indices] > 0.95) * 100.0)
                for group, indices in population_indices.items()
            },
        }
        return action
