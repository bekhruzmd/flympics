"""Artificial task-observation to biological-population stimulation adapter."""
from __future__ import annotations

from pathlib import Path
import csv
import math
import os

import numpy as np

from .connectome import ConnectomeCircuit

# Column order of the learned encoder matrix E (one row per sensory neuron).
ENCODER_INPUTS = ("distance", "velocity")
ENCODER_FIELDS = ["pre", "post", "weight"]


class SensoryEncoder:
    """Artificially stimulates selected front-leg proprioceptor nodes.

    Distance and velocity are virtual-sprint variables, not measurements known to a
    fly neuron. This component is intentionally replaceable.
    """
    def __init__(self, circuit: ConnectomeCircuit, sprint_distance: float = 100.0):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.sprint_distance = sprint_distance

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        distance_fraction = np.clip(observation["distance_to_finish"] / self.sprint_distance, 0.0, 1.0)
        velocity_fraction = np.clip(observation["velocity"] / 8.0, 0.0, 1.0)
        # Artificial tonic/proprioceptive-like drive, deliberately not a biological claim.
        level = 0.45 + 0.90 * distance_fraction + 0.25 * velocity_fraction
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = level
        return vector


class LearnedSensoryEncoder:
    """SensoryEncoder with its own distance/velocity weight per sensory neuron.

    weights has shape (n_sensory, 2): column 0 scales distance, column 1 velocity.
    The 0.45 tonic offset and the input normalization stay fixed; only weights learn.
    """
    tonic = 0.45

    def __init__(self, circuit: ConnectomeCircuit, weights: np.ndarray, sprint_distance: float = 100.0):
        self.size = circuit.size
        self.indices = circuit.indices("sensory")
        self.sprint_distance = sprint_distance
        self.weights = np.array(weights, dtype=float)
        if self.weights.shape != (len(self.indices), len(ENCODER_INPUTS)):
            raise ValueError(f"Expected encoder shape {(len(self.indices), len(ENCODER_INPUTS))}, "
                             f"got {self.weights.shape}")

    def features(self, observation: dict[str, float]) -> np.ndarray:
        return np.array([
            np.clip(observation["distance_to_finish"] / self.sprint_distance, 0.0, 1.0),
            np.clip(observation["velocity"] / 8.0, 0.0, 1.0),
        ])

    def encode(self, observation: dict[str, float]) -> np.ndarray:
        return self.encode_features(self.features(observation))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Stimulus from already-normalized [distance, velocity]; training calls this directly."""
        distance_fraction, velocity_fraction = features
        # Same operation order as SensoryEncoder, so equal weights give bit-identical stimuli.
        level = self.tonic + self.weights[:, 0] * distance_fraction + self.weights[:, 1] * velocity_fraction
        vector = np.zeros(self.size, dtype=float)
        vector[self.indices] = level
        return vector


def initial_encoder_weights(circuit: ConnectomeCircuit) -> np.ndarray:
    """Per-neuron copy of SensoryEncoder's distance/velocity coefficients (0.90, 0.25)."""
    return np.tile([0.90, 0.25], (len(circuit.indices("sensory")), 1))


def load_encoder_weights(path: Path, circuit: ConnectomeCircuit) -> np.ndarray:
    """Read an input->sensory edge CSV into E with shape (n_sensory, 2).

    Rows are matched by MANC id, so file order does not matter; every sensory
    neuron must have exactly one distance and one velocity weight.
    """
    sensory = circuit.indices("sensory")
    row_of = {circuit.neuron_ids[i]: r for r, i in enumerate(sensory)}
    col_of = {name: c for c, name in enumerate(ENCODER_INPUTS)}
    weights = np.zeros((len(sensory), len(ENCODER_INPUTS)))
    seen: set[tuple[int, int]] = set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ENCODER_FIELDS:
            raise ValueError(f"{path}: expected columns {ENCODER_FIELDS}, got {reader.fieldnames}")
        for line, edge in enumerate(reader, start=2):
            if edge["pre"] not in col_of:
                raise ValueError(f"{path}:{line}: unknown input {edge['pre']!r}; choose from {ENCODER_INPUTS}")
            if edge["post"] not in row_of:
                raise ValueError(f"{path}:{line}: {edge['post']!r} is not a sensory neuron")
            weight = float(edge["weight"])
            if not math.isfinite(weight):
                raise ValueError(f"{path}:{line}: weight must be finite, got {edge['weight']!r}")
            key = (row_of[edge["post"]], col_of[edge["pre"]])
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate edge {edge['pre']} -> {edge['post']}")
            seen.add(key)
            weights[key] = weight
    if len(seen) != weights.size:
        missing = [f"{ENCODER_INPUTS[c]} -> {circuit.neuron_ids[sensory[r]]}"
                   for r in range(len(sensory)) for c in range(len(ENCODER_INPUTS)) if (r, c) not in seen]
        raise ValueError(f"{path}: missing {len(missing)} edge(s), e.g. {missing[:3]}")
    return weights


def save_encoder_weights(path: Path, circuit: ConnectomeCircuit, weights: np.ndarray) -> None:
    """Write E as one row per (input, sensory neuron) edge; repr() keeps floats exact."""
    sensory = circuit.indices("sensory")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(sensory), len(ENCODER_INPUTS)):
        raise ValueError(f"Expected encoder shape {(len(sensory), len(ENCODER_INPUTS))}, got {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("Encoder weights must be finite.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(ENCODER_FIELDS)
        for r, i in enumerate(sensory):
            for c, name in enumerate(ENCODER_INPUTS):
                writer.writerow([name, circuit.neuron_ids[i], repr(float(weights[r, c]))])
    os.replace(temporary, path)
