"""Reward-modulated learning ("food for good shots") on the real connection strengths.

One fly learns from its own experience, arrow by arrow. It is node perturbation
(Fiete & Seung 2006), a reinforcement-learning rule often used as a model of
dopamine-gated plasticity:

1. Every neuron's drive gets a little random jitter, so the fly explores.
2. Each connection keeps an eligibility trace: how much the jitter at its target neuron
   coincided with activity at its source neuron while this arrow was being aimed.
3. When the arrow lands, food arrives (more near the centre, a bonus for a bullseye).
   Each connection changes by learning_rate * (food - usual food) * its trace.

What can change: the strength of each existing MANC connection, within a drift limit
(a multiplier on its measured strength, never below 0.5, so no connection switches off or
flips sign), and the input weights. What stays fixed: which neurons connect, the neuron
model, the motor readout thresholds, and the MANC data on disk. The reward is a modeled
signal broadcast to every connection; this circuit has no dopamine neurons of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from .connectome import ConnectomeCircuit

DRIFT_LIMITS = {"20": (0.8, 1.2), "50": (0.5, 1.5), "2x": (0.5, 2.0)}


def tuned_circuit(circuit: ConnectomeCircuit, multipliers: np.ndarray) -> ConnectomeCircuit:
    """A copy of the circuit with each connection's strength scaled; same edges, same order."""
    w = circuit.weights
    multipliers = np.asarray(multipliers, dtype=float)
    if multipliers.shape != w.data.shape:
        raise ValueError(f"Expected {w.data.shape[0]} multipliers, got {multipliers.shape}")
    if np.any(multipliers <= 0) or not np.all(np.isfinite(multipliers)):
        raise ValueError("Multipliers must be finite and positive, so no connection flips sign.")
    tuned = sparse.csr_matrix((w.data * multipliers, w.indices.copy(), w.indptr.copy()), shape=w.shape)
    return replace(circuit, weights=tuned)


def food(points: int | None, bullseye_bonus: float = 1.0) -> float:
    """Food for one arrow: points / 10, plus a bonus for a bullseye; nothing for a timeout."""
    if points is None:
        return 0.0
    return points / 10 + (bullseye_bonus if points == 10 else 0.0)


@dataclass
class LearningSettings:
    drift: str = "50"
    noise: float = 0.03            # exploration jitter on every neuron's drive
    weight_rate: float = 0.002     # learning rate for connection strengths
    input_rate: float = 0.02       # learning rate for input weights
    baseline_rate: float = 0.05    # how fast "usual food" tracks recent food
    explore_motor_only: bool = True  # jitter only the motor neurons that pick the action
    gain: float = 3.0
    leak: float = 0.35
    substeps: int = 2


class RewardModulatedFly:
    """The archery fly's circuit, learning online from food; act() returns a band index."""

    def __init__(self, circuit: ConnectomeCircuit, encoder, thresholds, settings: LearningSettings, seed: int = 0):
        self.circuit, self.encoder, self.settings = circuit, encoder, settings
        self.low, self.high = DRIFT_LIMITS[settings.drift]
        w = circuit.weights.tocoo()
        self.post, self.pre = w.row.astype(int), w.col.astype(int)
        self.base = w.data.astype(float).copy()
        self.strength = self.base.copy()
        # Bounds on each strength; ordered so they also work for inhibitory (negative) weights.
        ends = np.stack([self.base * self.low, self.base * self.high])
        self.lower, self.upper = ends.min(axis=0), ends.max(axis=0)
        self.matrix = circuit.weights.copy()  # same sparsity and [post, pre] order as the data
        self.inputs = np.array(encoder.weights, dtype=float)
        self.sensory = circuit.indices("sensory")
        self.extensor, self.flexor = circuit.indices("motor_extensor"), circuit.indices("motor_flexor")
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.baseline: float | None = None
        self.expected: dict[int, float] = {}  # critic: usual food for each wind strength
        noisy = np.concatenate([self.extensor, self.flexor]) if settings.explore_motor_only else np.arange(circuit.size)
        self.noisy = np.zeros(circuit.size, dtype=bool); self.noisy[noisy] = True
        self.explore = True
        self.reset()

    @property
    def multipliers(self) -> np.ndarray:
        # A connection stored with zero strength stays zero; report it as unchanged (1x).
        return np.divide(self.strength, self.base, out=np.ones_like(self.base), where=self.base != 0)

    def reset(self) -> None:
        self.state = np.zeros(self.circuit.size)
        self.clear_traces()

    def clear_traces(self) -> None:
        self.trace = np.zeros_like(self.strength)
        self.input_trace = np.zeros_like(self.inputs)

    def act(self, observation: dict) -> int:
        s = self.settings
        features = self.encoder.features(dict(observation))
        stimulus = np.zeros(self.circuit.size)
        stimulus[self.sensory] = self.encoder.tonic + self.inputs @ features
        self.matrix.data = self.strength  # CSR data order matches tocoo() order
        matrix = self.matrix
        for _ in range(s.substeps):
            jitter = self.rng.normal(0.0, s.noise, self.circuit.size) * self.noisy if self.explore else 0.0
            drive = s.gain * (matrix @ self.state) + stimulus + jitter
            if self.explore:
                self.trace += jitter[self.post] * self.state[self.pre]
                self.input_trace += np.outer(jitter[self.sensory], features)
            self.state += s.leak * (np.tanh(drive) - self.state)
        difference = self.state[self.extensor].mean() - self.state[self.flexor].mean()
        return int((self.thresholds < difference).sum())

    def reward(self, value: float, wind: float | None = None) -> float:
        """Learn from one arrow's food; returns the surprise (food minus usual food).

        With a wind, "usual food" is a critic's running estimate for that wind strength, so
        a hard gust that costs points no matter what is not blamed on the connections.
        """
        s = self.settings
        key = None if wind is None else int(round(abs(wind) / 0.5))
        usual = self.expected.get(key, self.baseline if self.baseline is not None else value)
        surprise = value - usual
        if key is not None:
            self.expected[key] = usual + s.baseline_rate * surprise
        self.baseline = value if self.baseline is None else self.baseline + s.baseline_rate * (value - self.baseline)
        self.strength = np.clip(self.strength + s.weight_rate * surprise * self.trace, self.lower, self.upper)
        self.inputs += s.input_rate * surprise * self.input_trace
        self.clear_traces()
        return surprise
