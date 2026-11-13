"""Evolution-strategy training for band-decoder events (Archery, Steeplechase).

What learns: the sensory encoder weights E and the decoder band thresholds. What stays
frozen: the MANC connectivity, the rate dynamics (gain, leak, substeps) and the motor
populations. Same split as the Always-Accelerate experiment, but the events are not
differentiable (discrete actions, jumps, scoring rings), so instead of BPTT we use a
black-box evolution strategy:

    OpenAI-ES (Salimans et al. 2017): antithetic Gaussian perturbations of the parameter
    vector, centred-rank fitness shaping, and Adam on the estimated gradient.

To make a generation affordable, every member of the population (and every training
seed) is simulated together by BatchedRateNetwork, a dense batched copy of RateNetwork.
Final results are always re-measured with the ordinary FlyBrainController.
"""
from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .connectome import ConnectomeCircuit
from .dynamics import RateNetwork


class BatchedRateNetwork:
    """RateNetwork for a batch of independent states; same update, dense weights.

    noise and seed work as in RateNetwork; every batch row gets its own noise draws.
    """

    def __init__(self, circuit: ConnectomeCircuit, batch: int, leak: float = 0.35, gain: float = 3.0,
                 noise: float = 0.0, seed: int | None = None, member_weights_t: np.ndarray | None = None):
        """member_weights_t, if given, has shape (members, n, n): each member's own transposed
        connection matrix, shared by batch // members consecutive rows (member-major)."""
        reference = RateNetwork(circuit, leak=leak, gain=gain, noise=noise)
        self.leak, self.gain, self.noise = reference.leak, reference.gain, reference.noise
        self.weights_t = circuit.weights.toarray().T  # state @ W.T == (W @ state.T).T
        self.member_weights_t = member_weights_t
        if member_weights_t is not None:
            n = circuit.size
            if member_weights_t.ndim != 3 or member_weights_t.shape[1:] != (n, n) or batch % len(member_weights_t):
                raise ValueError(f"member_weights_t must be (members, {n}, {n}) with batch a multiple of members.")
        self.silenced = reference.silenced
        self.state = np.zeros((batch, circuit.size))
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, stimulus: np.ndarray, substeps: int = 2) -> np.ndarray:
        if stimulus.shape != self.state.shape:
            raise ValueError(f"Expected stimulus shape {self.state.shape}, got {stimulus.shape}")
        for _ in range(substeps):
            if self.member_weights_t is None:
                recurrent = self.state @ self.weights_t
            else:
                members = len(self.member_weights_t)
                grouped = self.state.reshape(members, -1, self.state.shape[1])
                recurrent = np.matmul(grouped, self.member_weights_t).reshape(self.state.shape)
            drive = self.gain * recurrent + stimulus
            if self.noise:
                drive = drive + self.rng.normal(0.0, self.noise, drive.shape)
            target = np.tanh(drive)
            self.state += self.leak * (target - self.state)
            self.state[:, self.silenced] = 0.0
        return self.state


def batched_rollout(circuit: ConnectomeCircuit, make_event: Callable[[], object], seeds: Sequence[int],
                    encoder, actions: Sequence[str], weights: np.ndarray, thresholds: np.ndarray,
                    observer: Callable[[list, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
                    noise: float = 0.0, noise_seed: int | None = None, gain: float = 3.0,
                    member_weights_t: np.ndarray | None = None) -> list[list]:
    """Run every (member, seed) pair to the end and return the finished events.

    weights has shape (members, n_sensory, n_inputs) and thresholds (members, n_actions - 1),
    sorted per member. encoder only supplies features() and the tonic drive.
    observer, if given, sees every step before it is taken: (observations, difference,
    env_thresholds, active), environments ordered member-major (env = member * n_seeds + seed).
    noise and noise_seed go to BatchedRateNetwork (off by default). gain sets the neuron gain
    (a modeling choice). member_weights_t, if given, gives each member its own connection
    strengths (see connection_matrices).
    Returns events[member][seed_index].
    """
    members, seeds = len(weights), list(seeds)
    batch = members * len(seeds)
    member_of = np.repeat(np.arange(members), len(seeds))
    envs = [make_event() for _ in range(batch)]
    observations = [env.reset(seeds[i % len(seeds)]) for i, env in enumerate(envs)]
    network = BatchedRateNetwork(circuit, batch, gain=gain, noise=noise, seed=noise_seed,
                                 member_weights_t=member_weights_t)
    sensory = circuit.indices("sensory")
    extensor, flexor = circuit.indices("motor_extensor"), circuit.indices("motor_flexor")
    env_weights, env_thresholds = weights[member_of], thresholds[member_of]
    stimulus = np.zeros((batch, circuit.size))
    active = np.ones(batch, dtype=bool)
    while active.any():
        features = np.array([encoder.features(dict(o)) for o in observations])
        stimulus[:, sensory] = encoder.tonic + np.einsum("bsk,bk->bs", env_weights, features)
        activity = network.step(stimulus)
        difference = activity[:, extensor].mean(axis=1) - activity[:, flexor].mean(axis=1)
        # Same rule as BandDecoder: band = number of thresholds strictly below the difference.
        bands = (env_thresholds < difference[:, None]).sum(axis=1)
        if observer is not None:
            observer(observations, difference, env_thresholds, active)
        for i in np.flatnonzero(active):
            observations[i] = envs[i].step(actions[bands[i]])
            if envs[i].is_finished():
                active[i] = False
    return [envs[m * len(seeds):(m + 1) * len(seeds)] for m in range(members)]


def teacher_forced_differences(circuit: ConnectomeCircuit, features: np.ndarray, tonic: float,
                               weights: np.ndarray, noise: float = 0.0,
                               noise_seed: int | None = None, gain: float = 3.0) -> np.ndarray:
    """extensor - flexor for every candidate along fixed observation sequences.

    features has shape (episodes, steps, n_inputs): recorded episodes, padded to one length.
    weights has shape (members, n_sensory, n_inputs). Each (member, episode) pair starts
    from a silent network, exactly like a controller reset. Returns (members, episodes, steps).
    """
    members, (episodes, steps, _) = len(weights), features.shape
    network = BatchedRateNetwork(circuit, members * episodes, gain=gain, noise=noise, seed=noise_seed)
    sensory = circuit.indices("sensory")
    extensor, flexor = circuit.indices("motor_extensor"), circuit.indices("motor_flexor")
    stimulus = np.zeros((members * episodes, circuit.size))
    out = np.empty((members, episodes, steps))
    for t in range(steps):
        drive = tonic + np.einsum("msk,ek->mes", weights, features[:, t])
        stimulus[:, sensory] = drive.reshape(members * episodes, -1)
        activity = network.step(stimulus)
        out[:, :, t] = (activity[:, extensor].mean(axis=1) - activity[:, flexor].mean(axis=1)).reshape(members, episodes)
    return out


class ConnectionDrift:
    """Trainable strengths for the existing MANC connections, kept inside a drift limit.

    Each connection gets a multiplier on its measured strength,
    low + (high - low) * sigmoid(z), so it can never leave [low, high], switch off or flip
    sign. z starts where the multiplier is exactly 1 (the real fly). Which neurons connect
    never changes: only the strengths of connections that already exist in MANC.
    """

    def __init__(self, circuit: ConnectomeCircuit, low: float, high: float):
        if not 0 < low < 1 < high:
            raise ValueError("Drift limits must satisfy 0 < low < 1 < high.")
        self.low, self.high = float(low), float(high)
        w = circuit.weights.tocsr()
        self.size = w.nnz
        self.data = w.data.astype(float)
        self.post = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))  # CSR rows are [post, pre]
        self.pre = w.indices.copy()
        self.n = w.shape[0]
        self.z0 = float(np.log((1 - self.low) / (self.high - 1)))  # sigmoid(z0) puts the multiplier at 1

    def multipliers(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z, float), -50.0, 50.0)  # far past either limit; avoids exp overflow
        return self.low + (self.high - self.low) / (1.0 + np.exp(-z))

    def matrices(self, z: np.ndarray) -> np.ndarray:
        """(members, n, n) transposed connection matrices for BatchedRateNetwork."""
        multipliers = self.multipliers(np.atleast_2d(z))
        out = np.zeros((len(multipliers), self.n, self.n))
        out[:, self.pre, self.post] = self.data * multipliers
        return out


def save_connections(path: Path, circuit: ConnectomeCircuit, multipliers: np.ndarray) -> None:
    """One row per existing MANC connection: pre and post neuron ids, measured strength,
    trained multiplier. The MANC data files themselves are never written."""
    w = circuit.weights.tocsr()
    post = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    ids = circuit.neuron_ids
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pre", "post", "manc_weight", "multiplier"])
        for k in range(w.nnz):
            writer.writerow([ids[w.indices[k]], ids[post[k]], f"{w.data[k]:.10g}", f"{multipliers[k]:.10g}"])


def load_connections(path: Path, circuit: ConnectomeCircuit) -> np.ndarray:
    """Multipliers from save_connections, checked against the circuit's own edge list."""
    w = circuit.weights.tocsr()
    post = np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))
    ids = circuit.neuron_ids
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != w.nnz:
        raise ValueError(f"{path}: expected {w.nnz} connections, got {len(rows)}.")
    for k, row in enumerate(rows):
        if (str(row["pre"]), str(row["post"])) != (str(ids[w.indices[k]]), str(ids[post[k]])):
            raise ValueError(f"{path}: row {k} is not the circuit's connection {ids[w.indices[k]]}->{ids[post[k]]}.")
    return np.array([float(row["multiplier"]) for row in rows])


@dataclass
class ParameterLayout:
    """Packs encoder weights and band thresholds into one flat vector for the ES."""
    n_sensory: int
    n_inputs: int
    n_thresholds: int
    min_gap: float = 1e-6

    @property
    def size(self) -> int:
        return self.n_sensory * self.n_inputs + self.n_thresholds

    def pack(self, weights: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
        return np.concatenate([np.asarray(weights, float).ravel(), np.asarray(thresholds, float)])

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, float)
        weights = theta[..., :self.n_sensory * self.n_inputs].reshape(*theta.shape[:-1], self.n_sensory, self.n_inputs)
        thresholds = np.sort(np.clip(theta[..., self.n_sensory * self.n_inputs:], -1.0, 1.0), axis=-1)
        return weights, thresholds

    def decoder_thresholds(self, theta: np.ndarray) -> list[float]:
        """Thresholds for BandDecoder: sorted, in [-1, 1] and strictly increasing."""
        _, thresholds = self.unpack(theta)
        out = [float(thresholds[0])]
        for t in thresholds[1:]:
            out.append(max(float(t), out[-1] + self.min_gap))
        if out[-1] > 1.0:
            raise ValueError("Thresholds too close to 1 to keep them strictly increasing.")
        return out

    def sigma(self, weight_sigma: float, threshold_sigma: float) -> np.ndarray:
        return np.concatenate([np.full(self.n_sensory * self.n_inputs, weight_sigma),
                               np.full(self.n_thresholds, threshold_sigma)])


@dataclass
class EvolutionResult:
    best_theta: np.ndarray
    best_fitness: float
    history: list[dict] = field(default_factory=list)


def centred_ranks(values: np.ndarray) -> np.ndarray:
    """Fitness shaping: ranks mapped to [-0.5, 0.5]; robust to outliers and plateaus."""
    ranks = np.empty(len(values))
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def evolve(theta: np.ndarray, fitness: Callable[[np.ndarray], np.ndarray], *, generations: int,
           pairs: int, sigma: np.ndarray | float, lr: np.ndarray | float, seed: int = 0, weight_decay: float = 0.0,
           log: Callable[[str], None] = print) -> EvolutionResult:
    """OpenAI-ES with antithetic sampling, centred ranks and Adam. Higher fitness is better.

    fitness maps a (k, size) array of parameter vectors to k fitness values. Each
    generation evaluates the current mean plus 2 * pairs perturbed copies; the best
    vector ever evaluated is returned, so a noisy last step cannot lose a good policy.
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, float), theta.shape)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_theta, best_fitness = theta.copy(), -np.inf
    history = []
    for generation in range(1, generations + 1):
        noise = rng.standard_normal((pairs, theta.size))
        candidates = np.concatenate([theta[None], theta + sigma * noise, theta - sigma * noise])
        values = np.asarray(fitness(candidates), dtype=float)
        top = int(np.argmax(values))
        if values[top] > best_fitness:
            best_fitness, best_theta = float(values[top]), candidates[top].copy()
        shaped = centred_ranks(values[1:])
        gradient = ((shaped[:pairs] - shaped[pairs:]) @ noise) / (2 * pairs * sigma) - weight_decay * theta
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        step = (m / (1 - beta1**generation)) / (np.sqrt(v / (1 - beta2**generation)) + eps)
        theta = theta + lr * step  # ascent: higher fitness is better
        row = {"generation": generation, "mean_fitness": float(values[0]),
               "population_best": float(values[top]), "population_median": float(np.median(values[1:])),
               "best_so_far": best_fitness}
        history.append(row)
        log(f"gen {generation:4d}  mean {row['mean_fitness']:9.3f}  pop best {row['population_best']:9.3f}  "
            f"median {row['population_median']:9.3f}  best so far {best_fitness:9.3f}")
    return EvolutionResult(best_theta, best_fitness, history)
