"""Shared evolution-strategy training loop for the Archery and Steeplechase flies.

See brain/evolution.py for what learns (encoder weights E, band thresholds) and what
stays frozen (connectome, dynamics, motor populations).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from brain.connectome import ConnectomeCircuit, load_manc_front_leg_circuit
from brain.controller import FlyBrainController
from brain.evolution import ParameterLayout, batched_rollout, evolve, teacher_forced_differences
from brain.sensory import save_encoder_weights

HISTORY_FIELDS = ["stage", "generation", "mean_fitness", "population_best", "population_median", "best_so_far"]


@dataclass
class EventSpec:
    name: str                                   # "archery" or "steeplechase"
    make_event: Callable[[], object]
    encoder_class: type
    inputs: Sequence[str]
    actions: Sequence[str]
    initial_weights: Callable[[ConnectomeCircuit], np.ndarray]
    initial_thresholds: Sequence[float]
    train_seeds: Sequence[int]
    fitness: Callable[[list], float]           # finished events for one member -> fitness
    teacher: Callable[[dict], int]              # observation -> band index the rule baseline picks
    evaluate: Callable[[FlyBrainController], dict]  # real-controller report
    policy: Path
    encoder: Path


def fingerprint(circuit: ConnectomeCircuit) -> str:
    """Hash of the frozen connectome; training must not change it."""
    w = circuit.weights
    return hashlib.sha256(w.data.tobytes() + w.indices.tobytes() + w.indptr.tobytes()).hexdigest()


def _log(line: str) -> None:
    print(line, flush=True)


def demonstrations(spec: EventSpec, encoder, epsilon: float = 0.3, repeats: int = 3, seed: int = 0):
    """States from noisy rule rollouts, labelled with the rule's band.

    With probability epsilon a random action is taken instead of the rule's, so the data
    also covers mistakes and how to recover from them. Returns padded (episodes, steps,
    n_inputs) features, (episodes, steps) band labels and a validity mask.
    """
    rng = np.random.default_rng(seed)
    episodes = []
    for repeat in range(repeats):
        for train_seed in spec.train_seeds:
            event = spec.make_event()
            observation = event.reset(train_seed)
            rows = []
            while not event.is_finished():
                band = spec.teacher(dict(observation))
                rows.append((encoder.features(dict(observation)), band))
                act = band if rng.random() >= epsilon * (repeat > 0) else int(rng.integers(len(spec.actions)))
                observation = event.step(spec.actions[act])
            episodes.append(rows)
    steps = max(len(rows) for rows in episodes)
    features = np.zeros((len(episodes), steps, len(spec.inputs)))
    labels = np.zeros((len(episodes), steps), dtype=int)
    mask = np.zeros((len(episodes), steps), dtype=bool)
    for e, rows in enumerate(episodes):
        features[e, :len(rows)] = [f for f, _ in rows]
        labels[e, :len(rows)] = [b for _, b in rows]
        mask[e, :len(rows)] = True
    return features, labels, mask


def fit_band_thresholds(difference: np.ndarray, labels: np.ndarray, weight: np.ndarray,
                        cuts: int = 200) -> tuple[float, tuple[float, float]]:
    """Best two ordered thresholds for a 3-band decoder, by weighted agreement with labels.

    Sorts the differences and tries cut points at `cuts` quantiles; a threshold sits halfway
    between the two neighbouring differences. Returns (agreement, (low, high)).
    """
    order = np.argsort(difference, kind="stable")
    d, labels, weight = difference[order], labels[order], weight[order]
    cum = np.stack([np.concatenate([[0.0], np.cumsum(weight * (labels == k))]) for k in range(3)])
    idx = np.unique(np.linspace(0, len(d), cuts + 1).astype(int))
    c0, c1, c2 = cum[0][idx], cum[1][idx], cum[2][idx]
    score = c0[:, None] + (c1[None, :] - c1[:, None]) + (cum[2][-1] - c2[None, :])
    score = np.where(idx[:, None] <= idx[None, :], score, -np.inf)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    padded = np.concatenate([[d[0] - 1e-3], d, [d[-1] + 1e-3]])

    def threshold(cut: int) -> float:
        # Difference values at or below the threshold go to the lower band.
        return float(np.clip((padded[cut] + padded[cut + 1]) / 2, -1.0, 1.0))

    low, high = threshold(idx[i]), threshold(idx[j])
    if high <= low:
        high = min(1.0, low + 1e-6)
    return float(score[i, j]), (low, high)


def train(spec: EventSpec, generations: int, pairs: int, weight_sigma: float, threshold_sigma: float,
          lr: float, seed: int, imitation_generations: int = 0, start_sigma: float = 0.5,
          log=_log) -> tuple[dict, list[dict]]:
    circuit = load_manc_front_leg_circuit()
    frozen = fingerprint(circuit)
    n_sensory = len(circuit.indices("sensory"))
    layout = ParameterLayout(n_sensory, len(spec.inputs), len(spec.actions) - 1)
    initial_weights = spec.initial_weights(circuit)
    theta0 = layout.pack(initial_weights, spec.initial_thresholds)
    feature_encoder = spec.encoder_class(circuit, initial_weights)

    def fitness(candidates: np.ndarray) -> np.ndarray:
        weights, thresholds = layout.unpack(candidates)
        events = batched_rollout(circuit, spec.make_event, spec.train_seeds, feature_encoder,
                                 spec.actions, weights, thresholds)
        return np.array([spec.fitness(member) for member in events])

    n_weights = n_sensory * len(spec.inputs)
    if imitation_generations:
        features, labels, mask = demonstrations(spec, feature_encoder)
        class_weight = np.zeros(mask.shape)
        for band in np.unique(labels[mask]):
            chosen = mask & (labels == band)
            class_weight[chosen] = 1.0 / chosen.sum()
        class_weight /= class_weight.sum()

    def imitation_fit(flat_weights: np.ndarray) -> list[tuple[float, tuple[float, float]]]:
        weights = flat_weights.reshape(len(flat_weights), n_sensory, len(spec.inputs))
        differences = teacher_forced_differences(circuit, features, feature_encoder.tonic, weights)
        return [fit_band_thresholds(differences[m][mask], labels[mask], class_weight[mask])
                for m in range(len(weights))]

    def imitation(flat_weights: np.ndarray) -> np.ndarray:
        """Class-balanced agreement with the rule on recorded states, with each candidate's
        band thresholds fitted exactly."""
        return np.array([score for score, _ in imitation_fit(flat_weights)])

    sigma = layout.sigma(weight_sigma, threshold_sigma)
    step = lr * sigma / weight_sigma
    started = time.perf_counter()
    history = []
    if imitation_generations:
        log("Stage 1: imitate the rule baseline (random-search start, then ES on the encoder)")
        rng = np.random.default_rng(seed)
        starts = np.concatenate([theta0[None, :n_weights],
                                 rng.normal(0.0, start_sigma, (2 * pairs, n_weights))])
        scores = imitation(starts)
        log(f"random start: initial weights {scores[0]:.3f}, best of {len(starts)} {scores.max():.3f}")
        warm = evolve(starts[int(np.argmax(scores))], imitation, generations=imitation_generations,
                      pairs=pairs, sigma=weight_sigma, lr=lr, seed=seed, log=log)
        history += [{"stage": "imitation", **row} for row in warm.history]
        _, thresholds = imitation_fit(warm.best_theta[None])[0]
        theta0 = layout.pack(warm.best_theta.reshape(n_sensory, len(spec.inputs)), thresholds)
        log("Stage 2: optimise the event outcome")
    result = evolve(theta0, fitness, generations=generations, pairs=pairs, sigma=sigma,
                    lr=step, seed=seed + 1, log=log)
    history += [{"stage": "outcome", **row} for row in result.history]
    elapsed = time.perf_counter() - started
    if fingerprint(circuit) != frozen:
        raise RuntimeError("The frozen connectome changed during training.")

    weights, _ = layout.unpack(result.best_theta)
    thresholds = layout.decoder_thresholds(result.best_theta)
    trained = FlyBrainController.band_event(spec.name, weights, thresholds, circuit)
    untrained = FlyBrainController.band_event(spec.name, initial_weights, list(spec.initial_thresholds), circuit)
    save_encoder_weights(spec.encoder, circuit, weights, spec.inputs)
    policy = {
        "version": 1,
        "event": spec.name,
        "method": "OpenAI-ES (antithetic sampling, centred-rank fitness, Adam) on encoder weights "
                  "and band thresholds: teacher-forced imitation of the rule baseline (thresholds fitted "
                  "exactly per candidate), then the event outcome; "
                  "connectome and dynamics frozen",
        "encoder": spec.encoder.name,
        "inputs": list(spec.inputs),
        "actions": list(spec.actions),
        "thresholds": thresholds,
        "hyperparameters": {"generations": generations, "pairs": pairs, "population": 2 * pairs + 1,
                            "weight_sigma": weight_sigma, "threshold_sigma": threshold_sigma,
                            "lr": lr, "seed": seed, "train_seeds": list(spec.train_seeds),
                            "imitation_generations": imitation_generations, "start_sigma": start_sigma},
        "train_fitness": result.best_fitness,
        "training_seconds": round(elapsed, 1),
        "trained": spec.evaluate(trained),
        "untrained": spec.evaluate(untrained),
    }
    spec.policy.parent.mkdir(parents=True, exist_ok=True)
    spec.policy.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    return policy, history


def save_history(history: list[dict], directory: Path, title: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    import matplotlib.pyplot as plt
    gens = list(range(1, len(history) + 1))
    switch = next((i for i, row in enumerate(history) if row["stage"] == "outcome"), 0)
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.plot(gens, [row["population_median"] for row in history], color="tab:gray", label="population median")
    axis.plot(gens, [row["mean_fitness"] for row in history], color="tab:blue", label="mean parameters")
    axis.plot(gens, [row["best_so_far"] for row in history], color="tab:green", label="best so far")
    if switch:
        axis.axvline(switch + 0.5, color="tab:orange", linestyle="--", label="imitation -> outcome")
    axis.set(xlabel="generation (both stages)", ylabel="fitness (stage 1: agreement)", title=title)
    axis.legend()
    fig.savefig(directory / "history.png", dpi=150)
    plt.close(fig)


def cli(spec: EventSpec, description: str, *, generations: int, imitation_generations: int, pairs: int,
        weight_sigma: float, threshold_sigma: float, lr: float) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--generations", type=int, default=generations)
    parser.add_argument("--imitation-generations", type=int, default=imitation_generations,
                        help="Stage-1 generations imitating the rule baseline (0 skips it)")
    parser.add_argument("--pairs", type=int, default=pairs, help="Antithetic pairs per generation")
    parser.add_argument("--weight-sigma", type=float, default=weight_sigma)
    parser.add_argument("--threshold-sigma", type=float, default=threshold_sigma)
    parser.add_argument("--lr", type=float, default=lr, help="Adam step for encoder weights")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the ES noise")
    parser.add_argument("--policy", type=Path, default=spec.policy)
    parser.add_argument("--results-dir", type=Path, default=Path("results") / f"{spec.name}_training")
    args = parser.parse_args()
    spec.policy = args.policy
    spec.encoder = args.policy.with_name(f"{args.policy.stem}_encoder.csv")
    policy, history = train(spec, args.generations, args.pairs, args.weight_sigma,
                            args.threshold_sigma, args.lr, args.seed, args.imitation_generations)
    save_history(history, args.results_dir, f"{spec.name.title()}: evolution-strategy training")
    print(f"Trained in {policy['training_seconds']} s. Trained: {policy['trained']}")
    print(f"Untrained: {policy['untrained']}")
    print(f"Policy: {spec.policy}; encoder: {spec.encoder}; history: {args.results_dir / 'history.csv'}")
