"""Teach the fly to always accelerate by training only its 24x2 sensory encoder weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain.controller import SPRINT_ENCODER, SPRINT_ENCODER_INIT, FlyBrainController
from brain.dynamics import RateNetwork
from brain.motor import MotorDecoder
from brain.sensory import LearnedSensoryEncoder, load_encoder_weights, save_encoder_weights
from events.sprint import ACCELERATE, Sprint

SUBSTEPS = 2  # RateNetwork.step default, used by FlyBrainController.act
# The teacher: with constant force and drag only, ACCELERATE is the best action in every sprint state.
MARGIN = 0.035  # required clearance past the decoder threshold: 0.015 + 0.035 = target 0.05
RESULTS_DIR = Path("results") / "always_accelerate"
HISTORY_FIELDS = ["epoch", "loss", "distance", "time", "steps", "accelerate_fraction",
                  "min_difference", "gradient_norm", "weight_change"]


@dataclass
class Episode:
    """One closed-loop sprint where the fly made its own decisions with fixed encoder weights."""
    observations: list[dict[str, float]]
    inputs: np.ndarray       # (T, 2) encoder features [distance/100, velocity/8] the fly saw
    actions: list[str]       # (T,) the fly's own decisions
    differences: np.ndarray  # (T,) mean(extensor) - mean(flexor) behind each decision; the loss uses this
    tanh: np.ndarray         # (T, SUBSTEPS, n_neurons) tanh(drive) per substep, kept for backprop
    distance: float
    time: float


def _network_step(network: RateNetwork, stimulus: np.ndarray, state: np.ndarray,
                  tanh_rows: np.ndarray) -> np.ndarray:
    """RateNetwork.step, operation for operation, but keeping tanh(drive) of every substep."""
    for k in range(len(tanh_rows)):
        tanh_rows[k] = np.tanh(network.gain * (network.circuit.weights @ state) + stimulus)
        state = state + network.leak * (tanh_rows[k] - state)
    return state


def rollout(network: RateNetwork, decoder: MotorDecoder, weights: np.ndarray) -> Episode:
    """The fly races with its own decisions; W, gain, leak and the decoder are only read."""
    encoder = LearnedSensoryEncoder(network.circuit, weights)
    event = Sprint()
    event.reset()
    state = np.zeros(network.circuit.size)
    observations, inputs, actions, differences, tanh = [], [], [], [], []
    while not event.is_finished():
        observation = dict(event.observe())
        features = encoder.features(observation)
        rows = np.empty((SUBSTEPS, network.circuit.size))
        state = _network_step(network, encoder.encode_features(features), state, rows)
        action, motor = decoder.decode(state)
        event.step(action)
        observations.append(observation)
        inputs.append(features)
        actions.append(action)
        differences.append(motor["difference"])
        tanh.append(rows)
    return Episode(observations, np.array(inputs), actions, np.array(differences), np.array(tanh),
                   event.position, round(event.time, 8))


def replay(network: RateNetwork, decoder: MotorDecoder, weights: np.ndarray,
           inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rerun the network over a fixed list of inputs (the path does not react to decisions).

    Used for the gradient check: unlike rollout, the result is smooth in the weights.
    """
    encoder = LearnedSensoryEncoder(network.circuit, weights)
    state = np.zeros(network.circuit.size)
    differences = np.empty(len(inputs))
    tanh = np.empty((len(inputs), SUBSTEPS, network.circuit.size))
    for t, features in enumerate(inputs):
        state = _network_step(network, encoder.encode_features(features), state, tanh[t])
        differences[t] = decoder.decode(state)[1]["difference"]
    return differences, tanh


def loss(differences: np.ndarray, threshold: float, margin: float = MARGIN) -> tuple[float, np.ndarray]:
    """Squared hinge: how far each step falls short of a confident ACCELERATE.

    The teacher says ACCELERATE at every state, so every step wants difference >= threshold + margin.
    Returns the mean loss and d(loss)/d(difference) per step, which backprop starts from.
    """
    shortfall = np.maximum(0.0, (threshold + margin) - differences)
    return float(np.mean(shortfall ** 2)), -2.0 * shortfall / len(differences)


def backward(network: RateNetwork, decoder: MotorDecoder, inputs: np.ndarray, tanh: np.ndarray,
             dloss: np.ndarray) -> np.ndarray:
    """Backprop through time: d(loss)/d(encoder weights), shape (n_sensory, 2).

    inputs and tanh are what the forward pass recorded (rollout or replay); dloss is
    d(loss)/d(difference) per step. The visited path is treated as fixed, so a change in
    weights changes the differences but not which observations the fly saw.
    """
    circuit = network.circuit
    sensory = circuit.indices("sensory")
    # difference = mean(extensor) - mean(flexor) = readout @ state
    readout = np.zeros(circuit.size)
    readout[decoder.extensor] = 1.0 / len(decoder.extensor)
    readout[decoder.flexor] = -1.0 / len(decoder.flexor)
    weights_T = circuit.weights.T.tocsr()
    grad = np.zeros((len(sensory), inputs.shape[1]))
    delta = np.zeros(circuit.size)  # d(loss)/d(state) after the current substep
    for t in range(len(inputs) - 1, -1, -1):
        delta = delta + dloss[t] * readout
        dstimulus = np.zeros(circuit.size)
        for k in range(tanh.shape[1] - 1, -1, -1):
            # state' = state + leak * (tanh(gain * W @ state + stimulus) - state)
            dz = network.leak * (1.0 - tanh[t, k] ** 2) * delta
            dstimulus += dz  # the same stimulus feeds every substep of this observation
            delta = (1.0 - network.leak) * delta + network.gain * (weights_T @ dz)
        # stimulus[sensory] = tonic + weights @ features, so each weight sees its feature
        grad += np.outer(dstimulus[sensory], inputs[t])
    return grad


def fingerprint(network: RateNetwork, decoder: MotorDecoder) -> str:
    """Hash of everything that must stay frozen: W, gain, leak, threshold and the motor readout."""
    weights = network.circuit.weights
    digest = hashlib.sha256()
    for array in (weights.data, weights.indices, weights.indptr, decoder.extensor, decoder.flexor,
                  np.array([network.gain, network.leak, decoder.threshold])):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def train(network: RateNetwork, decoder: MotorDecoder, initial: np.ndarray, epochs: int = 300,
          lr: float = 0.3, clip: float = 1.0, log=print) -> tuple[np.ndarray, list[dict]]:
    """On-policy training of the encoder weights with Adam; nothing else is touched.

    Every epoch the fly races with the current weights, then is judged against the teacher
    (always ACCELERATE) at every state it visited. Each history row describes the weights
    before that epoch's update, so the returned weights are the last ones evaluated.
    """
    frozen = fingerprint(network, decoder)
    weights = np.array(initial, dtype=float)
    first_moment, second_moment = np.zeros_like(weights), np.zeros_like(weights)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = []
    for epoch in range(1, epochs + 1):
        episode = rollout(network, decoder, weights)
        value, dloss = loss(episode.differences, decoder.threshold)
        grad = backward(network, decoder, episode.inputs, episode.tanh, dloss)
        norm = float(np.linalg.norm(grad))
        row = {
            "epoch": epoch,
            "loss": value,
            "distance": round(episode.distance, 4),
            "time": episode.time,
            "steps": len(episode.actions),
            "accelerate_fraction": episode.actions.count(ACCELERATE) / len(episode.actions),
            "min_difference": float(episode.differences.min()),
            "gradient_norm": norm,
            "weight_change": float(np.linalg.norm(weights - initial)),
        }
        history.append(row)
        log(f"{epoch:4d}  loss {value:.4e}  {row['distance']:6.2f} m  {row['time']:5.1f} s  "
            f"ACC {row['accelerate_fraction']:6.1%}  min diff {row['min_difference']:+.2e}  "
            f"|grad| {norm:.2e}  |E-E0| {row['weight_change']:.3f}")
        if value == 0.0 or epoch == epochs:
            break
        grad = grad * min(1.0, clip / (norm + 1e-12))
        first_moment = beta1 * first_moment + (1 - beta1) * grad
        second_moment = beta2 * second_moment + (1 - beta2) * grad ** 2
        step = (first_moment / (1 - beta1 ** epoch)) / (np.sqrt(second_moment / (1 - beta2 ** epoch)) + eps)
        weights = weights - lr * step
    if fingerprint(network, decoder) != frozen:
        raise RuntimeError("Frozen parameters changed during training.")
    return weights, history


def save_history(history: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    top.semilogy(epochs, [max(row["loss"], 1e-12) for row in history], color="tab:red")
    top.set(ylabel="loss", title="Always-Accelerate: encoder-only training")
    middle.plot(epochs, [row["distance"] for row in history], color="tab:green")
    middle.axhline(100, color="tab:orange", linestyle="--")
    middle.set(ylabel="distance (m)")
    bottom.plot(epochs, [row["accelerate_fraction"] for row in history], color="tab:blue")
    bottom.set(xlabel="epoch", ylabel="ACCELERATE fraction", ylim=(0, 1.05))
    fig.savefig(directory / "history.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=SPRINT_ENCODER_INIT)
    parser.add_argument("--output", type=Path, default=SPRINT_ENCODER)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.output.resolve() == SPRINT_ENCODER_INIT.resolve():
        parser.error(f"refusing to overwrite the frozen baseline {SPRINT_ENCODER_INIT}")
    controller = FlyBrainController()  # default 0.015 threshold, untouched
    initial = load_encoder_weights(args.init, controller.circuit)
    weights, history = train(controller.network, controller.decoder, initial, args.epochs, args.lr)
    save_encoder_weights(args.output, controller.circuit, weights)
    save_history(history, args.results_dir)
    final = history[-1]
    print(f"Trained in {final['epoch']} epochs: {final['distance']:.1f} m in {final['time']:.1f} s, "
          f"ACCELERATE {final['accelerate_fraction']:.1%}. Weights: {args.output}; "
          f"history: {args.results_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
