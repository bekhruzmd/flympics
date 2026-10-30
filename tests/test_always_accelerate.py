import sys

import numpy as np
import pytest

from brain.controller import SPRINT_ENCODER, SPRINT_ENCODER_INIT, FlyBrainController
from brain.sensory import (LearnedSensoryEncoder, SensoryEncoder, initial_encoder_weights,
                           load_encoder_weights, save_encoder_weights)
from events.sprint import ACCELERATE, Sprint
from experiments import train_always_accelerate as experiment
from experiments.run_sprint import run

OBSERVATIONS = [{"distance_to_finish": 100.0, "velocity": 0.0},
                {"distance_to_finish": 42.5, "velocity": 3.3},
                {"distance_to_finish": 0.0, "velocity": 9.0}]


@pytest.fixture(scope="module")
def brain():
    return FlyBrainController()


@pytest.fixture(scope="module")
def short_training(brain):
    initial = load_encoder_weights(SPRINT_ENCODER_INIT, brain.circuit)
    return experiment.train(brain.network, brain.decoder, initial, epochs=15, log=lambda line: None)


def test_init_csv_holds_the_untrained_coefficients(brain):
    weights = load_encoder_weights(SPRINT_ENCODER_INIT, brain.circuit)
    assert weights.shape == (len(brain.circuit.indices("sensory")), 2)
    assert np.array_equal(weights, initial_encoder_weights(brain.circuit))


def test_csv_round_trip_is_exact(brain, tmp_path):
    weights = np.random.default_rng(0).normal(size=(len(brain.circuit.indices("sensory")), 2))
    path = tmp_path / "encoder.csv"
    save_encoder_weights(path, brain.circuit, weights)
    assert np.array_equal(load_encoder_weights(path, brain.circuit), weights)
    rows = path.read_text().splitlines()
    assert rows[0] == "pre,post,weight" and len(rows) == 1 + weights.size


@pytest.mark.parametrize("edit, message", [
    (lambda rows: ["a,b,c"] + rows[1:], "expected columns"),
    (lambda rows: rows[:1] + ["speed" + rows[1][rows[1].index(","):]] + rows[2:], "unknown input"),
    (lambda rows: rows[:1] + ["distance,1,0.5"] + rows[2:], "not a sensory neuron"),
    (lambda rows: rows[:1] + [rows[1].rsplit(",", 1)[0] + ",nan"] + rows[2:], "finite"),
    (lambda rows: rows + [rows[1]], "duplicate"),
    (lambda rows: rows[:-1], "missing 1 edge"),
])
def test_bad_encoder_csv_is_rejected(brain, tmp_path, edit, message):
    rows = SPRINT_ENCODER_INIT.read_text().splitlines()
    path = tmp_path / "encoder.csv"
    path.write_text("\n".join(edit(rows)) + "\n")
    with pytest.raises(ValueError, match=message):
        load_encoder_weights(path, brain.circuit)


def test_wrong_encoder_shape_is_rejected(brain, tmp_path):
    with pytest.raises(ValueError, match="shape"):
        LearnedSensoryEncoder(brain.circuit, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        save_encoder_weights(tmp_path / "encoder.csv", brain.circuit, np.zeros((3, 2)))


def test_init_weights_reproduce_the_untrained_fly(brain):
    learned = LearnedSensoryEncoder(brain.circuit, initial_encoder_weights(brain.circuit))
    original = SensoryEncoder(brain.circuit)
    for observation in OBSERVATIONS:
        assert np.array_equal(learned.encode(observation), original.encode(observation))
    assert run(controller=FlyBrainController(encoder_weights=SPRINT_ENCODER_INIT)) == run("fly")


def test_stimulus_only_reaches_sensory_neurons(brain):
    weights = np.random.default_rng(1).normal(size=(len(brain.circuit.indices("sensory")), 2))
    encoder = LearnedSensoryEncoder(brain.circuit, weights)
    others = np.setdiff1d(np.arange(brain.circuit.size), brain.circuit.indices("sensory"))
    for observation in OBSERVATIONS:
        assert not encoder.encode(observation)[others].any()


def test_rollout_and_replay_match_the_real_controller(brain):
    initial = initial_encoder_weights(brain.circuit)
    episode = experiment.rollout(brain.network, brain.decoder, initial)
    records = run("fly")
    assert episode.actions == [row["action"] for row in records]
    assert episode.differences.tolist() == [row["motor_difference"] for row in records]
    differences, tanh = experiment.replay(brain.network, brain.decoder, initial, episode.inputs)
    assert np.array_equal(differences, episode.differences) and np.array_equal(tanh, episode.tanh)


def test_loss_only_rewards_confident_acceleration():
    value, dloss = experiment.loss(np.array([0.0, 0.015, 0.03, 0.05, 0.9]), threshold=0.015)
    assert value == pytest.approx((0.05 ** 2 + 0.035 ** 2 + 0.02 ** 2) / 5)
    assert dloss == pytest.approx([-0.02, -0.014, -0.008, 0.0, 0.0])


def test_backward_matches_finite_differences(brain):
    sensory = brain.circuit.indices("sensory")
    weights = np.random.default_rng(0).normal(0, 0.3, (len(sensory), 2))
    inputs = experiment.rollout(brain.network, brain.decoder, initial_encoder_weights(brain.circuit)).inputs[:8]

    def objective(candidate):
        return experiment.loss(experiment.replay(brain.network, brain.decoder, candidate, inputs)[0],
                               brain.decoder.threshold)

    differences, tanh = experiment.replay(brain.network, brain.decoder, weights, inputs)
    grad = experiment.backward(brain.network, brain.decoder, inputs, tanh,
                               experiment.loss(differences, brain.decoder.threshold)[1])
    numeric = np.zeros_like(grad)
    for index in np.ndindex(*grad.shape):
        up, down = weights.copy(), weights.copy()
        up[index] += 1e-6
        down[index] -= 1e-6
        numeric[index] = (objective(up)[0] - objective(down)[0]) / 2e-6
    assert np.max(np.abs(grad - numeric)) < 1e-6 * np.max(np.abs(numeric))
    # Sensory neurons with no outgoing synapse cannot influence the motor readout.
    isolated = [row for row, neuron in enumerate(sensory) if brain.circuit.weights[:, neuron].nnz == 0]
    assert len(isolated) == 3 and not grad[isolated].any()


def test_training_only_moves_the_encoder(brain, short_training):
    before = experiment.fingerprint(brain.network, brain.decoder)
    weights, history = short_training
    assert experiment.fingerprint(brain.network, brain.decoder) == before
    assert weights.shape == initial_encoder_weights(brain.circuit).shape
    assert history[0]["weight_change"] == 0.0 and history[-1]["weight_change"] > 0


def test_training_learns_to_accelerate(short_training):
    _, history = short_training
    first, last = history[0], history[-1]
    assert first["distance"] < 10 and first["accelerate_fraction"] < 0.05
    assert last["distance"] == 100.0 and last["accelerate_fraction"] == 1.0
    assert last["loss"] < first["loss"] / 100


def test_training_is_reproducible(brain, short_training):
    initial = load_encoder_weights(SPRINT_ENCODER_INIT, brain.circuit)
    weights, history = experiment.train(brain.network, brain.decoder, initial, epochs=15, log=lambda line: None)
    assert np.array_equal(weights, short_training[0]) and history == short_training[1]


def test_history_files_are_written(short_training, tmp_path):
    experiment.save_history(short_training[1], tmp_path)
    rows = (tmp_path / "history.csv").read_text().splitlines()
    assert rows[0] == ",".join(experiment.HISTORY_FIELDS) and len(rows) == 1 + len(short_training[1])
    assert (tmp_path / "history.png").stat().st_size > 0


def test_training_refuses_to_overwrite_the_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--output", str(SPRINT_ENCODER_INIT), "--epochs", "1"])
    with pytest.raises(SystemExit):
        experiment.main()


def test_saved_encoder_matches_acceleration_reference():
    reference = Sprint()
    reference.reset()
    while not reference.is_finished():
        reference.step(ACCELERATE)
    records = run("accelerate")
    assert records[-1]["position"] == 100.0
    assert records[-1]["simulation_time"] == pytest.approx(reference.score())
    assert all(row["action"] == ACCELERATE for row in records)
    assert records == run("accelerate", seed=123)
    # The decoder threshold stays at the untrained default; only the encoder learned.
    assert FlyBrainController.always_accelerate().decoder.threshold == 0.015


def test_missing_encoder_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_always_accelerate"):
        FlyBrainController.always_accelerate(tmp_path / "missing.csv")
    assert SPRINT_ENCODER.exists()
