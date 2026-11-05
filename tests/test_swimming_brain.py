import numpy as np
import pytest

from brain.controller import SWIMMING_ENCODER_INIT, FlyBrainController
from brain.motor import MotorDecoder
from brain.sensory import load_encoder_weights, save_encoder_weights
from brain.swimming_io import (SWIMMING_INPUTS, SwimmingDecoder, SwimmingEncoder,
                               initial_swimming_encoder_weights)
from events.swimming import PULL, RECOVER
from experiments.run_swimming import run, score


@pytest.fixture(scope="module")
def brain():
    return FlyBrainController.swimming()


def test_initial_csv_matches_the_starting_weights(brain):
    weights = load_encoder_weights(SWIMMING_ENCODER_INIT, brain.circuit, SWIMMING_INPUTS)
    assert weights.shape == (24, 3)
    assert np.array_equal(weights, initial_swimming_encoder_weights(brain.circuit))
    assert np.array_equal(brain.encoder.weights, weights)


def test_csv_round_trip_and_input_names(brain, tmp_path):
    weights = np.random.default_rng(0).normal(size=(24, 3))
    path = tmp_path / "encoder.csv"
    save_encoder_weights(path, brain.circuit, weights, SWIMMING_INPUTS)
    assert np.array_equal(load_encoder_weights(path, brain.circuit, SWIMMING_INPUTS), weights)
    with pytest.raises(ValueError):
        load_encoder_weights(path, brain.circuit)  # Sprint's input names
    with pytest.raises(ValueError):
        SwimmingEncoder(brain.circuit, np.zeros((24, 2)))


def test_encoder_formula_and_clipping(brain):
    weights = np.random.default_rng(1).normal(size=(24, 3))
    encoder = SwimmingEncoder(brain.circuit, weights)
    observation = {"distance_to_wall": 25.0, "velocity": 6.0, "stroke_phase": 0.5}
    features = encoder.features(observation)
    assert features.tolist() == [0.5, 1.0, 0.5]  # velocity 6 clips to 1
    stimulus = encoder.encode(observation)
    assert np.allclose(stimulus[encoder.indices], 0.45 + weights @ features)
    others = np.setdiff1d(np.arange(brain.circuit.size), encoder.indices)
    assert not stimulus[others].any()


def test_decoder_reads_the_sprint_comparison_as_pull(brain):
    swimming, sprint = SwimmingDecoder(brain.circuit), MotorDecoder(brain.circuit)
    for seed in range(20):
        activity = np.random.default_rng(seed).uniform(-1, 1, brain.circuit.size)
        action, motor = swimming.decode(activity)
        _, sprint_motor = sprint.decode(activity)
        assert motor == sprint_motor
        assert action == (PULL if motor["difference"] > swimming.threshold else RECOVER)


def test_untrained_fly_swims_a_full_race():
    records = run("fly", seed=2)
    assert records == run("fly", seed=5)  # the race and the brain ignore the seed
    assert all(row["action"] in {PULL, RECOVER} for row in records)
    assert records[-1]["position"] == 100.0 or records[-1]["simulation_time"] == pytest.approx(120.0)
    assert "motor_difference" in records[-1]
    assert score(records) > 0


def test_controller_arguments():
    with pytest.raises(ValueError):
        FlyBrainController.swimming(motor_threshold=2.0)
