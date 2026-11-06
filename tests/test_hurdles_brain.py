import json

import numpy as np
import pytest

from brain.controller import HURDLES_ENCODER_INIT, HURDLES_GAIN1_POLICY, HURDLES_POLICY, FlyBrainController
from brain.hurdles_io import (HURDLES_INPUTS, HurdlesDecoder, HurdlesEncoder,
                              initial_hurdles_encoder_weights)
from brain.motor import MotorDecoder
from brain.sensory import load_encoder_weights, save_encoder_weights
from events.hurdles import ACCELERATE, COAST, JUMP
from experiments.run_hurdles import run, score


@pytest.fixture(scope="module")
def brain():
    return FlyBrainController.hurdles()


def test_initial_csv_matches_the_starting_weights(brain):
    weights = load_encoder_weights(HURDLES_ENCODER_INIT, brain.circuit, HURDLES_INPUTS)
    assert weights.shape == (24, 3)
    assert np.array_equal(weights, initial_hurdles_encoder_weights(brain.circuit))
    assert np.array_equal(brain.encoder.weights, weights)


def test_csv_round_trip_and_input_names(brain, tmp_path):
    weights = np.random.default_rng(0).normal(size=(24, 3))
    path = tmp_path / "encoder.csv"
    save_encoder_weights(path, brain.circuit, weights, HURDLES_INPUTS)
    assert np.array_equal(load_encoder_weights(path, brain.circuit, HURDLES_INPUTS), weights)
    # Sprint's default input names do not match a Hurdles file.
    with pytest.raises(ValueError):
        load_encoder_weights(path, brain.circuit)
    with pytest.raises(ValueError):
        HurdlesEncoder(brain.circuit, np.zeros((24, 2)))


def test_encoder_formula_and_clipping(brain):
    weights = np.random.default_rng(1).normal(size=(24, 3))
    encoder = HurdlesEncoder(brain.circuit, weights)
    observation = {"distance_to_next_hurdle": 10.0, "distance_to_second_hurdle": 40.0, "velocity": 12.0}
    features = encoder.features(observation)
    assert features.tolist() == [0.25, 1.0, 1.0]  # velocity 12 clips to 1
    stimulus = encoder.encode(observation)
    assert np.allclose(stimulus[encoder.indices], 0.45 + weights @ features)
    others = np.setdiff1d(np.arange(brain.circuit.size), encoder.indices)
    assert not stimulus[others].any()


def test_decoder_reads_the_sprint_comparison_as_jump(brain):
    hurdles, sprint = HurdlesDecoder(brain.circuit), MotorDecoder(brain.circuit)
    assert (hurdles.coast_threshold, hurdles.jump_threshold) == (0.015, 0.30)
    for seed in range(20):
        activity = np.random.default_rng(seed).uniform(-1, 1, brain.circuit.size)
        action, motor = hurdles.decode(activity)
        sprint_action, sprint_motor = sprint.decode(activity)
        assert motor == sprint_motor
        # Below the jump band, the decoder makes Sprint's push / no-push choice.
        if motor["difference"] <= hurdles.jump_threshold:
            assert action == {"ACCELERATE": ACCELERATE, "COAST": COAST}[sprint_action]


def motor_activity(circuit, difference):
    """Activity whose extensor - flexor means differ by exactly `difference`."""
    activity = np.zeros(circuit.size)
    activity[circuit.indices("motor_extensor")] = 0.5 + difference / 2
    activity[circuit.indices("motor_flexor")] = 0.5 - difference / 2
    return activity


@pytest.mark.parametrize("difference, expected", [
    (-0.2, COAST), (0.0, COAST), (0.01, COAST),
    (0.02, ACCELERATE), (0.2, ACCELERATE), (0.29, ACCELERATE),
    (0.31, JUMP), (0.9, JUMP),
])
def test_decoder_bands(brain, difference, expected):
    action, motor = HurdlesDecoder(brain.circuit).decode(motor_activity(brain.circuit, difference))
    assert motor["difference"] == pytest.approx(difference)
    assert action == expected


def test_untrained_fly_stalls_like_the_untrained_sprint_fly():
    for layout in ("benchmark", "seeded"):
        records = run("fly", seed=2, layout=layout)
        assert records == run("fly", seed=2, layout=layout)
        assert all(row["action"] in {ACCELERATE, COAST, JUMP} for row in records)
        assert "motor_difference" in records[-1]
        # The network saturates within a few steps; the difference settles in the COAST band.
        assert all(row["action"] == COAST for row in records[10:])
        assert records[-1]["simulation_time"] == pytest.approx(90.0) and records[-1]["position"] < 15.0


def test_controller_arguments(brain):
    with pytest.raises(ValueError):
        FlyBrainController(encoder_weights=np.zeros((24, 2)), encoder=brain.encoder)
    with pytest.raises(ValueError):
        FlyBrainController.hurdles(jump_threshold=2.0)
    with pytest.raises(ValueError):
        FlyBrainController.hurdles(coast_threshold=0.5, jump_threshold=0.3)
    custom = FlyBrainController.hurdles(coast_threshold=0.0, jump_threshold=0.5)
    assert (custom.decoder.coast_threshold, custom.decoder.jump_threshold) == (0.0, 0.5)


def test_hurdles_es_reproduces_benchmark_score():
    assert HURDLES_POLICY.exists(), f"Missing policy file: {HURDLES_POLICY}"
    policy = json.loads(HURDLES_POLICY.read_text())
    assert policy.get("version") == 1
    assert policy.get("event") == "hurdles"
    controller = FlyBrainController.from_policy(HURDLES_POLICY)
    records = run(controller=controller, layout="benchmark")
    benchmark_score = score(records)
    assert benchmark_score == pytest.approx(policy["trained"]["benchmark_score"])
    trained_records = run("trained", layout="benchmark")
    assert score(trained_records) == pytest.approx(benchmark_score)


def test_hurdles_es_gain1_reproduces_benchmark_score():
    assert HURDLES_GAIN1_POLICY.exists(), f"Missing policy file: {HURDLES_GAIN1_POLICY}"
    policy = json.loads(HURDLES_GAIN1_POLICY.read_text())
    assert policy.get("version") == 1
    assert policy.get("event") == "hurdles"
    assert policy.get("gain") == 1.0
    controller = FlyBrainController.from_policy(HURDLES_GAIN1_POLICY)
    assert controller.network.gain == 1.0
    records = run(controller=controller, layout="benchmark")
    benchmark_score = score(records)
    assert benchmark_score == pytest.approx(policy["trained"]["benchmark_score"])
