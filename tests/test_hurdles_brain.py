import numpy as np
import pytest

from brain.controller import HURDLES_ENCODER_INIT, FlyBrainController
from brain.hurdles_io import (HURDLES_INPUTS, HurdlesDecoder, HurdlesEncoder,
                              initial_hurdles_encoder_weights)
from brain.motor import MotorDecoder
from brain.sensory import load_encoder_weights, save_encoder_weights
from events.hurdles import JUMP, RUN
from experiments.run_hurdles import run


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
    for seed in range(20):
        activity = np.random.default_rng(seed).uniform(-1, 1, brain.circuit.size)
        action, motor = hurdles.decode(activity)
        _, sprint_motor = sprint.decode(activity)
        assert motor == sprint_motor
        assert action == (JUMP if motor["difference"] > hurdles.threshold else RUN)


def test_untrained_fly_runs_full_races():
    for layout in ("benchmark", "seeded"):
        records = run("fly", seed=2, layout=layout)
        assert records == run("fly", seed=2, layout=layout)
        assert all(row["action"] in {RUN, JUMP} for row in records)
        assert records[-1]["position"] == 200.0 or records[-1]["simulation_time"] == pytest.approx(90.0)
        assert "motor_difference" in records[-1]


def test_controller_arguments(brain):
    with pytest.raises(ValueError):
        FlyBrainController(encoder_weights=np.zeros((24, 2)), encoder=brain.encoder)
    with pytest.raises(ValueError):
        FlyBrainController.hurdles(motor_threshold=2.0)
