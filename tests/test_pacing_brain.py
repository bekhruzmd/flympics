import numpy as np
import pytest

from brain.controller import PACING_ENCODER_INIT, PACING_POLICY, FlyBrainController
from brain.pacing_io import (PACING_ACTIONS, PACING_INPUTS, PacingDecoder, PacingEncoder,
                            initial_pacing_encoder_weights)
from brain.sensory import load_encoder_weights, save_encoder_weights
from events.pacing import ACCELERATE, COAST
from experiments.run_pacing import run, score


@pytest.fixture(scope="module")
def brain():
    return FlyBrainController.pacing()


def test_initial_csv_matches_starting_weights(brain):
    weights = load_encoder_weights(PACING_ENCODER_INIT, brain.circuit, PACING_INPUTS)
    assert weights.shape == (24, 3)
    assert np.array_equal(weights, initial_pacing_encoder_weights(brain.circuit))
    assert np.array_equal(brain.encoder.weights, weights)


def test_csv_round_trip_and_input_names(brain, tmp_path):
    weights = np.random.default_rng(0).normal(size=(24, 3))
    path = tmp_path / "pacing_encoder.csv"
    save_encoder_weights(path, brain.circuit, weights, PACING_INPUTS)
    assert np.array_equal(load_encoder_weights(path, brain.circuit, PACING_INPUTS), weights)
    with pytest.raises(ValueError, match="unknown input"):
        load_encoder_weights(path, brain.circuit)  # Sprint expects ("distance", "velocity")
    with pytest.raises(ValueError, match="Expected encoder shape"):
        PacingEncoder(brain.circuit, np.zeros((24, 2)))


def test_encoder_formula_and_clipping(brain):
    weights = np.random.default_rng(42).normal(size=(24, 3))
    encoder = PacingEncoder(brain.circuit, weights, distance=400.0, max_velocity=8.0, energy_budget=100.0)
    observation = {"distance_to_finish": 200.0, "velocity": 10.0, "energy": 50.0}
    features = encoder.features(observation)
    assert features.tolist() == [0.5, 1.0, 0.5]  # velocity 10.0 clips to 1.0
    stimulus = encoder.encode(observation)
    assert np.allclose(stimulus[encoder.indices], 0.45 + weights @ features)
    others = np.setdiff1d(np.arange(brain.circuit.size), encoder.indices)
    assert not stimulus[others].any()


def test_decoder_reads_extensor_minus_flexor(brain):
    pacing_dec = PacingDecoder(brain.circuit, thresholds=(0.015,))
    for seed in range(20):
        activity = np.random.default_rng(seed).uniform(-1, 1, brain.circuit.size)
        act_pacing, motor_pacing = pacing_dec.decode(activity)
        expected_act = ACCELERATE if motor_pacing["difference"] > 0.015 else COAST
        assert act_pacing == expected_act


def test_untrained_fly_runs_honestly():
    records = run("fly", seed=7)
    assert all(row["action"] in {ACCELERATE, COAST} for row in records)
    assert "motor_difference" in records[-1]
    final = records[-1]
    assert final["position"] < 10.0
    assert final["simulation_time"] == pytest.approx(240.0)
    assert score(records) > 600.0


def test_trained_fly_completes_race():
    assert PACING_POLICY.exists()
    records = run("trained", seed=7)
    final = records[-1]
    assert final["position"] == pytest.approx(400.0)
    assert final["simulation_time"] == pytest.approx(198.1)
    assert score(records) == pytest.approx(198.1)
