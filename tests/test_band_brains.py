import json

import numpy as np
import pytest

from brain.archery_io import (ARCHERY_ACTIONS, ARCHERY_INPUTS, ArcheryDecoder, ArcheryEncoder,
                              initial_archery_encoder_weights)
from brain.controller import ARCHERY_ENCODER_INIT, STEEPLECHASE_ENCODER_INIT, FlyBrainController
from brain.event_io import BandDecoder
from brain.evolution import ParameterLayout, batched_rollout, centred_ranks, evolve
from brain.sensory import load_encoder_weights, save_encoder_weights
from brain.steeplechase_io import (STEEPLECHASE_ACTIONS, STEEPLECHASE_INPUTS, SteeplechaseEncoder,
                                   initial_steeplechase_encoder_weights)
from events.archery import Archery
from events.steeplechase import Steeplechase
from experiments import run_archery, run_steeplechase


@pytest.fixture(scope="module")
def circuit():
    return FlyBrainController.archery().circuit


def test_initial_csvs_match_the_starting_weights(circuit):
    assert np.array_equal(load_encoder_weights(ARCHERY_ENCODER_INIT, circuit, ARCHERY_INPUTS),
                          initial_archery_encoder_weights(circuit))
    assert np.array_equal(load_encoder_weights(STEEPLECHASE_ENCODER_INIT, circuit, STEEPLECHASE_INPUTS),
                          initial_steeplechase_encoder_weights(circuit))


def test_encoder_features(circuit):
    archery = ArcheryEncoder(circuit, np.zeros((24, 3)))
    assert archery.features({"aim_offset": -1.0, "wind": 2.0, "draw": 0.5}).tolist() == [-1.0, 0.5, 0.5]
    weights = np.random.default_rng(0).normal(size=(24, 5))
    steeple = SteeplechaseEncoder(circuit, weights)
    observation = {"distance_to_barrier": 20.0, "water_next": 1.0, "velocity": 16.0,
                   "stamina": 0.25, "distance_to_finish": 300.0}
    features = steeple.features(observation)
    assert features.tolist() == [0.5, 1.0, 1.0, 0.25, 0.5]
    assert np.allclose(steeple.encode(observation)[steeple.indices], 0.45 + weights @ features)
    with pytest.raises(ValueError):
        SteeplechaseEncoder(circuit, np.zeros((24, 3)))


def test_band_decoder(circuit):
    decoder = BandDecoder(circuit, ("A", "B", "C"), (-0.1, 0.2))
    assert [decoder.band(d) for d in (-0.5, -0.1, 0.0, 0.2, 0.21)] == [0, 0, 1, 1, 2]
    for bad in ((0.2, -0.1), (0.1, 0.1), (-2.0, 0.0), (0.0,)):
        with pytest.raises(ValueError):
            BandDecoder(circuit, ("A", "B", "C"), bad)
    archery = ArcheryDecoder(circuit, (-0.01, 0.01))
    activity = np.zeros(circuit.size)
    activity[archery.flexor] = 0.5
    assert archery.decode(activity)[0] == "LEFT"
    activity[archery.extensor] = 1.0
    assert archery.decode(activity)[0] == "RIGHT"


def test_untrained_flies_run_their_events():
    archery = run_archery.run("fly", seed=3)
    assert all(r["action"] in ARCHERY_ACTIONS for r in archery)
    assert sum(r["shot"] for r in archery) == 6 and "motor_difference" in archery[-1]
    steeple = run_steeplechase.run("fly")
    assert all(r["action"] in STEEPLECHASE_ACTIONS for r in steeple)


def test_batched_rollout_matches_the_controller(circuit):
    rng = np.random.default_rng(3)
    weights = rng.normal(0, 1.5, (2, 24, 3))
    thresholds = np.sort(rng.normal(0, 0.05, (2, 2)), axis=1)
    events = batched_rollout(circuit, Archery, [5, 6], ArcheryEncoder(circuit, weights[0]),
                             ARCHERY_ACTIONS, weights, thresholds)
    for member in range(2):
        for j, seed in enumerate((5, 6)):
            fly = FlyBrainController.band_event("archery", weights[member], thresholds[member], circuit)
            records = run_archery.run(seed=seed, controller=fly)
            assert events[member][j].points == [r["points"] for r in records if r["shot"]]
    weights = rng.normal(0, 1.5, (2, 24, 5))
    thresholds = np.sort(rng.normal(0, 0.1, (2, 2)), axis=1)
    events = batched_rollout(circuit, Steeplechase, [0], SteeplechaseEncoder(circuit, weights[0]),
                             STEEPLECHASE_ACTIONS, weights, thresholds)
    for member in range(2):
        fly = FlyBrainController.band_event("steeplechase", weights[member], thresholds[member], circuit)
        assert events[member][0].score() == pytest.approx(run_steeplechase.score(run_steeplechase.run(controller=fly)))


def test_parameter_layout_round_trip():
    layout = ParameterLayout(24, 3, 2)
    weights = np.arange(72.0).reshape(24, 3)
    theta = layout.pack(weights, (0.3, -0.2))
    unpacked, thresholds = layout.unpack(theta)
    assert np.array_equal(unpacked, weights) and thresholds.tolist() == [-0.2, 0.3]
    assert layout.decoder_thresholds(layout.pack(weights, (0.1, 0.1)))[1] > 0.1


def test_centred_ranks_and_evolve_on_a_toy_problem():
    assert centred_ranks(np.array([3.0, 1.0, 2.0])).tolist() == [0.5, -0.5, 0.0]
    target = np.array([1.0, -2.0, 0.5])
    result = evolve(np.zeros(3), lambda xs: -((xs - target) ** 2).sum(axis=1), generations=150,
                    pairs=8, sigma=0.1, lr=0.05, seed=0, log=lambda _: None)
    assert np.abs(result.best_theta - target).max() < 0.15
    assert result.history[-1]["best_so_far"] == result.best_fitness


def test_from_policy(tmp_path, circuit):
    weights = np.random.default_rng(1).normal(size=(24, 5))
    save_encoder_weights(tmp_path / "steeple_encoder.csv", circuit, weights, STEEPLECHASE_INPUTS)
    policy = {"version": 1, "event": "steeplechase", "encoder": "steeple_encoder.csv", "thresholds": [-0.1, 0.2]}
    (tmp_path / "steeple.json").write_text(json.dumps(policy))
    fly = FlyBrainController.from_policy(tmp_path / "steeple.json", circuit)
    assert np.array_equal(fly.encoder.weights, weights) and fly.decoder.thresholds == (-0.1, 0.2)
    (tmp_path / "bad.json").write_text(json.dumps({**policy, "event": "sprint"}))
    with pytest.raises(ValueError):
        FlyBrainController.from_policy(tmp_path / "bad.json", circuit)
    with pytest.raises(FileNotFoundError):
        FlyBrainController.from_policy(tmp_path / "missing.json", circuit)


def test_teacher_forced_replay_matches_the_controller(circuit):
    from brain.evolution import teacher_forced_differences
    weights = np.random.default_rng(4).normal(0, 1.0, (2, 24, 3))
    encoder = ArcheryEncoder(circuit, weights[0])
    observations = [{"aim_offset": 0.05 * k - 0.3, "wind": 3.0 - 0.4 * k, "draw": min(1.0, k / 10)}
                    for k in range(15)]
    features = np.array([[encoder.features(o) for o in observations]])
    replay = teacher_forced_differences(circuit, features, encoder.tonic, weights)
    for member in range(2):
        fly = FlyBrainController.band_event("archery", weights[member], (-0.1, 0.1), circuit)
        fly.reset()
        for k, observation in enumerate(observations):
            fly.act(observation)
            assert replay[member, 0, k] == pytest.approx(fly.last_summary["motor_difference"], abs=1e-12)


def test_fit_band_thresholds_separates_ordered_classes():
    from experiments.es_training import fit_band_thresholds
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 300)
    difference = labels * 0.1 + rng.uniform(-0.03, 0.03, 300)
    score, (low, high) = fit_band_thresholds(difference, labels, np.full(300, 1 / 300))
    assert score > 0.99  # cuts are tried at 200 quantiles, so the exact gap can be missed
    assert 0.02 <= low <= 0.08 and 0.12 <= high <= 0.18
    decoder = BandDecoder(FlyBrainController.archery().circuit, ("A", "B", "C"), (low, high))
    assert np.mean([decoder.band(d) == label for d, label in zip(difference, labels)]) > 0.99
