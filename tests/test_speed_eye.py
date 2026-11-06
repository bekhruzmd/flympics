import json

import numpy as np
import pytest

from brain.controller import FlyBrainController, SPRINT_EYE_POLICY
from brain.sensory import initial_encoder_weights
from brain.speed_eye import FlyVisSpeedEye, SprintEyeEncoder, blindfolded_eye
from experiments.train_sprint_eye import compare


def test_speed_calibration_trends_with_speed():
    eye = FlyVisSpeedEye()
    assert len(eye.speeds) == 21
    assert (eye.speeds[0], eye.speeds[-1]) == (0.0, 8.0)
    assert np.all(np.diff(eye.signal) > 0)
    assert eye(0.0) == 0.0
    assert eye(9.0) == eye(8.0)
    assert eye(0.2) == pytest.approx(eye(0.4) / 2)


def test_blindfold_zeroes_only_the_eye_feature():
    circuit = FlyBrainController().circuit
    weights = initial_encoder_weights(circuit)
    sighted = SprintEyeEncoder(circuit, weights)
    blindfolded = SprintEyeEncoder(circuit, weights, eye=blindfolded_eye)
    for speed in (0.0, 2.0, 6.0, 8.0):
        observation = {"distance_to_finish": 40.0, "velocity": speed}
        assert sighted.features(observation) == pytest.approx([0.4, sighted.eye(speed)])
        assert blindfolded.features(observation) == pytest.approx([0.4, 0.0])
    assert sighted.features(observation)[1] != 0.0
    assert not np.array_equal(sighted.encode(observation), blindfolded.encode(observation))


def test_saved_policy_reproduces_the_sprint_times():
    policy = json.loads(SPRINT_EYE_POLICY.read_text())
    results = compare()
    assert results["eye"] == pytest.approx(policy["trained"])
    assert results["eye"]["finished"]
    for name, expected in policy["comparison"].items():
        assert results[name] == pytest.approx(expected)
