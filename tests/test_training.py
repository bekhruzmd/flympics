import json

import numpy as np
import pytest

from brain.controller import FlyBrainController, SPRINT_POLICY
from events.sprint import ACCELERATE, COAST, Sprint
from experiments.run_sprint import run
from experiments.train_sprint import train


def test_training_reproduces_saved_policy_and_improves_finish():
    policy = train(seed=7)
    assert policy == json.loads(SPRINT_POLICY.read_text())
    assert policy["best"]["finished"]
    assert not policy["untrained"]["finished"]
    assert policy["best"]["score"] < policy["untrained"]["score"]


def test_trained_policy_matches_acceleration_reference():
    reference = Sprint()
    reference.reset()
    while not reference.is_finished():
        reference.step(ACCELERATE)
    records = run("trained")
    assert records[-1]["position"] == 100.0
    assert records[-1]["simulation_time"] == pytest.approx(reference.score())
    assert records == run("trained", seed=123)
    assert all("motor_difference" in row for row in records)


def test_trained_decoder_still_requires_neural_activity():
    controller = FlyBrainController.trained()
    action, _ = controller.decoder.decode(np.zeros(controller.circuit.size))
    assert action == COAST
    controller.reset()
    first = controller.act({"distance_to_finish": 100.0, "velocity": 0.0})
    summary = controller.last_summary.copy()
    controller.reset()
    assert controller.act({"distance_to_finish": 100.0, "velocity": 0.0}) == first
    assert controller.last_summary == summary


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        FlyBrainController(threshold)


def test_policy_errors_are_explicit(tmp_path):
    path = tmp_path / "policy.json"
    with pytest.raises(FileNotFoundError, match="train_sprint"):
        FlyBrainController.trained(path)
    path.write_text(json.dumps({"version": 2, "event": "sprint"}))
    with pytest.raises(ValueError, match="version 1"):
        FlyBrainController.trained(path)
