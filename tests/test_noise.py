"""Optional Gaussian noise in the rate network: off is exactly the old model, on is seeded."""
import json

import numpy as np
import pytest

from brain.connectome import load_manc_front_leg_circuit
from brain.controller import ARCHERY_POLICY, STEEPLECHASE_POLICY, FlyBrainController
from brain.dynamics import RateNetwork
from brain.evolution import BatchedRateNetwork
from experiments import run_archery, run_steeplechase
from experiments.train_archery import evaluate as evaluate_archery
from experiments.train_steeplechase import evaluate as evaluate_steeplechase

CIRCUIT = load_manc_front_leg_circuit()


def old_step(circuit, state, stimulus, leak=0.35, gain=3.0, substeps=2):
    """The update RateNetwork.step used before noise existed, copied verbatim."""
    state = state.copy()
    silenced = np.asarray(circuit.ablated, dtype=int)
    for _ in range(substeps):
        target = np.tanh(gain * (circuit.weights @ state) + stimulus)
        state += leak * (target - state)
        state[silenced] = 0.0
    return state


def stimuli(steps=20, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 1.0, (steps, CIRCUIT.size))


def test_noise_off_is_bit_for_bit_the_old_update():
    network = RateNetwork(CIRCUIT, noise=0.0, seed=123)
    state = np.zeros(CIRCUIT.size)
    for stimulus in stimuli():
        state = old_step(CIRCUIT, state, stimulus)
        np.testing.assert_array_equal(network.step(stimulus), state)


def test_noise_off_matches_saved_trained_results():
    archery = FlyBrainController.from_policy(ARCHERY_POLICY, CIRCUIT, noise=0.0)
    steeple = FlyBrainController.from_policy(STEEPLECHASE_POLICY, CIRCUIT, noise=0.0)
    assert evaluate_archery(archery) == json.loads(ARCHERY_POLICY.read_text())["trained"]
    assert evaluate_steeplechase(steeple) == json.loads(STEEPLECHASE_POLICY.read_text())["trained"]


def test_noise_on_is_reproducible_with_a_seed_and_changes_with_it():
    def trace(seed):
        network = RateNetwork(CIRCUIT, noise=1e-3)
        network.reset(seed)
        return np.array([network.step(s) for s in stimuli()])

    np.testing.assert_array_equal(trace(5), trace(5))
    assert not np.array_equal(trace(5), trace(6))
    # And the noise really is there: it moves the state away from the noiseless one.
    clean = RateNetwork(CIRCUIT)
    assert not np.array_equal(trace(5), np.array([clean.step(s) for s in stimuli()]))


def test_seeded_noisy_controller_runs_are_reproducible():
    def archery(seed):
        controller = FlyBrainController.from_policy(ARCHERY_POLICY, CIRCUIT, noise=1e-3)
        return [r["action"] for r in run_archery.run(seed=seed, controller=controller)]

    def steeple(seed):
        controller = FlyBrainController.from_policy(STEEPLECHASE_POLICY, CIRCUIT, noise=1e-3)
        return run_steeplechase.score(run_steeplechase.run(seed=seed, controller=controller))

    assert archery(1000) == archery(1000)
    assert steeple(1000) == steeple(1000)


def test_batched_network_noise_off_matches_and_noise_on_is_seeded():
    batch = stimuli(seed=1)[:3]
    single = [RateNetwork(CIRCUIT) for _ in batch]
    batched = BatchedRateNetwork(CIRCUIT, len(batch))
    out = batched.step(batch)
    np.testing.assert_allclose(out, [n.step(s) for n, s in zip(single, batch)], rtol=0, atol=1e-12)

    def noisy(seed):
        return BatchedRateNetwork(CIRCUIT, len(batch), noise=1e-3, seed=seed).step(batch).copy()

    np.testing.assert_array_equal(noisy(2), noisy(2))
    assert not np.array_equal(noisy(2), noisy(3))


def test_toy_threshold_flip_rates():
    from experiments.noise_robustness import toy_flip_rates
    rates = toy_flip_rates((0.0, 1e-6, 1e-5, 1e-3), draws=20_000)
    assert rates[0.0] == 0.0 and rates[1e-6] == 0.0   # margin is 10 sigma
    assert 0.13 < rates[1e-5] < 0.19                    # P(z > 1) ~ 0.159
    assert 0.45 < rates[1e-3] < 0.55                    # a coin flip


@pytest.mark.parametrize("noise", [-1e-3, float("nan"), float("inf")])
def test_invalid_noise_is_rejected(noise):
    with pytest.raises(ValueError):
        RateNetwork(CIRCUIT, noise=noise)
