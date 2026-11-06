import numpy as np
import pytest

from brain.archery_io import ARCHERY_INITIAL_THRESHOLDS, initial_archery_encoder_weights
from brain.connectome import load_manc_front_leg_circuit
from brain.eye import ArcheryEyeEncoder, FlyVisWindEye
from brain.reward_learning import DRIFT_LIMITS, LearningSettings, RewardModulatedFly, food, tuned_circuit
from events.archery import Archery
from experiments.es_training import fingerprint
from experiments.train_archery_food import practise_end


@pytest.fixture(scope="module")
def circuit():
    return load_manc_front_leg_circuit()


def make_fly(circuit, drift="50", seed=0, **settings):
    encoder = ArcheryEyeEncoder(circuit, initial_archery_encoder_weights(circuit))
    return RewardModulatedFly(circuit, encoder, ARCHERY_INITIAL_THRESHOLDS,
                              LearningSettings(drift=drift, noise=0.1, gain=1.0, **settings), seed)


def test_tuned_circuit_keeps_the_wiring_map_and_leaves_the_data_alone(circuit):
    before = fingerprint(circuit)
    multipliers = np.linspace(0.5, 1.5, circuit.weights.nnz)
    tuned = tuned_circuit(circuit, multipliers)
    assert fingerprint(circuit) == before
    assert np.array_equal(tuned.weights.indices, circuit.weights.indices)
    assert np.array_equal(tuned.weights.indptr, circuit.weights.indptr)
    assert tuned.weights.shape == circuit.weights.shape
    np.testing.assert_allclose(tuned.weights.data, circuit.weights.data * multipliers)
    assert tuned.neuron_ids == circuit.neuron_ids and tuned.roles == circuit.roles


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_tuned_circuit_refuses_multipliers_that_could_flip_or_cut_a_connection(circuit, bad):
    multipliers = np.ones(circuit.weights.nnz)
    multipliers[0] = bad
    with pytest.raises(ValueError):
        tuned_circuit(circuit, multipliers)


@pytest.mark.parametrize("drift", sorted(DRIFT_LIMITS))
def test_learning_stays_inside_the_drift_limit_and_keeps_signs(circuit, drift):
    fly = make_fly(circuit, drift=drift, weight_rate=5.0)  # a huge rate pushes against the limits
    for seed in range(5):
        practise_end(fly, seed)
    low, high = DRIFT_LIMITS[drift]
    m = fly.multipliers[fly.base > 0]
    assert m.min() >= low - 1e-12 and m.max() <= high + 1e-12
    assert np.all(fly.strength >= 0)
    assert np.all(fly.strength[fly.base == 0] == 0)  # zero-strength connections stay zero
    assert fingerprint(circuit) == fingerprint(load_manc_front_leg_circuit())


def test_no_surprise_means_no_change(circuit):
    fly = make_fly(circuit)
    event = Archery(); observation = event.reset(0)
    for _ in range(5):
        fly.act(observation)
    fly.reward(0.7)                 # first meal sets "usual food"; surprise is zero
    np.testing.assert_array_equal(fly.strength, fly.base)


def test_practice_is_reproducible(circuit):
    a, b = make_fly(circuit, seed=3), make_fly(circuit, seed=3)
    for seed in range(3):
        assert practise_end(a, seed) == practise_end(b, seed)
    np.testing.assert_array_equal(a.strength, b.strength)
    np.testing.assert_array_equal(a.inputs, b.inputs)


def test_food():
    assert food(None) == 0.0
    assert food(0) == 0.0
    assert food(7) == pytest.approx(0.7)
    assert food(10) == pytest.approx(2.0)  # bullseye bonus


def test_flyvis_eye_is_centred_monotonic_and_clamped():
    eye = FlyVisWindEye()
    assert eye(0.0) == pytest.approx(0.0, abs=1e-12)
    winds = np.linspace(-4, 4, 33)
    values = np.array([eye(w) for w in winds])
    # Stronger rightward wind gives a lower T4/T5 a-b signal. The measured curve has small
    # bumps from FlyVis's response noise, so check the trend rather than every step.
    assert np.corrcoef(winds, values)[0, 1] < -0.97  # measured: about -0.98
    assert np.mean(np.diff(values) < 0) > 0.8
    assert np.max(np.abs(values)) == pytest.approx(1.0)
    assert eye(10.0) == eye(4.0) and eye(-10.0) == eye(-4.0)
