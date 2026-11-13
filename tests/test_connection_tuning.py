"""Training MANC connection strengths: the wiring map, signs and data stay fixed."""
import json

import numpy as np
import pytest

from brain.connectome import load_manc_front_leg_circuit
from brain.controller import FlyBrainController
from brain.evolution import BatchedRateNetwork, ConnectionDrift, batched_rollout, load_connections, save_connections
from brain.hurdles_io import HURDLES_ACTIONS, HurdlesEncoder, initial_hurdles_encoder_weights
from brain.reward_learning import DRIFT_LIMITS, tuned_circuit
from events.hurdles import Hurdles
from experiments.es_training import fingerprint
from experiments.run_hurdles import run

CIRCUIT = load_manc_front_leg_circuit()


@pytest.mark.parametrize("drift", sorted(DRIFT_LIMITS))
def test_multipliers_start_at_one_and_never_leave_the_limit(drift):
    low, high = DRIFT_LIMITS[drift]
    layout = ConnectionDrift(CIRCUIT, low, high)
    np.testing.assert_allclose(layout.multipliers(np.full(layout.size, layout.z0)), 1.0)
    extreme = layout.multipliers(np.array([-1e3, -5.0, 0.0, 5.0, 1e3]))
    assert extreme.min() >= low and extreme.max() <= high


def test_tuned_matrices_keep_the_wiring_map_and_signs():
    layout = ConnectionDrift(CIRCUIT, 0.5, 1.5)
    z = np.random.default_rng(0).normal(0.0, 3.0, (3, layout.size))
    base = CIRCUIT.weights.toarray().T
    for matrix in layout.matrices(z):
        assert np.array_equal(matrix != 0, base != 0)  # no new or removed connections
        assert np.all(matrix >= 0)
    assert fingerprint(CIRCUIT) == fingerprint(load_manc_front_leg_circuit())  # MANC data untouched


def test_untuned_member_matches_the_plain_network():
    layout = ConnectionDrift(CIRCUIT, 0.5, 1.5)
    matrices = layout.matrices(np.full((2, layout.size), layout.z0))
    stimulus = np.random.default_rng(1).uniform(0, 1, (4, CIRCUIT.size))
    plain, tuned = BatchedRateNetwork(CIRCUIT, 4), BatchedRateNetwork(CIRCUIT, 4, member_weights_t=matrices)
    # Batched and plain matrix products may round the last bit differently on some BLAS builds.
    for _ in range(5):
        np.testing.assert_allclose(tuned.step(stimulus), plain.step(stimulus), rtol=1e-12, atol=1e-12)


def test_batched_tuned_race_matches_the_saved_controller(tmp_path):
    """A tuned member raced in the batch must finish exactly like the controller built from
    its saved connections file (sparse network, ordinary run loop)."""
    layout = ConnectionDrift(CIRCUIT, 0.5, 1.5)
    z = np.random.default_rng(2).normal(layout.z0, 1.0, layout.size)
    multipliers = layout.multipliers(z)
    path = tmp_path / "connections.csv"
    save_connections(path, CIRCUIT, multipliers)
    np.testing.assert_allclose(load_connections(path, CIRCUIT), multipliers, rtol=1e-9)

    weights = initial_hurdles_encoder_weights(CIRCUIT)
    thresholds = (0.015, 0.30)
    encoder = HurdlesEncoder(CIRCUIT, weights)
    event, = batched_rollout(CIRCUIT, lambda: Hurdles(seeded=True), [3], encoder, HURDLES_ACTIONS,
                             weights[None], np.array([thresholds]), gain=1.5,
                             member_weights_t=layout.matrices(z[None]))[0]
    fly = FlyBrainController.band_event("hurdles", weights, list(thresholds),
                                        tuned_circuit(CIRCUIT, load_connections(path, CIRCUIT)), gain=1.5)
    records = run(controller=fly, seed=3, layout="seeded")
    assert records[-1]["position"] == pytest.approx(event.position, abs=1e-9)
    assert records[-1]["simulation_time"] == pytest.approx(event.time, abs=1e-9)


def test_from_policy_loads_the_tuned_connections(tmp_path):
    layout = ConnectionDrift(CIRCUIT, 0.8, 1.2)
    multipliers = layout.multipliers(np.linspace(-4, 4, layout.size))
    save_connections(tmp_path / "c.csv", CIRCUIT, multipliers)
    from brain.sensory import save_encoder_weights
    from brain.hurdles_io import HURDLES_INPUTS
    save_encoder_weights(tmp_path / "e.csv", CIRCUIT, initial_hurdles_encoder_weights(CIRCUIT), HURDLES_INPUTS)
    (tmp_path / "p.json").write_text(json.dumps({
        "version": 1, "event": "hurdles", "encoder": "e.csv", "thresholds": [0.015, 0.3],
        "gain": 1.5, "connections": {"file": "c.csv"}}))
    fly = FlyBrainController.from_policy(tmp_path / "p.json", CIRCUIT)
    np.testing.assert_allclose(fly.circuit.weights.data, CIRCUIT.weights.data * multipliers)
    assert fly.network.gain == 1.5


def test_connections_file_must_match_the_circuit(tmp_path):
    path = tmp_path / "c.csv"
    save_connections(path, CIRCUIT, np.ones(CIRCUIT.weights.nnz))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:1] + [lines[2], lines[1]] + lines[3:]) + "\n")
    with pytest.raises(ValueError):
        load_connections(path, CIRCUIT)


def test_one_threshold_fit_maximises_agreement():
    from experiments.es_training import fit_one_threshold
    differences = np.array([-0.3, -0.1, 0.0, 0.2, 0.4])
    labels = np.array([0, 0, 0, 1, 1])
    agreement, (threshold,) = fit_one_threshold(differences, labels, np.ones(5))
    assert agreement == 5.0
    assert 0.0 < threshold < 0.2  # everything at or below it is band 0, above it band 1
