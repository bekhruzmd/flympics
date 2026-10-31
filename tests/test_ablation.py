import numpy as np
import pytest

from brain.ablation import ablate, top_drivers
from brain.connectome import load_manc_front_leg_circuit
from brain.controller import FlyBrainController
from experiments.ablate_sprint import run_experiment
from experiments.run_sprint import run


def test_empty_ablation_matches_unchanged_controllers():
    circuit = ablate(load_manc_front_leg_circuit())
    assert run(seed=7, controller=FlyBrainController(circuit=circuit)) == run("fly")
    assert run(seed=7, controller=FlyBrainController.trained(circuit=circuit)) == run("trained")


def test_ablation_copies_circuit_and_removes_all_touching_edges():
    original = load_manc_front_leg_circuit()
    removed = [int(i) for i in original.indices("motor_extensor")[:2]]
    lesioned = ablate(original, removed)
    assert original.weights.nnz == 2770 and original.ablated == ()
    assert lesioned.ablated == tuple(sorted(removed))
    assert lesioned.weights[removed].nnz == 0 and lesioned.weights[:, removed].nnz == 0
    # Count stored edges: the vendored data include a few zero-weight entries.
    coo = original.weights.tocoo()
    touching = (np.isin(coo.row, removed) | np.isin(coo.col, removed)).sum()
    assert lesioned.weights.nnz == 2770 - touching


def test_ablated_neurons_stay_silent_even_when_stimulated():
    circuit = load_manc_front_leg_circuit()
    removed = [int(i) for i in circuit.indices("sensory")[:5]]
    controller = FlyBrainController(circuit=ablate(circuit, removed))
    controller.reset()
    for _ in range(20):
        controller.act({"distance_to_finish": 100.0, "velocity": 0.0})
        assert (controller.network.state[removed] == 0).all()
    assert controller.network.state[circuit.indices("sensory")[5:]].max() > 0


def test_no_sensory_input_means_no_acceleration():
    circuit = load_manc_front_leg_circuit()
    lesioned = ablate(circuit, circuit.indices("sensory"))
    records = run(seed=7, controller=FlyBrainController.trained(circuit=lesioned))
    assert all(row["action"] == "COAST" for row in records)


def test_edge_ablation_and_invalid_input():
    circuit = load_manc_front_leg_circuit()
    coo = circuit.weights.tocoo()
    first = int(np.flatnonzero(coo.data)[0])
    pre, post = int(coo.col[first]), int(coo.row[first])
    assert ablate(circuit, edges=[(pre, post)]).weights.nnz == 2769
    missing_post, missing_pre = np.argwhere(circuit.weights.toarray() == 0)[0]
    with pytest.raises(ValueError, match="edge"):
        ablate(circuit, edges=[(missing_pre, missing_post)])
    with pytest.raises(ValueError, match="indices"):
        ablate(circuit, [circuit.size])


def test_top_drivers_are_intrinsic_and_ranked():
    circuit = load_manc_front_leg_circuit()
    drivers = top_drivers(circuit, "motor_extensor", 5)
    assert set(drivers) <= set(circuit.indices("intrinsic"))
    targets = circuit.indices("motor_extensor")
    drive = [circuit.weights[targets][:, [i]].sum() for i in drivers]
    assert drive == sorted(drive, reverse=True)


def test_experiment_is_deterministic():
    first = run_experiment(k=3, seeds=(0, 1))
    assert first == run_experiment(k=3, seeds=(0, 1))
    control = [row for row in first if row["condition"] == "control"]
    assert {row["policy"]: row["finished"] for row in control} == {"untrained": False, "trained": True}
