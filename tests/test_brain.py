import numpy as np

from brain.connectome import load_manc_front_leg_circuit
from brain.dynamics import RateNetwork
from brain.motor import MotorDecoder
from brain.sensory import SensoryEncoder


def test_real_derived_connectome_loads_sparse():
    circuit = load_manc_front_leg_circuit()
    assert circuit.size == 135
    assert circuit.weights.nnz == 2770
    assert len(circuit.indices("sensory")) == 24


def test_encoder_network_and_decoder_dimensions():
    circuit = load_manc_front_leg_circuit()
    stimulus = SensoryEncoder(circuit).encode({"distance_to_finish": 100.0, "velocity": 0.0})
    assert stimulus.shape == (circuit.size,)
    activity = RateNetwork(circuit).step(stimulus)
    assert activity.shape == (circuit.size,)
    action, summary = MotorDecoder(circuit).decode(activity)
    assert action in {"COAST", "ACCELERATE"}
    assert set(summary) == {"extensor", "flexor", "difference"}
    assert np.isfinite(activity).all()
