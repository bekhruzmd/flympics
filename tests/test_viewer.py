import numpy as np

from brain.connectome import load_manc_front_leg_circuit
from experiments.serve_view import viewer_payload
from experiments.run_sprint import run


def test_viewer_neurons_match_recorded_activity_and_connectivity():
    payload = viewer_payload(7)
    circuit = load_manc_front_leg_circuit()
    assert [node['id'] for node in payload['circuit']['nodes']] == list(circuit.neuron_ids)
    assert len(payload['circuit']['edges']) == circuit.weights.nnz
    for edge in payload['circuit']['edges']:
        assert edge['weight'] == circuit.weights[edge['post'], edge['pre']]
    for name in ('trained', 'fly'):
        rows = payload['runs'][name]
        activity = np.asarray([row['neuron_activity'] for row in rows])
        assert activity.shape == (len(rows), circuit.size)
        assert np.isfinite(activity).all()
        assert ((activity >= 0) & (activity <= 1)).all()
        for role, field in [('sensory', 'sensory_mean'), ('intrinsic', 'intrinsic_mean'),
                            ('motor_extensor', 'motor_extensor'), ('motor_flexor', 'motor_flexor')]:
            np.testing.assert_allclose(activity[:, circuit.indices(role)].mean(axis=1),
                                       [row[field] for row in rows])
        assert [row['action'] for row in rows] == [
            'ACCELERATE' if row['motor_difference'] > row['motor_threshold'] else 'COAST'
            for row in rows]
    assert 'neuron_activity' not in payload['runs']['rule'][0]
    assert 'neuron_activity' not in payload['runs']['random'][0]


def test_activity_recording_preserves_simulation():
    recorded = run('trained', record_activity=True)
    assert [{key: value for key, value in row.items()
             if key not in ('neuron_activity', 'motor_threshold')} for row in recorded] == run('trained')
