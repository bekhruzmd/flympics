"""Local-only website showing recorded output from the real Python simulations."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json

from experiments import run_archery, run_hurdles, run_pacing, run_steeplechase, run_swimming
from events.archery import Archery
from events.pacing import Pacing400m
from events.steeplechase import BARRIERS, WATER, Steeplechase
from experiments.run_sprint import run
from events.hurdles import Hurdles
from events.swimming import Swimming
from brain.connectome import load_manc_front_leg_circuit
from brain.controller import MODELS_DIR, FlyBrainController
from brain.eye import EYE_CALIBRATION, BlindfoldedEye, FlyVisWindEye, SwappedEye
import numpy as np


# Hurdles controllers: 'trained' is "Trained fly (evolution-trained)"
HURDLES_CONTROLLERS = ('trained', 'fly', 'rule', 'nocoast', 'random')
# The Hurdles fly whose MANC connection strengths were tuned too (gain 1.5, up to 2x drift).
HURDLES_CONNECTIONS_POLICY = MODELS_DIR / 'hurdles_connections_gain1.5_drift2x_long.json'
STEEPLECHASE_CONNECTIONS_POLICY = MODELS_DIR / 'steeplechase_connections_gain1.5_drift2x_long.json'
ARCHERY_CONNECTIONS_POLICY = MODELS_DIR / 'archery_connections_gain1.5_drift2x_long.json'
PACING_CONNECTIONS_POLICY = MODELS_DIR / 'pacing_connections_gain1.5_drift2x_long.json'
SWIMMING_TRAINED_POLICY = MODELS_DIR / 'swimming_gain1.5_long.json'
SWIMMING_CONNECTIONS_POLICY = MODELS_DIR / 'swimming_connections_gain1.5_drift2x_long.json'
SWIMMING_CONTROLLERS = ('fly', 'rule', 'full_strokes', 'random')
ARCHERY_CONTROLLERS = ('trained', 'fly', 'rule', 'random')
STEEPLECHASE_CONTROLLERS = ('trained', 'fly', 'rule', 'random')
PACING_CONTROLLERS = ('trained', 'fly', 'rule', 'random')


ARCHERY_EYE_POLICY = MODELS_DIR / 'archery_eye.json'
# The eye-trained Archery fly, and the same fly with its eye blindfolded or reversed.
EYE_VARIANTS = {'eye': FlyVisWindEye, 'eye_blind': BlindfoldedEye, 'eye_swapped': SwappedEye}


def event_run(module, name: str, seed: int, *args, controller=None) -> list[dict]:
    """An event runner's records, plus each step's neuron activity when a fly brain plays.

    Activity is rounded to two decimals to keep the page's download small; the run itself
    is unchanged (the activity is read after each decision, never fed back).
    """
    controller = controller if controller is not None else module.make_controller(name)
    states = None
    if isinstance(controller, FlyBrainController):
        states, act = [], controller.act

        def act_and_record(observation):
            action = act(observation)
            states.append(np.round(controller.network.state, 2).tolist())
            return action
        controller.act = act_and_record
    records = module.run(name, seed, *args, controller=controller)
    if states is not None:
        for record, state in zip(records, states):
            record['neuron_activity'] = state
    return records


def eye_fly(variant: str) -> FlyBrainController:
    controller = FlyBrainController.from_policy(ARCHERY_EYE_POLICY)
    controller.encoder.eye = EYE_VARIANTS[variant]()
    return controller


def viewer_payload(seed: int) -> dict:
    circuit = load_manc_front_leg_circuit()
    edges = circuit.weights.tocoo()
    return {
        'seed': seed,
        'circuit': {
            'nodes': [dict(id=node_id, role=role, cell_type=cell_type)
                      for node_id, role, cell_type in zip(circuit.neuron_ids, circuit.roles, circuit.cell_types)],
            'edges': [dict(pre=int(pre), post=int(post), weight=float(weight))
                      for pre, post, weight in zip(edges.col, edges.row, edges.data)],
        },
        'runs': {name: run(name, seed, record_activity=True)
                 for name in ('trained', 'fly', 'rule', 'random')},
        'events': {
            'pacing': {
                'distance': Pacing400m().distance,
                'dt': Pacing400m().dt,
                'runs': {
                    **{name: event_run(run_pacing, name, seed) for name in PACING_CONTROLLERS},
                    'connections': event_run(run_pacing, 'connections', seed,
                                             controller=FlyBrainController.from_policy(PACING_CONNECTIONS_POLICY)),
                },
            },
            'hurdles': {
                'distance': Hurdles().distance,
                'dt': Hurdles().dt,
                'hurdles': list(run_hurdles.course('benchmark', seed)),
                'runs': {
                    **{name: event_run(run_hurdles, name, seed, 'benchmark') for name in HURDLES_CONTROLLERS},
                    'connections': event_run(run_hurdles, 'connections', seed, 'benchmark',
                                             controller=FlyBrainController.from_policy(HURDLES_CONNECTIONS_POLICY)),
                },
            },
            'swimming': {
                'distance': Swimming().distance,
                'dt': Swimming().dt,
                'pool_length': Swimming().pool_length,
                'runs': {
                    **{name: event_run(run_swimming, name, seed) for name in SWIMMING_CONTROLLERS},
                    'trained': event_run(run_swimming, 'trained', seed,
                                         controller=FlyBrainController.from_policy(SWIMMING_TRAINED_POLICY)),
                    'connections': event_run(run_swimming, 'connections', seed,
                                             controller=FlyBrainController.from_policy(SWIMMING_CONNECTIONS_POLICY)),
                },
            },
            'steeplechase': {
                'distance': Steeplechase().distance,
                'dt': Steeplechase().dt,
                'hurdles': list(BARRIERS),
                'water': sorted(WATER),
                'pit_length': Steeplechase().pit_length,
                'runs': {
                    **{name: event_run(run_steeplechase, name, seed) for name in STEEPLECHASE_CONTROLLERS},
                    'connections': event_run(run_steeplechase, 'connections', seed,
                                             controller=FlyBrainController.from_policy(STEEPLECHASE_CONNECTIONS_POLICY)),
                },
            },
            'archery': {
                'dt': Archery().dt,
                'arrows': Archery().arrows,
                'max_points': Archery().arrows * Archery().rings,
                'ring_width': Archery().ring_width,
                'rings': Archery().rings,
                'runs': {**{name: event_run(run_archery, name, seed) for name in ARCHERY_CONTROLLERS},
                         **{name: event_run(run_archery, name, seed, controller=eye_fly(name)) for name in EYE_VARIANTS},
                         'connections': event_run(run_archery, 'connections', seed,
                                                  controller=FlyBrainController.from_policy(ARCHERY_CONNECTIONS_POLICY))},
                'eye': json.loads(EYE_CALIBRATION.read_text()),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    payload = json.dumps(viewer_payload(args.seed), allow_nan=False).encode()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/runs.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                super().do_GET()

    directory = Path(__file__).resolve().parents[1] / 'visualization' / 'web'
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(directory)))
    print(f'Flympics website: http://localhost:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
