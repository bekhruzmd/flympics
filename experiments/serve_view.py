"""Local-only website showing recorded output from the real Python simulations."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json

from experiments.run_sprint import run
from brain.connectome import load_manc_front_leg_circuit


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
