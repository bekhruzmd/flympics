"""Write the viewer's recorded runs to a static runs.json, for hosting the page without Python."""
from pathlib import Path
import argparse
import json

from experiments.serve_view import viewer_payload

WEB_DIR = Path(__file__).resolve().parents[1] / 'visualization' / 'web'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=WEB_DIR / 'runs.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(viewer_payload(args.seed), allow_nan=False))
    print(f'Wrote {args.output} (seed {args.seed})')


if __name__ == '__main__':
    main()
