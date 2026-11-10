"""Train the Hurdles fly with an evolution strategy.

By default the encoder weights and band thresholds learn and the connectome stays frozen.
With --drift, the strengths of the existing MANC connections learn too, each kept within
the drift limit of its measured value ("Connection-tuned fly"); the wiring map never
changes. --gain sets the neuron gain, a modeling choice (default 3.0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain.controller import HURDLES_POLICY, MODELS_DIR, FlyBrainController
from brain.hurdles_io import (HURDLES_ACTIONS, HURDLES_INITIAL_THRESHOLDS, HURDLES_INPUTS,
                              HurdlesEncoder, initial_hurdles_encoder_weights)
from controllers.hurdle_baselines import HurdleRuleController
from events.hurdles import Hurdles
from experiments.es_training import EventSpec, save_history, train
from brain.reward_learning import DRIFT_LIMITS
from experiments.run_hurdles import run, score

TRAIN_SEEDS = tuple(range(12))
# Held-out courses: never seen in training.
HELD_OUT_SEEDS = tuple(range(1000, 1050))

_RULE = HurdleRuleController()


def teacher(observation: dict) -> int:
    """The rule's choice as a decoder band."""
    return HURDLES_ACTIONS.index(_RULE.act(observation))


def fitness(events: list[Hurdles]) -> float:
    """Minus the race score: faster finishes, or more metres before the time limit, are better."""
    return -float(np.mean([e.score() for e in events]))


def evaluate(controller, seeds=HELD_OUT_SEEDS) -> dict:
    """Evaluation on the benchmark course and held-out seeded courses."""
    bench_records = run(controller=controller, layout="benchmark")
    bench_score = score(bench_records)
    bench_final = bench_records[-1]

    held_out = [run(controller=controller, seed=s, layout="seeded") for s in seeds]
    held_out_scores = [score(r) for r in held_out]

    return {
        "benchmark_score": round(bench_score, 3),
        "benchmark_finished": bench_final["position"] >= 200.0,
        "benchmark_time": round(bench_final["simulation_time"], 3),
        "benchmark_hits": bench_final["hits"],
        "benchmark_jumps": bench_final["jumps"],
        "benchmark_coasts": bench_final["coasts"],
        "held_out_seeds": f"{seeds[0]}-{seeds[-1]}",
        "held_out_mean_score": round(float(np.mean(held_out_scores)), 3),
        "held_out_min_score": round(float(np.min(held_out_scores)), 3),
        "held_out_max_score": round(float(np.max(held_out_scores)), 3),
        "held_out_finished_rate": round(float(np.mean([r[-1]["position"] >= 200.0 for r in held_out])), 3),
        "held_out_mean_hits": round(float(np.mean([r[-1]["hits"] for r in held_out])), 2),
        "held_out_mean_jumps": round(float(np.mean([r[-1]["jumps"] for r in held_out])), 2),
        "held_out_mean_coasts": round(float(np.mean([r[-1]["coasts"] for r in held_out])), 2),
    }


def _gain_label(gain: float) -> str:
    return str(int(gain)) if float(gain).is_integer() else f"{gain:g}"


def make_spec(policy_path: Path | None = None, gain: float = 3.0, drift: str | None = None) -> EventSpec:
    if policy_path is None:
        if drift is not None:
            policy_path = MODELS_DIR / f"hurdles_connections_gain{_gain_label(gain)}_drift{drift}.json"
        elif gain == 3.0:
            policy_path = HURDLES_POLICY
        else:
            policy_path = MODELS_DIR / f"hurdles_es_gain{_gain_label(gain)}.json"
    encoder_path = policy_path.with_name(f"{policy_path.stem}_encoder.csv")
    return EventSpec(
        name="hurdles",
        make_event=lambda: Hurdles(seeded=True),
        encoder_class=HurdlesEncoder,
        inputs=HURDLES_INPUTS,
        actions=HURDLES_ACTIONS,
        initial_weights=initial_hurdles_encoder_weights,
        initial_thresholds=HURDLES_INITIAL_THRESHOLDS,
        train_seeds=TRAIN_SEEDS,
        fitness=fitness,
        teacher=teacher,
        evaluate=evaluate,
        policy=policy_path,
        encoder=encoder_path,
    )


SPEC = make_spec(HURDLES_POLICY, gain=3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Rate network gain (default 3.0). Modeling choice to avoid saturation.")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--imitation-generations", type=int, default=150,
                        help="Stage-1 generations imitating the rule baseline (0 skips it)")
    parser.add_argument("--pairs", type=int, default=24, help="Antithetic pairs per generation")
    parser.add_argument("--weight-sigma", type=float, default=0.3)
    parser.add_argument("--threshold-sigma", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.05, help="Adam step for encoder weights")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the ES noise")
    parser.add_argument("--drift", choices=sorted(DRIFT_LIMITS), default=None,
                        help="Also train MANC connection strengths within this limit: "
                             "20 = x0.8-1.2, 50 = x0.5-1.5, 2x = x0.5-2")
    parser.add_argument("--connection-sigma", type=float, default=0.3,
                        help="ES noise on the connection parameters (before the bounding sigmoid)")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    spec = make_spec(args.policy, gain=args.gain, drift=args.drift)
    results_dir = args.results_dir or Path("results") / f"{spec.policy.stem}_training"
    connections = spec.policy.with_name(f"{spec.policy.stem}_connections.csv") if args.drift else None

    policy, history = train(spec, args.generations, args.pairs, args.weight_sigma,
                            args.threshold_sigma, args.lr, args.seed, args.imitation_generations,
                            gain=args.gain, drift=DRIFT_LIMITS[args.drift] if args.drift else None,
                            connection_sigma=args.connection_sigma, connections=connections)
    policy_data = json.loads(spec.policy.read_text())
    policy_data["gain"] = args.gain
    spec.policy.write_text(json.dumps(policy_data, indent=2, allow_nan=False) + "\n")

    save_history(history, results_dir, f"{spec.name.title()} (gain {args.gain}): evolution-strategy training")
    print(f"Trained in {policy['training_seconds']} s. Trained: {policy['trained']}")
    print(f"Untrained: {policy['untrained']}")
    print(f"Policy: {spec.policy}; encoder: {spec.encoder}; history: {results_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
