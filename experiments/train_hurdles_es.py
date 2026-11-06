"""Train the Hurdles fly with an evolution strategy; connectome and dynamics stay frozen."""
from __future__ import annotations

import numpy as np

from brain.controller import HURDLES_POLICY
from brain.hurdles_io import (HURDLES_ACTIONS, HURDLES_INITIAL_THRESHOLDS, HURDLES_INPUTS,
                              HurdlesEncoder, initial_hurdles_encoder_weights)
from controllers.hurdle_baselines import HurdleRuleController
from events.hurdles import Hurdles
from experiments.es_training import EventSpec, cli
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


SPEC = EventSpec(
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
    policy=HURDLES_POLICY,
    encoder=HURDLES_POLICY.with_name("hurdles_es_encoder.csv"),
)


def main() -> None:
    cli(SPEC, __doc__, generations=60, imitation_generations=150, pairs=24, weight_sigma=0.3, threshold_sigma=0.02, lr=0.05)


if __name__ == "__main__":
    main()
