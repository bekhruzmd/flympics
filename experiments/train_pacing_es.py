"""Train the 400 m Pacing fly with an evolution strategy; connectome and dynamics stay frozen."""
from __future__ import annotations

from brain.controller import PACING_POLICY
from brain.pacing_io import (PACING_ACTIONS, PACING_INPUTS, PacingEncoder,
                             initial_pacing_encoder_weights)
from events.pacing import Pacing400m
from experiments.es_training import EventSpec, cli
from experiments.run_pacing import run, score

TRAIN_SEEDS = (0,)  # race is the same every time


def fitness(events: list[Pacing400m]) -> float:
    # trainer wants higher = better, race time is lower = better
    return -sum(e.score() for e in events) / len(events)


def teacher(observation: dict) -> int:
    return 1  # not used, imitation is off


def evaluate(controller) -> dict:
    records = run(controller=controller, seed=7)
    return {
        "time_or_score": score(records, Pacing400m()),
        "finished": records[-1]["position"] >= 400.0,
    }


SPEC = EventSpec(
    name="pacing",
    make_event=Pacing400m,
    encoder_class=PacingEncoder,
    inputs=PACING_INPUTS,
    actions=PACING_ACTIONS,
    initial_weights=initial_pacing_encoder_weights,
    initial_thresholds=(8.62e-06,),  # your trained threshold
    train_seeds=TRAIN_SEEDS,
    fitness=fitness,
    teacher=teacher,
    evaluate=evaluate,
    policy=PACING_POLICY.with_name("pacing_es.json"),
    encoder=PACING_POLICY.with_name("pacing_es_encoder.csv"),
)


def main() -> None:
    cli(
        SPEC,
        __doc__,
        generations=60,
        imitation_generations=0,
        pairs=24,
        weight_sigma=0.3,
        threshold_sigma=0.00001,
        lr=0.05,
    )


if __name__ == "__main__":
    main()
