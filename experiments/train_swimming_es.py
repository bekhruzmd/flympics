"""Train the Swimming fly with an evolution strategy.

The fly decides each step whether to PULL (power stroke) or RECOVER, so timing the stroke
is the skill. Stage 1 copies the rule baseline's stroke timing on recorded states, then
stage 2 optimises the race. The encoder weights and the motor threshold learn; with --drift
the MANC connection strengths learn too. --gain sets the neuron gain, a modeling choice. The race
is one fixed 100 m course, so there are no held-out courses.

    python -m experiments.train_swimming_es --gain 1.5 --policy models/swimming_gain1.5.json
"""
from __future__ import annotations

from brain.controller import MODELS_DIR
from controllers.swim_baselines import SwimRuleController
from brain.swimming_io import (SWIMMING_ACTIONS, SWIMMING_INPUTS, SwimmingEncoder,
                               initial_swimming_encoder_weights)
from events.swimming import Swimming
from experiments.es_training import EventSpec, cli
from experiments.run_swimming import run, score

TRAIN_SEEDS = (0,)  # the race is the same every time


def fitness(events: list[Swimming]) -> float:
    """Minus the race score: faster finishes, or more metres before the time limit, are better."""
    return -sum(e.score() for e in events) / len(events)


class RuleTeacher:
    """The rule baseline's choice as a band (0 RECOVER, 1 PULL). It remembers whether it is
    mid-pull, so it is reset at the start of every recorded race."""

    def __init__(self):
        self.rule = SwimRuleController()
        self.rule.reset(0)

    def reset(self, seed: int | None = None) -> None:
        self.rule.reset(seed)

    def __call__(self, observation: dict) -> int:
        return SWIMMING_ACTIONS.index(self.rule.act(observation))


teacher = RuleTeacher()


def evaluate(controller) -> dict:
    """The fixed race, with the ordinary (unbatched) controller."""
    records = run(controller=controller)
    final = records[-1]
    return {
        "score": round(score(records), 3),
        "finished": final["position"] >= Swimming().distance,
        "time": round(final["simulation_time"], 2),
        "position": round(final["position"], 2),
        "strokes": final.get("strokes"),
        "turns": final.get("turns"),
    }


SPEC = EventSpec(
    name="swimming",
    make_event=Swimming,
    encoder_class=SwimmingEncoder,
    inputs=SWIMMING_INPUTS,
    actions=SWIMMING_ACTIONS,
    initial_weights=initial_swimming_encoder_weights,
    initial_thresholds=(0.015,),  # the untrained Swimming fly's threshold
    train_seeds=TRAIN_SEEDS,
    fitness=fitness,
    teacher=teacher,
    evaluate=evaluate,
    policy=MODELS_DIR / "swimming_es.json",
    encoder=MODELS_DIR / "swimming_es_encoder.csv",
)


def main() -> None:
    cli(SPEC, __doc__, generations=60, imitation_generations=150, pairs=24,
        weight_sigma=0.3, threshold_sigma=0.02, lr=0.05)


if __name__ == "__main__":
    main()
