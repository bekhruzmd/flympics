"""Train the Archery fly with an evolution strategy; connectome and dynamics stay frozen."""
from __future__ import annotations

import numpy as np

from brain.archery_io import (ARCHERY_ACTIONS, ARCHERY_INITIAL_THRESHOLDS, ARCHERY_INPUTS, ArcheryEncoder,
                              initial_archery_encoder_weights)
from brain.controller import ARCHERY_POLICY
from controllers.archery_baselines import ArcheryRuleController
from events.archery import HOLD, RELEASE, Archery
from experiments.es_training import EventSpec, cli
from experiments.run_archery import run, summary

TRAIN_SEEDS = tuple(range(12))
# Held-out winds: never seen in training.
VALIDATION_SEEDS = tuple(range(1000, 1100))
TIMEOUT_REWARD = -2.0


def arrow_reward(event: Archery, landing: float | None) -> float:
    """Points without the floor(): 10 at the centre, one less per ring width. Smooth for the ES."""
    if landing is None:
        return TIMEOUT_REWARD
    return max(TIMEOUT_REWARD, event.rings - abs(landing) / event.ring_width)


_RULE = ArcheryRuleController()


def teacher(observation: dict) -> int:
    """The rule's choice as a decoder band; its HOLD (bow not drawn yet) is the RELEASE band."""
    action = _RULE.act(observation)
    return ARCHERY_ACTIONS.index(RELEASE if action == HOLD else action)


def fitness(events: list[Archery]) -> float:
    return float(np.mean([sum(arrow_reward(e, x) for x in e.landings) for e in events]))


def evaluate(controller, seeds=VALIDATION_SEEDS) -> dict:
    """Mean points per end on held-out winds, with the ordinary (unbatched) controller."""
    ends = [summary(run(seed=s, controller=controller)) for s in seeds]
    return {
        "validation_seeds": f"{seeds[0]}-{seeds[-1]}",
        "mean_points": round(float(np.mean([e["points"] for e in ends])), 3),
        "min_points": min(e["points"] for e in ends),
        "perfect_tens_rate": round(float(np.mean([p == 10 for e in ends for p in e["arrow_points"]])), 3),
        "timeouts": sum(e["timeouts"] for e in ends),
        "mean_end_time": round(float(np.mean([e["time"] for e in ends])), 2),
    }


SPEC = EventSpec(
    name="archery", make_event=Archery, encoder_class=ArcheryEncoder, inputs=ARCHERY_INPUTS,
    actions=ARCHERY_ACTIONS, initial_weights=initial_archery_encoder_weights,
    initial_thresholds=ARCHERY_INITIAL_THRESHOLDS, train_seeds=TRAIN_SEEDS, fitness=fitness, teacher=teacher,
    evaluate=evaluate, policy=ARCHERY_POLICY, encoder=ARCHERY_POLICY.with_name("archery_encoder.csv"),
)


def main() -> None:
    cli(SPEC, __doc__, generations=60, imitation_generations=150, pairs=24, weight_sigma=0.3, threshold_sigma=0.02, lr=0.05)


if __name__ == "__main__":
    main()
