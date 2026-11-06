"""Train the Steeplechase fly with an evolution strategy; connectome and dynamics stay frozen."""
from __future__ import annotations

from brain.controller import STEEPLECHASE_POLICY
from brain.steeplechase_io import (STEEPLECHASE_ACTIONS, STEEPLECHASE_INITIAL_THRESHOLDS,
                                   STEEPLECHASE_INPUTS, SteeplechaseEncoder,
                                   initial_steeplechase_encoder_weights)
from controllers.steeple_baselines import SteepleRuleController
from events.steeplechase import Steeplechase
from experiments.es_training import EventSpec, cli
from experiments.run_steeplechase import run, score


_RULE = SteepleRuleController()


def teacher(observation: dict) -> int:
    """The rule's choice as a decoder band."""
    return STEEPLECHASE_ACTIONS.index(_RULE.act(observation))


def fitness(events: list[Steeplechase]) -> float:
    """Minus the race score: faster finishes, or more metres before the time limit, are better."""
    return -events[0].score()


def evaluate(controller) -> dict:
    """The fixed race, with the ordinary (unbatched) controller."""
    records = run(controller=controller)
    final = records[-1]
    return {
        "score": round(score(records), 3),
        "finished": final["position"] >= 600.0,
        "clears": final["clears"],
        "hits": final["hits"],
        "splashes": final["splashes"],
        "stamina_left": round(final["stamina"], 3),
    }


SPEC = EventSpec(
    name="steeplechase", make_event=Steeplechase, encoder_class=SteeplechaseEncoder,
    inputs=STEEPLECHASE_INPUTS, actions=STEEPLECHASE_ACTIONS,
    initial_weights=initial_steeplechase_encoder_weights,
    initial_thresholds=STEEPLECHASE_INITIAL_THRESHOLDS, train_seeds=(0,), fitness=fitness, teacher=teacher,
    evaluate=evaluate, policy=STEEPLECHASE_POLICY,
    encoder=STEEPLECHASE_POLICY.with_name("steeplechase_encoder.csv"),
)


def main() -> None:
    cli(SPEC, __doc__, generations=200, imitation_generations=150, pairs=24, weight_sigma=0.3, threshold_sigma=0.02, lr=0.05)


if __name__ == "__main__":
    main()
