"""Train Sprint with the FlyVis speed eye and compare it with the existing controllers."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from brain.controller import FlyBrainController, SPRINT_EYE_POLICY
from brain.sensory import initial_encoder_weights
from brain.speed_eye import SPRINT_EYE_ACTIONS, SPRINT_EYE_INPUTS, SprintEyeEncoder, blindfolded_eye
from events.sprint import Sprint
from experiments.es_training import EventSpec, save_history, train
from experiments.run_sprint import run

SEED = 7


def fitness(events: list[Sprint]) -> float:
    return -sum(event.score() for event in events) / len(events)


def evaluate(controller: FlyBrainController) -> dict:
    final = run(seed=SEED, controller=controller)[-1]
    return {
        "time": final["simulation_time"],
        "finished": final["distance_to_finish"] == 0.0,
        "distance": final["position"],
    }


def always_accelerate_band(_observation: dict) -> int:
    return 1


SPEC = EventSpec(
    name="sprint_eye",
    make_event=Sprint,
    encoder_class=SprintEyeEncoder,
    inputs=SPRINT_EYE_INPUTS,
    actions=SPRINT_EYE_ACTIONS,
    initial_weights=initial_encoder_weights,
    initial_thresholds=(3e-5,),  # Start near the motor differences of the saturated circuit.
    train_seeds=(SEED,),  # Sprint is deterministic.
    fitness=fitness,
    teacher=always_accelerate_band,  # Required by EventSpec; imitation is disabled.
    evaluate=evaluate,
    policy=SPRINT_EYE_POLICY,
    encoder=SPRINT_EYE_POLICY.with_name("sprint_eye_encoder.csv"),
)


def compare(path: Path = SPRINT_EYE_POLICY) -> dict:
    eye_fly = FlyBrainController.from_policy(path)
    circuit = eye_fly.circuit
    eye_result = evaluate(eye_fly)
    eye_fly.encoder.eye = blindfolded_eye
    return {
        "untrained": evaluate(FlyBrainController(circuit=circuit)),
        "existing_trained": evaluate(FlyBrainController.trained(circuit=circuit)),
        "eye": eye_result,
        "blindfolded": evaluate(eye_fly),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--weight-sigma", type=float, default=0.3)
    parser.add_argument("--threshold-sigma", type=float, default=0.00001)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Seeds the ES noise")
    parser.add_argument("--imitation-generations", type=int, choices=(0,), default=0,
                        help="Sprint has two actions, so skip the three-band imitation stage")
    parser.add_argument("--policy", type=Path, default=SPEC.policy)
    parser.add_argument("--results-dir", type=Path, default=Path("results/sprint_eye_training"))
    parser.add_argument("--compare", action="store_true", help="Run the four-way comparison from a saved policy")
    args = parser.parse_args()

    if not args.compare:
        spec = replace(SPEC, policy=args.policy, encoder=args.policy.with_name(f"{args.policy.stem}_encoder.csv"))
        policy, history = train(spec, args.generations, args.pairs, args.weight_sigma,
                                args.threshold_sigma, args.lr, args.seed, args.imitation_generations)
        policy["method"] = ("OpenAI-ES (antithetic sampling, centred-rank fitness, Adam) on encoder "
                            "weights and the motor threshold, with Sprint score as the objective. "
                            "Connectome and dynamics frozen.")
        policy["comparison_seed"] = SEED
        policy["comparison"] = compare(spec.policy)
        spec.policy.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
        save_history(history, args.results_dir, "Sprint speed eye: evolution-strategy training")
        print(f"Policy: {spec.policy}; encoder: {spec.encoder}")
        results = policy["comparison"]
    else:
        results = compare(args.policy)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
