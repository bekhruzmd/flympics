"""Fit an artificial motor threshold for 400 m Pacing; biological connectivity stays fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain.controller import PACING_POLICY, FlyBrainController
from brain.pacing_io import PACING_ACTIONS, PACING_INPUTS
from events.pacing import Pacing400m
from experiments.run_pacing import run, score


def train(seed: int = 7) -> dict:
    """Evaluate candidate motor thresholds and select the best one on 400 m Pacing.

    Biological connectivity, dynamics, and sensory encoding stay completely frozen.
    Only the artificial decision threshold between extensor and flexor readout changes.
    """
    thresholds = [0.015, *np.geomspace(0.1, 1e-8, 32).tolist(), 0.0]
    trials = []
    event = Pacing400m()
    for threshold in thresholds:
        controller = FlyBrainController.pacing(thresholds=(threshold,))
        records = run(controller=controller, event=event, seed=seed)
        final = records[-1]
        finished = final["position"] >= event.distance
        trial_score = score(records, event)
        trials.append({
            "motor_threshold": threshold,
            "score": round(trial_score, 8),
            "finished": finished,
            "time": final["simulation_time"],
            "distance": round(final["position"], 4),
            "final_energy": round(final["energy"], 4),
            "exhausted_steps": sum(r["exhausted"] for r in records),
        })

    best = min(trials, key=lambda trial: trial["score"])
    return {
        "version": 1,
        "event": "pacing",
        "method": "deterministic motor-threshold grid search",
        "encoder": "pacing_encoder_init.csv",
        "inputs": list(PACING_INPUTS),
        "actions": list(PACING_ACTIONS),
        "thresholds": [best["motor_threshold"]],
        "seed": seed,
        "best": best,
        "untrained": trials[0],
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=PACING_POLICY)
    args = parser.parse_args()

    policy = train(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")

    best = policy["best"]
    untrained = policy["untrained"]
    print(f"Untrained: {untrained['distance']:.1f} m in {untrained['time']:.1f} s, "
          f"score={untrained['score']:.2f}, threshold={untrained['motor_threshold']:.6g}")
    print(f"Trained:   {best['distance']:.1f} m in {best['time']:.1f} s, "
          f"score={best['score']:.2f}, threshold={best['motor_threshold']:.8g}")
    print(f"Saved policy and {len(policy['trials'])} trials to {args.output}")


if __name__ == "__main__":
    main()
