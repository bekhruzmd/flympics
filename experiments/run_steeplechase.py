"""Run the Flympics 600 m Steeplechase (12 barriers, 4 water jumps, limited stamina)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import STEEPLECHASE_POLICY, FlyBrainController
from controllers.base import Controller
from controllers.steeple_baselines import AlwaysController, RandomController, SteepleRuleController
from events.steeplechase import ACCELERATE, COAST, Steeplechase

CONTROLLERS = ("rule", "always_accelerate", "always_coast", "random", "fly", "trained")


def make_controller(name: str) -> Controller:
    return {
        "rule": SteepleRuleController,
        "always_accelerate": lambda: AlwaysController(ACCELERATE),
        "always_coast": lambda: AlwaysController(COAST),
        "random": RandomController,
        "fly": FlyBrainController.steeplechase,
        "trained": lambda: FlyBrainController.from_policy(STEEPLECHASE_POLICY),
    }[name]()


def run(controller_name: str = "rule", seed: int = 7, *,
        controller: Controller | None = None, event: Steeplechase | None = None) -> list[dict]:
    """One fixed race. The seed only seeds the controller (e.g. random)."""
    event = event if event is not None else Steeplechase()
    if controller is None:
        controller = make_controller(controller_name)
    event.reset(seed)
    controller.reset(seed)
    records: list[dict] = []
    while not event.is_finished():
        action = controller.act(event.observe())
        observation = event.step(action)
        records.append({
            "step": event.steps,
            "simulation_time": round(event.time, 8),
            "position": event.position,
            "velocity": event.velocity,
            "stamina": event.stamina,
            "distance_to_barrier": observation["distance_to_barrier"],
            "water_next": observation["water_next"],
            "height": event.height,
            "airborne": event.airborne,
            "hits": event.hits,
            "clears": event.clears,
            "splashes": event.splashes,
            "jumps": event.jumps,
            "action": action,
        })
        if isinstance(controller, FlyBrainController):
            records[-1].update(controller.last_summary)
    return records


def score(records: list[dict], event: Steeplechase | None = None) -> float:
    """Event score recomputed from the last record (same formula as Steeplechase.score)."""
    event = event if event is not None else Steeplechase()
    final = records[-1]
    if final["position"] >= event.distance:
        return final["simulation_time"]
    return event.max_time + (event.distance - final["position"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, default="rule")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seeds random controllers only; the race itself is fixed.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip the distance, speed and stamina plot")
    args = parser.parse_args()
    records = run(args.controller, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"steeplechase_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if not args.no_plot:
        from visualization.steeplechase_view import save_steeplechase_plot
        save_steeplechase_plot(records, stem.with_suffix(".png"))
    final = records[-1]
    print(f"{args.controller}: {final['position']:.1f} m in {final['simulation_time']:.1f} s, "
          f"score {score(records):.2f}; {final['clears']} cleared, {final['hits']} hit, "
          f"{final['splashes']} splash(es), stamina left {final['stamina']:.2f}. Logs: {stem}.json/.csv")


if __name__ == "__main__":
    main()
