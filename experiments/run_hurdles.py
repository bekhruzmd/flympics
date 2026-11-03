from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from controllers.base import Controller
from controllers.hurdle_baselines import (AlwaysController, HurdleRuleController, RandomJumpController)
from events.hurdles import JUMP, RUN, Hurdles

CONTROLLERS = ("always_run", "always_jump", "random", "rule1", "rule")
LAYOUTS = ("benchmark", "seeded")


def make_controller(name: str) -> Controller:
    return {
        "always_run": lambda: AlwaysController(RUN),
        "always_jump": lambda: AlwaysController(JUMP),
        "random": RandomJumpController,
        "rule1": lambda: HurdleRuleController(look_ahead=1),
        "rule": HurdleRuleController,
    }[name]()


def run(controller_name: str = "rule", seed: int = 7, layout: str = "benchmark", *,
        controller: Controller | None = None) -> list[dict]:
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; choose from {LAYOUTS}")
    event = Hurdles(seeded=layout == "seeded")
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
            "distance_to_next_hurdle": observation["distance_to_next_hurdle"],
            "distance_to_second_hurdle": observation["distance_to_second_hurdle"],
            "height": event.height,
            "airborne": event.airborne,
            "hits": event.hits,
            "jumps": event.jumps,
            "action": action,
        })
    return records


def score(records: list[dict], event: Hurdles | None = None) -> float:
    """Event score recomputed from the last record (same formula as Hurdles.score)."""
    event = event if event is not None else Hurdles()
    final = records[-1]
    if final["position"] >= event.distance:
        return final["simulation_time"]
    return event.max_time + (event.distance - final["position"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flympics 200 m Hurdles.")
    parser.add_argument("--controller", choices=CONTROLLERS, default="rule")
    parser.add_argument("--layout", choices=LAYOUTS, default="benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    records = run(args.controller, args.seed, args.layout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"hurdles_{args.layout}_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    final = records[-1]
    print(f"{args.controller} on {args.layout}: {final['position']:.1f} m in {final['simulation_time']:.2f} s, "
          f"score {score(records):.2f}; {final['hits']} hits, {final['jumps']} jumps. Logs: {stem}.json/.csv")


if __name__ == "__main__":
    main()
