from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import FlyBrainController
from controllers.base import Controller
from controllers.swim_baselines import (AlwaysController, ChoppyController, RandomController,
                                        SwimRuleController)
from events.swimming import PULL, RECOVER, Swimming

CONTROLLERS = ("always_pull", "always_recover", "choppy", "random", "full_strokes", "rule", "fly")


def make_controller(name: str) -> Controller:
    return {
        "always_pull": lambda: AlwaysController(PULL),
        "always_recover": lambda: AlwaysController(RECOVER),
        "choppy": ChoppyController,
        "random": RandomController,
        "full_strokes": lambda: SwimRuleController(v_break=None),
        "rule": SwimRuleController,
        "fly": FlyBrainController.swimming,
    }[name]()


def run(controller_name: str = "rule", seed: int = 7, *,
        controller: Controller | None = None, event: Swimming | None = None) -> list[dict]:
    """One fixed 100 m race. The seed only seeds the controller (e.g. random)."""
    event = event if event is not None else Swimming()
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
            "lap": event.lap,
            "velocity": event.velocity,
            "distance_to_wall": observation["distance_to_wall"],
            "stroke_phase": observation["stroke_phase"],
            "grip": event.grip,
            "gliding": event.gliding,
            "strokes": event.strokes,
            "turns": event.turns,
            "action": action,
        })
        if isinstance(controller, FlyBrainController):
            records[-1].update(controller.last_summary)
    return records


def score(records: list[dict], event: Swimming | None = None) -> float:
    """Event score recomputed from the last record (same formula as Swimming.score)."""
    event = event if event is not None else Swimming()
    final = records[-1]
    if final["position"] >= event.distance:
        return final["simulation_time"]
    return event.max_time + (event.distance - final["position"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flympics 100 m Swimming.")
    parser.add_argument("--controller", choices=CONTROLLERS, default="rule")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seeds random controllers only; the race itself is fixed.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip the distance and speed plot")
    parser.add_argument("--view", action="store_true", help="Open the animated pool and circuit activity viewer")
    parser.add_argument("--gif", action="store_true", help="Save a compact animated GIF (5x playback speed)")
    args = parser.parse_args()
    records = run(args.controller, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"swimming_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if not args.no_plot:
        from visualization.swimming_view import save_swimming_plot
        save_swimming_plot(records, stem.with_suffix(".png"))
    final = records[-1]
    glides = sum(row["gliding"] for row in records)
    print(f"{args.controller}: {final['position']:.1f} m in {final['simulation_time']:.2f} s, "
          f"score {score(records):.2f}; {final['strokes']} strokes, {final['turns']} turn(s), "
          f"{glides} glide steps. Logs: {stem}.json/.csv")
    if args.view or args.gif:
        from visualization.swimming_view import animate_swimming
        animate_swimming(records, args.controller, show=args.view,
                         output=stem.with_suffix(".gif") if args.gif else None)


if __name__ == "__main__":
    main()
