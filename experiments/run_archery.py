"""Run one Flympics Archery end (6 arrows in a gusty crosswind)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import ARCHERY_POLICY, FlyBrainController
from controllers.archery_baselines import AlwaysController, ArcheryRuleController, RandomController
from controllers.base import Controller
from events.archery import HOLD, RELEASE, Archery

CONTROLLERS = ("rule", "always_release", "always_hold", "random", "fly", "trained")


def make_controller(name: str) -> Controller:
    return {
        "rule": ArcheryRuleController,
        "always_release": lambda: AlwaysController(RELEASE),
        "always_hold": lambda: AlwaysController(HOLD),
        "random": RandomController,
        "fly": FlyBrainController.archery,
        "trained": lambda: FlyBrainController.from_policy(ARCHERY_POLICY),
    }[name]()


def run(controller_name: str = "rule", seed: int = 7, *,
        controller: Controller | None = None, event: Archery | None = None) -> list[dict]:
    """One end. The seed sets the wind and sway, and seeds random controllers."""
    event = event if event is not None else Archery()
    if controller is None:
        controller = make_controller(controller_name)
    event.reset(seed)
    controller.reset(seed)
    records: list[dict] = []
    while not event.is_finished():
        before = event.observe()
        arrow = event.arrow
        action = controller.act(before)
        event.step(action)
        shot = len(event.points) > arrow
        records.append({
            "step": event.steps,
            "simulation_time": round(event.time, 8),
            "arrow": arrow + 1,
            "aim_offset": before["aim_offset"],
            "wind": before["wind"],
            "draw": before["draw"],
            "hold_time": before["hold_time"],
            "action": action,
            "shot": shot,
            "landing": event.landings[arrow] if shot else None,
            "points": event.points[arrow] if shot else None,
            "total_points": event.total_points,
        })
        if isinstance(controller, FlyBrainController):
            records[-1].update(controller.last_summary)
    return records


def score(records: list[dict]) -> float:
    """Total points (higher is better), the same as Archery.score."""
    return float(records[-1]["total_points"])


def summary(records: list[dict]) -> dict:
    shots = [r for r in records if r["shot"]]
    return {
        "points": score(records),
        "arrow_points": [r["points"] for r in shots],
        "timeouts": sum(r["landing"] is None for r in shots),
        "time": records[-1]["simulation_time"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, default="rule")
    parser.add_argument("--seed", type=int, default=7, help="Sets the wind and sway.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip the target and aim plot")
    args = parser.parse_args()
    records = run(args.controller, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"archery_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if not args.no_plot:
        from visualization.archery_view import save_archery_plot
        save_archery_plot(records, stem.with_suffix(".png"))
    info = summary(records)
    print(f"{args.controller}: {info['points']:.0f} / 60 points {info['arrow_points']} in "
          f"{info['time']:.1f} s, {info['timeouts']} timed out. Logs: {stem}.json/.csv")


if __name__ == "__main__":
    main()
