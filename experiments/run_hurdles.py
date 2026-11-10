from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import HURDLES_POLICY, FlyBrainController
from controllers.base import Controller
from controllers.hurdle_baselines import AlwaysController, HurdleRuleController, RandomController
from events.hurdles import ACCELERATE, BENCHMARK_LAYOUT, COAST, JUMP, Hurdles, generate_layout

CONTROLLERS = ("rule", "nocoast", "always_accelerate", "always_coast", "always_jump", "random", "fly", "trained")
LAYOUTS = ("benchmark", "seeded")


def make_controller(name: str, policy: Path | None = None) -> Controller:
    if name == "trained" and policy is not None:
        return FlyBrainController.from_policy(policy)
    return {
        "rule": HurdleRuleController,
        "nocoast": lambda: HurdleRuleController(coast=False),
        "always_accelerate": lambda: AlwaysController(ACCELERATE),
        "always_coast": lambda: AlwaysController(COAST),
        "always_jump": lambda: AlwaysController(JUMP),
        "random": RandomController,
        "fly": FlyBrainController.hurdles,
        "trained": lambda: FlyBrainController.from_policy(HURDLES_POLICY),
    }[name]()


def course(layout: str = "benchmark", seed: int = 7) -> tuple[float, ...]:
    """Hurdle positions that run() races on, for plots and viewers."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; choose from {LAYOUTS}")
    return generate_layout(seed) if layout == "seeded" else BENCHMARK_LAYOUT


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
            "coasts": event.coasts,
            "takeoff_keep": event.last_takeoff_keep,
            "action": action,
        })
        if isinstance(controller, FlyBrainController):
            records[-1].update(controller.last_summary)
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
    parser.add_argument("--policy", type=Path, default=None, help="Policy JSON file for the trained controller")
    parser.add_argument("--layout", choices=LAYOUTS, default="benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip the distance and speed plot")
    parser.add_argument("--view", action="store_true", help="Open the animated track and circuit activity viewer")
    parser.add_argument("--gif", action="store_true", help="Save an animated GIF (2.5x playback speed)")
    args = parser.parse_args()
    controller = make_controller(args.controller, policy=args.policy) if args.policy is not None else None
    records = run(args.controller, args.seed, args.layout, controller=controller)
    hurdles = course(args.layout, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"hurdles_{args.layout}_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if not args.no_plot:
        from visualization.hurdles_view import save_hurdles_plot
        save_hurdles_plot(records, hurdles, stem.with_suffix(".png"))
    final = records[-1]
    print(f"{args.controller} on {args.layout}: {final['position']:.1f} m in {final['simulation_time']:.2f} s, "
          f"score {score(records):.2f}; {final['hits']} hits, {final['jumps']} jumps, "
          f"{final['coasts']} coast steps. Logs: {stem}.json/.csv")
    if args.view or args.gif:
        from visualization.hurdles_view import animate_hurdles
        animate_hurdles(records, hurdles, args.controller, show=args.view,
                        output=stem.with_suffix(".gif") if args.gif else None)


if __name__ == "__main__":
    main()
