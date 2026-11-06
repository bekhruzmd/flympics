"""Run a 400 m pacing race with limited energy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import PACING_POLICY, FlyBrainController
from controllers.base import Controller
from controllers.pacing_baselines import (AlwaysAccelerateController, AlwaysCoastController,
                                          PacingRuleController, RandomPacingController,
                                          SprintPacingController)
from events.pacing import Pacing400m

CONTROLLERS = ("rule", "always_acc", "always_coast", "random", "sprint_rule", "fly", "trained", "trained_es", "trained_threshold")


def make_controller(name: str) -> Controller:
    es_policy = PACING_POLICY.with_name("pacing_es.json")
    return {
        "rule": PacingRuleController,
        "always_acc": AlwaysAccelerateController,
        "always_coast": AlwaysCoastController,
        "random": RandomPacingController,
        "sprint_rule": SprintPacingController,
        "fly": FlyBrainController.pacing,
        "trained": lambda: FlyBrainController.from_policy(es_policy if es_policy.exists() else PACING_POLICY),
        "trained_es": lambda: FlyBrainController.from_policy(es_policy),
        "trained_threshold": lambda: FlyBrainController.from_policy(PACING_POLICY),
    }[name]()


def run(controller_name: str = "rule", seed: int = 7, *,
        controller: Controller | None = None, event: Pacing400m | None = None,
        record_activity: bool = False) -> list[dict]:
    """One deterministic 400 m pacing race."""
    event = event if event is not None else Pacing400m()
    if controller is None:
        controller = make_controller(controller_name)
    event.reset(seed)
    controller.reset(seed)
    records: list[dict] = []
    while not event.is_finished():
        observation = event.observe()
        action = controller.act(observation)
        next_obs = event.step(action)
        record = {
            "step": event.steps,
            "simulation_time": round(event.time, 8),
            "position": event.position,
            "velocity": event.velocity,
            "energy": event.energy,
            "distance_to_finish": next_obs["distance_to_finish"],
            "action": action,
            "exhausted": event.energy <= 0.0,
        }
        if isinstance(controller, FlyBrainController):
            record.update(controller.last_summary)
            if record_activity:
                record["neuron_activity"] = controller.network.state.tolist()
                th = getattr(controller.decoder, "thresholds", None)
                record["motor_threshold"] = th[0] if th else getattr(controller.decoder, "threshold", 0.015)
        records.append(record)
    return records


def score(records: list[dict], event: Pacing400m | None = None) -> float:
    """Event score recomputed from the last record (same formula as Pacing400m.score)."""
    event = event if event is not None else Pacing400m()
    final = records[-1]
    if final["position"] >= event.distance:
        return final["simulation_time"]
    return event.max_time + (event.distance - final["position"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flympics 400 m Pacing event.")
    parser.add_argument("--controller", choices=CONTROLLERS, default="rule")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seeds random controllers; the race physics are deterministic.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the plot")
    parser.add_argument("--view", action="store_true", help="Open animated viewer")
    parser.add_argument("--gif", action="store_true", help="Save animation as GIF")
    args = parser.parse_args()

    records = run(args.controller, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"pacing_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    if not args.no_plot:
        from visualization.pacing_view import save_pacing_plot
        save_pacing_plot(records, stem.with_suffix(".png"))

    final = records[-1]
    exhausted_count = sum(row["exhausted"] for row in records)
    print(f"{args.controller}: {final['position']:.1f} m in {final['simulation_time']:.2f} s, "
          f"score {score(records):.2f}; final energy={final['energy']:.1f}, "
          f"{exhausted_count} exhausted steps. Logs: {stem}.json/.csv")

    if args.view or args.gif:
        from visualization.pacing_view import animate_pacing
        animate_pacing(records, args.controller, show=args.view,
                       output=stem.with_suffix(".gif") if args.gif else None)


if __name__ == "__main__":
    main()
