from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brain.controller import FlyBrainController
from controllers.random_controller import RandomController
from controllers.base import Controller
from controllers.rule_controller import RuleController
from events.sprint import Sprint


def make_controller(name: str):
    if name == "trained":
        return FlyBrainController.trained()
    return {"random": RandomController, "rule": RuleController, "fly": FlyBrainController}[name]()


def run(controller_name: str = "fly", seed: int = 7, *, controller: Controller | None = None,
        record_activity: bool = False) -> list[dict]:
    event = Sprint()
    if controller is None:
        controller = make_controller(controller_name)
    event.reset(seed)
    controller.reset(seed)
    records: list[dict] = []
    while not event.is_finished():
        observation = event.observe()
        action = controller.act(observation)
        event.step(action)
        record = {
            "step": event.steps,
            "simulation_time": round(event.time, 8),
            "position": event.position,
            "velocity": event.velocity,
            "distance_to_finish": event.observe()["distance_to_finish"],
            "action": action,
        }
        if isinstance(controller, FlyBrainController):
            record.update(controller.last_summary)
            if record_activity:
                record["neuron_activity"] = controller.network.state.tolist()
                record["motor_threshold"] = controller.decoder.threshold
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flympics 100 m Sprint MVP.")
    parser.add_argument("--controller", choices=("random", "rule", "fly", "trained"), default="fly")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--view", action="store_true", help="Open animated sprint and circuit activity viewer")
    parser.add_argument("--gif", action="store_true", help="Save a compact animated GIF (5x playback speed)")
    args = parser.parse_args()
    records = run(args.controller, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"sprint_{args.controller}_seed{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader(); writer.writerows(records)
    if not args.no_plot:
        from visualization.sprint_view import save_sprint_plot
        save_sprint_plot(records, stem.with_suffix(".png"))
    final = records[-1]
    print(f"{args.controller}: {final['position']:.1f} m in {final['simulation_time']:.1f} s; "
          f"last action={final['action']}. Logs: {stem}.json/.csv")
    if args.view or args.gif:
        from visualization.sprint_view import animate_sprint
        animate_sprint(records, args.controller, show=args.view,
                       output=stem.with_suffix(".gif") if args.gif else None)


if __name__ == "__main__":
    main()
