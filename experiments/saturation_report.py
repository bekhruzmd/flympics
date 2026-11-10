"""How saturated is the MANC circuit, and how does the neuron gain change that?

For one event and a list of gains, this report races the event's untrained fly at each
gain and measures:

- each neuron group's average activity over the race (sensory, intrinsic, extensor,
  flexor). Activity runs from 0 to 1; near 1 means the group is saturated;
- at each race step, the percentage of neurons in each group whose activity exceeds 0.95,
  averaged over the race;
- for each encoder input, how much "extensor minus flexor" changes when that input goes
  from its low to high test value, with the other inputs held at their midpoints and the
  network left to settle. Near 0 means the circuit no longer passes that input to the
  motor readout;
- the race score.

Gain is a modeling choice (RateNetwork's default is 3.0), not a measured property of
the fly. This report only sets it on its own controllers; the default is unchanged.

    python -m experiments.saturation_report --event sprint
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from brain.controller import SPRINT_ENCODER_INIT, FlyBrainController
from brain.archery_io import ARCHERY_INPUTS
from brain.sensory import ENCODER_INPUTS
from brain.steeplechase_io import STEEPLECHASE_INPUTS
from events.sprint import Sprint
from experiments.run_sprint import run as run_sprint
from brain.hurdles_io import HURDLES_INPUTS
from brain.pacing_io import PACING_INPUTS
from brain.swimming_io import SWIMMING_INPUTS
from experiments import run_archery, run_hurdles, run_pacing, run_steeplechase, run_swimming

DEFAULT_GAINS = (0.5, 1.0, 1.5, 2.0, 3.0)
GROUPS = (("sensory", "sensory_mean"), ("intrinsic", "intrinsic_mean"),
          ("extensor", "motor_extensor"), ("flexor", "motor_flexor"))
SATURATION_GROUPS = tuple((group, f"{group}_over_095_pct") for group, _ in GROUPS)
CIRCUIT_ROLES = {"sensory": "sensory", "intrinsic": "intrinsic",
                 "extensor": "motor_extensor", "flexor": "motor_flexor"}
SATURATED = 0.95
SETTLE_STEPS = 50  # steps of constant input before reading the motor difference
HELD_LEVEL = 0.5   # level of the inputs that are not being tested


@dataclass(frozen=True)
class EventSpec:
    """What the report needs from an event; add one per event."""
    name: str
    inputs: tuple[str, ...]                                # encoder feature names, in order
    make_fly: Callable[[], FlyBrainController]             # the untrained fly
    race: Callable[[FlyBrainController], list[dict]]       # one race; per-step records
    score: Callable[[list[dict]], float]
    score_note: str
    score_short: str                                       # a few words for the plot
    # Per input: low test value, high test value, and held midpoint.
    input_ranges: tuple[tuple[float, float, float], ...] | None = None


def _sprint_score(records: list[dict]) -> float:
    final = records[-1]
    if final["distance_to_finish"] == 0:
        return float(final["simulation_time"])
    return Sprint().max_time + float(final["distance_to_finish"])


EVENTS = {
    "sprint": EventSpec(
        name="sprint",
        inputs=ENCODER_INPUTS,
        # The init weights (0.90, 0.25) give the same stimulus as the plain `fly`, and this
        # encoder also exposes encode_features() for the low/high input test.
        make_fly=lambda: FlyBrainController(encoder_weights=SPRINT_ENCODER_INIT),
        race=lambda fly: run_sprint(controller=fly),
        score=_sprint_score,
        score_note="seconds to finish; unfinished = 60 + metres left (lower is better)",
        score_short="s; 60 + m left if unfinished; lower = better",
    ),
    "hurdles": EventSpec(
        name="hurdles",
        inputs=HURDLES_INPUTS,
        make_fly=FlyBrainController.hurdles,
        race=lambda fly: run_hurdles.run(controller=fly),   # benchmark layout, seed 7
        score=run_hurdles.score,
        score_note="seconds to finish 200 m; unfinished = 90 + metres left (lower is better)",
        score_short="s; 90 + m left if unfinished; lower = better",
    ),
    "swimming": EventSpec(
        name="swimming",
        inputs=SWIMMING_INPUTS,
        make_fly=FlyBrainController.swimming,
        race=lambda fly: run_swimming.run(controller=fly),
        score=run_swimming.score,
        score_note="seconds to finish 100 m; unfinished = 120 + metres left (lower is better)",
        score_short="s; 120 + m left if unfinished; lower = better",
    ),
    "pacing": EventSpec(
        name="pacing",
        inputs=PACING_INPUTS,
        make_fly=FlyBrainController.pacing,
        race=lambda fly: run_pacing.run(controller=fly),
        score=run_pacing.score,
        score_note="seconds to finish 400 m; unfinished = 240 + metres left (lower is better)",
        score_short="s; 240 + m left if unfinished; lower = better",
    ),
    "steeplechase": EventSpec(
        name="steeplechase",
        inputs=STEEPLECHASE_INPUTS,
        make_fly=FlyBrainController.steeplechase,
        race=lambda fly: run_steeplechase.run(controller=fly),
        score=run_steeplechase.score,
        score_note="seconds to finish 600 m; unfinished score includes max_time + metres left (lower is better)",
        score_short="s; unfinished is penalized; lower = better",
    ),
    "archery": EventSpec(
        name="archery",
        inputs=ARCHERY_INPUTS,
        make_fly=FlyBrainController.archery,
        race=lambda fly: run_archery.run(controller=fly, seed=7),
        score=run_archery.score,
        score_note="total points across six arrows, 0-60 (higher is better)",
        score_short="points / 60; higher = better",
        # Aim and wind are signed; draw is in [0, 1]. Hold each at its domain midpoint.
        input_ranges=((-1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 1.0, 0.5)),
    ),
}


def input_effect(fly: FlyBrainController, n_inputs: int, index: int,
                 steps: int = SETTLE_STEPS, held: float | Sequence[float] = HELD_LEVEL,
                 low: float = 0.0, high: float = 1.0) -> float:
    """Settled motor difference at the input's high value minus its low value."""
    differences = []
    held_values = np.full(n_inputs, held) if np.isscalar(held) else np.asarray(held, dtype=float)
    if held_values.shape != (n_inputs,):
        raise ValueError(f"Expected {n_inputs} held input values, got {held_values.shape}.")
    for level in (low, high):
        features = held_values.copy()
        features[index] = level
        stimulus = fly.encoder.encode_features(features)
        fly.network.reset()
        for _ in range(steps):
            activity = fly.network.step(stimulus)
        differences.append(fly.decoder.decode(activity)[1]["difference"])
    fly.network.reset()
    return differences[1] - differences[0]


def measure(spec: EventSpec, gain: float) -> dict:
    """One row of the report: group activity, input effects and score at this gain."""
    fly = spec.make_fly()
    fly.network.gain = gain
    # Count saturated neurons on each race step by watching the network from outside, so
    # the controller itself stays unchanged.
    indices = {group: fly.circuit.indices(role) for group, role in CIRCUIT_ROLES.items()}
    saturated = {key: [] for _, key in SATURATION_GROUPS}
    plain_step = fly.network.step

    def counting_step(stimulus, *args, **kwargs):
        activity = plain_step(stimulus, *args, **kwargs)
        for group, key in SATURATION_GROUPS:
            saturated[key].append(float(np.mean(activity[indices[group]] > SATURATED) * 100.0))
        return activity

    fly.network.step = counting_step
    try:
        records = spec.race(fly)
    finally:
        del fly.network.step  # back to the class's own step
    row = {"event": spec.name, "gain": gain}
    for group, key in GROUPS:
        row[f"{group}_mean"] = float(np.mean([record[key] for record in records]))
    for _, key in SATURATION_GROUPS:
        row[key] = float(np.mean(saturated[key]))
    row["difference_mean"] = float(np.mean([record["motor_difference"] for record in records]))
    input_ranges = spec.input_ranges or ((0.0, 1.0, HELD_LEVEL),) * len(spec.inputs)
    if len(input_ranges) != len(spec.inputs):
        raise ValueError(f"{spec.name}: input_ranges must match its {len(spec.inputs)} inputs.")
    held_values = tuple(test_range[2] for test_range in input_ranges)
    for index, (name, (low, high, _)) in enumerate(zip(spec.inputs, input_ranges)):
        row[f"effect_{name}"] = input_effect(
            fly, len(spec.inputs), index, held=held_values, low=low, high=high)
    row["score"] = spec.score(records)
    row["steps"] = len(records)
    return row


def report(spec: EventSpec, gains: Sequence[float] = DEFAULT_GAINS) -> list[dict]:
    return [measure(spec, gain) for gain in gains]


def markdown_table(spec: EventSpec, rows: list[dict]) -> str:
    effect_cols = [f"effect_{name}" for name in spec.inputs]
    header = (["gain"] + [group for group, _ in GROUPS]
              + [f"{group} >.95 %" for group, _ in SATURATION_GROUPS]
              + [f"effect of {name}" for name in spec.inputs] + ["score"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for row in rows:
        cells = ([f"{row['gain']:g}"] + [f"{row[f'{group}_mean']:.3f}" for group, _ in GROUPS]
                 + [f"{row[key]:.1f}" for _, key in SATURATION_GROUPS]
                 + [f"{row[col]:.2e}" for col in effect_cols] + [f"{row['score']:.1f}"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(spec: EventSpec, rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [row["gain"] for row in rows]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for group, _ in GROUPS:
        axes[0].plot(gains, [row[f"{group}_mean"] for row in rows], marker="o", label=group)
    axes[0].set(title="Average activity over the race", xlabel="gain", ylabel="activity (0-1)", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    for group, key in SATURATION_GROUPS:
        axes[1].plot(gains, [row[key] for row in rows], marker="o", label=group)
    axes[1].set(title="Neurons above 0.95 activity", xlabel="gain",
                ylabel="neurons in group (%)", ylim=(0, 100))
    axes[1].legend(fontsize=8)
    for name in spec.inputs:
        axes[2].plot(gains, [abs(row[f"effect_{name}"]) for row in rows], marker="o", label=name)
    axes[2].set(title="|change in extensor - flexor|\nwhen one input goes low -> high", xlabel="gain",
                ylabel="|Δ difference| (log)", yscale="log")
    axes[2].legend(fontsize=8)
    axes[3].plot(gains, [row["score"] for row in rows], marker="o", color="black")
    axes[3].set(title=f"Untrained fly's {spec.name} score\n({spec.score_short})", xlabel="gain", ylabel="score")
    for axis in axes:
        axis.axvline(3.0, color="grey", linestyle=":", linewidth=1)  # the project default
        axis.grid(alpha=0.3)
    fig.suptitle(f"{spec.name.capitalize()}: circuit saturation vs neuron gain (dotted line: default gain 3)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", choices=sorted(EVENTS), default="sprint")
    parser.add_argument("--gains", type=float, nargs="+", default=list(DEFAULT_GAINS))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "saturation")
    parser.add_argument("--plot", type=Path, default=None, help="Plot path (default <output-dir>/<event>.png)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    spec = EVENTS[args.event]
    rows = report(spec, args.gains)
    save_csv(rows, args.output_dir / f"{spec.name}.csv")
    print(markdown_table(spec, rows))
    print(f"\nScore: {spec.score_note}. CSV: {args.output_dir / f'{spec.name}.csv'}")
    if not args.no_plot:
        plot = args.plot or args.output_dir / f"{spec.name}.png"
        save_plot(spec, rows, plot)
        print(f"Plot: {plot}")


if __name__ == "__main__":
    main()
