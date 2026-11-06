"""Static plot for logged Steeplechase runs; drawing never changes physics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from events.steeplechase import BARRIERS, WATER

DISTANCE = 600.0


def save_steeplechase_plot(records: Sequence[dict], output: str | Path) -> None:
    """Distance, speed and stamina over simulation time; jumps shaded, hits and splashes marked."""
    if not records:
        raise ValueError("The plot needs at least one timestep.")
    times = [r["simulation_time"] for r in records]
    fig, (track, speed, stamina) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    track.plot(times, [r["position"] for r in records], color="tab:green", label="athlete position")
    for i, barrier in enumerate(BARRIERS):
        water = i in WATER
        track.axhline(barrier, color="tab:blue" if water else "tab:gray", linewidth=0.8 if water else 0.5,
                      alpha=0.6, label=("water jump" if i == min(WATER) else "barrier" if i == 0 else None))
    track.axhline(DISTANCE, color="tab:orange", linestyle="--", label="finish")
    track.set(ylabel="metres", title="Flympics — 600 m Steeplechase")
    speed.plot(times, [r["velocity"] for r in records], color="tab:blue", linewidth=1)
    start = None
    spans = []
    for previous, row in zip([None, *records], records):
        if row["airborne"] and start is None:
            start = previous["simulation_time"] if previous else 0.0
        elif not row["airborne"] and start is not None:
            spans.append((start, row["simulation_time"])); start = None
    for i, (a, b) in enumerate(spans):
        speed.axvspan(a, b, color="tab:cyan", alpha=0.3, label="jump" if i == 0 else None)
    for key, marker, color, label in (("hits", "x", "tab:red", "hit"), ("splashes", "v", "tab:blue", "splash")):
        events = [i for i, (p, r) in enumerate(zip([None, *records], records))
                  if r[key] > (p[key] if p else 0)]
        if events:
            speed.plot([times[i] for i in events], [records[i]["velocity"] for i in events], marker,
                       color=color, ms=8, mew=2, linestyle="none", label=label)
    speed.set(ylabel="m/s")
    stamina.plot(times, [r["stamina"] for r in records], color="tab:purple")
    stamina.set(xlabel="simulation time (s)", ylabel="stamina", ylim=(0, 1.05))
    track.legend(loc="lower right", fontsize=8)
    if speed.get_legend_handles_labels()[0]:
        speed.legend(loc="lower right", fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)
