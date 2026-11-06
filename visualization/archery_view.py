"""Static plot for logged Archery runs; drawing never changes the result."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from events.archery import Archery

RING_COLORS = ("#f2f2f2", "#f2f2f2", "#333333", "#333333", "#3b8fd6", "#3b8fd6",
               "#e04a3f", "#e04a3f", "#f5d142", "#f5d142")


def save_archery_plot(records: Sequence[dict], output: str | Path, event: Archery | None = None) -> None:
    """Left: the target face with every landing. Right: aim and ideal aim (-drift * wind) over time."""
    if not records:
        raise ValueError("The plot needs at least one timestep.")
    event = event if event is not None else Archery()
    fig, (face, aim) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained",
                                    gridspec_kw={"width_ratios": (1, 1.6)})
    radius = event.ring_width * event.rings
    for ring in range(event.rings):
        face.add_patch(Circle((0, 0), radius - ring * event.ring_width, color=RING_COLORS[ring],
                              ec="#888888", lw=0.5))
    shots = [r for r in records if r["shot"]]
    for r in shots:
        if r["landing"] is None:
            continue
        # Only horizontal aim is modelled; arrows are spread vertically just to be readable.
        y = (r["arrow"] - (len(shots) + 1) / 2) * 0.03
        face.plot(r["landing"], y, "o", color="#2ca02c", mec="black", ms=7)
        face.annotate(str(r["arrow"]), (r["landing"], y), xytext=(5, 3), textcoords="offset points", fontsize=8)
    total = records[-1]["total_points"]
    face.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), aspect="equal",
             title=f"{total} / {event.rings * event.arrows} points", xlabel="horizontal landing (m)")
    face.set_yticks([])
    times = [r["simulation_time"] for r in records]
    aim.plot(times, [r["aim_offset"] for r in records], color="tab:blue", label="aim")
    aim.plot(times, [-event.drift_per_wind * r["wind"] for r in records], color="tab:orange",
             linestyle="--", label="aim that cancels the wind")
    for i, r in enumerate(shots):
        aim.axvline(r["simulation_time"], color="tab:gray", linewidth=0.6,
                    label="release" if i == 0 else None)
    aim.set(xlabel="simulation time (s)", ylabel="m at the target", title="Flympics — Archery")
    aim.legend(loc="upper right", fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)
