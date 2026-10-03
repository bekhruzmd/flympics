from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def save_sprint_plot(records: Sequence[dict], output: str | Path) -> None:
    """Save one simple static chart; no real-time GUI or frontend is required."""
    times = [r["simulation_time"] for r in records]
    positions = [r["position"] for r in records]
    velocities = [r["velocity"] for r in records]
    fig, (track, speed) = plt.subplots(2, 1, figsize=(8, 5), layout="constrained")
    track.plot(times, positions, color="tab:green", label="athlete position")
    track.axhline(100, color="tab:orange", linestyle="--", label="finish")
    track.set(ylabel="metres", title="Flympics — 100 m Sprint")
    track.legend()
    speed.plot(times, velocities, color="tab:blue")
    speed.set(xlabel="simulation time (s)", ylabel="m/s")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def create_activity_view(records: Sequence[dict], controller_name: str):
    """Build a simple viewer from measured logs; drawing never changes physics.

    Activity is dimensionless modeled rate (0–1), not firing Hz. The little fly
    and circuit diagram are illustrations, not reconstructed neural anatomy.
    """
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Ellipse

    if not records:
        raise ValueError("The activity viewer needs at least one timestep.")
    fig = plt.figure(figsize=(11, 6), facecolor="#f5f5ef", layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1])
    track = fig.add_subplot(grid[0, :])
    history = fig.add_subplot(grid[1, 0])
    card = fig.add_subplot(grid[1, 1])
    fig.suptitle(f"FLYMPICS / 100 m Sprint / {controller_name} controller", fontsize=16)
    track.set(xlim=(-3, 104), ylim=(-0.6, 1), xlabel="Distance (m)", yticks=[])
    track.spines[["left", "right", "top"]].set_visible(False)
    track.axhline(0, color="#b4bcac", linewidth=2)
    track.axvline(0, color="#b4bcac", linestyle=":")
    track.axvline(100, color="#b4bcac", linestyle=":")
    track.text(0, 0.72, "START", fontsize=9)
    track.text(100, 0.72, "FINISH", ha="right", fontsize=9)
    status = track.text(0.02, 0.02, "", transform=track.transAxes, fontsize=11)
    fly = DrawingArea(70, 48, 0, 0)
    # Six static legs and translucent wings: diagrammatic athlete, not gait data.
    for x in (23, 33, 43):
        for direction in (-1, 1):
            fly.add_artist(plt.Line2D([x, x - 7], [24, 24 + direction * 17],
                                      color="#343d35", linewidth=1.5))
    for y, angle in ((33, 22), (15, -22)):
        fly.add_artist(Ellipse((29, y), 35, 15, angle=angle,
                               facecolor="#d5e8e4", edgecolor="#839c96", alpha=0.85))
    fly.add_artist(Ellipse((31, 24), 29, 14, facecolor="#655448"))
    fly.add_artist(Ellipse((47, 24), 17, 16, facecolor="#363d34"))
    fly.add_artist(Ellipse((52, 28), 6, 6, facecolor="#c95a47"))
    athlete = AnnotationBbox(fly, (0, 0.18), frameon=False)
    track.add_artist(athlete)

    times = np.array([r["simulation_time"] for r in records])
    velocities = np.array([r["velocity"] for r in records])
    history.set(xlim=(0, max(times[-1], 1)), ylim=(0, max(0.6, velocities.max() * 1.2)),
                xlabel="Simulation time (s)", ylabel="Speed (m/s)", title="Athlete speed")
    history.spines[["right", "top"]].set_visible(False)
    speed_line, = history.plot([], [], color="#35765a", linewidth=2)

    neural = controller_name in {"fly", "trained"}
    keys = ("sensory_mean", "intrinsic_mean", "motor_extensor", "motor_flexor")
    labels = ("Sensory input", "VNC interneurons", "Extensor output", "Flexor output")
    colors = ("#4089aa", "#8b71b3", "#3d946d", "#d19543")
    card.set(xlim=(0, 1.23), ylim=(-0.7, 3.7), yticks=range(4),
             yticklabels=labels, xticks=[0, 0.5, 1], xlabel="Mean modeled activity (0–1)",
             title="Circuit activity · MANC nerve cord" if neural else "Circuit activity · unavailable")
    card.invert_yaxis()
    card.spines[["right", "top"]].set_visible(False)
    card.barh(range(4), [1] * 4, color="#e7e9e2", height=0.5)
    bars = card.barh(range(4), [0] * 4, color=colors, height=0.5)
    values = [card.text(1.04, i, "", va="center", fontsize=10) for i in range(4)]
    note = fig.supxlabel("Selected VNC populations, not a whole-brain map. Rates and action mapping are modeled.",
                        fontsize=9, color="#60665d")
    if not neural:
        note.set_text("Baseline controller: no neural model. Activity bars are not applicable.")

    def update(index):
        row = records[index]
        athlete.xy = (row["position"], 0.18)
        athlete.xybox = athlete.xy
        speed_line.set_data(times[:index + 1], velocities[:index + 1])
        ending = ""
        if index == len(records) - 1:
            ending = "  |  FINISHED" if row["distance_to_finish"] <= 0 else "  |  TIME LIMIT"
        status.set_text(f"{row['simulation_time']:5.1f} s   |   {row['position']:.2f} / 100 m   |   "
                        f"{row['velocity']:.2f} m/s   |   {row['action']}{ending}")
        for bar, value, key in zip(bars, values, keys):
            activity = row.get(key)
            bar.set_width(float(activity) if neural and activity is not None else 0)
            value.set_text(f"{activity:.3f}" if neural and activity is not None else "N/A")
        return [athlete, speed_line, status, *bars, *values]

    update(0)
    return fig, update


def animate_sprint(records: Sequence[dict], controller_name: str, *, show: bool = True,
                   output: str | Path | None = None) -> None:
    """Display a logged run or export a compact 5x-speed GIF. Physics stays fixed."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, update = create_activity_view(records, controller_name)
    if output is not None:
        frames = sorted(set([0, *range(4, len(records), 5), len(records) - 1]))
        export = FuncAnimation(fig, update, frames=frames, interval=100, repeat=False)
        export.save(str(output), writer=PillowWriter(fps=10), dpi=85)
        fig.savefig(Path(output).with_suffix(".activity.png"), dpi=130)
    if show:
        playback = FuncAnimation(fig, update, frames=len(records), interval=100, repeat=False)
        # Keep animation alive until the window closes. Space pauses/resumes it.
        paused = False

        def on_key(event):
            nonlocal paused
            if event.key == " ":
                paused = not paused
                playback.pause() if paused else playback.resume()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.suptitle(f"FLYMPICS / {controller_name} / Space to pause or resume", fontsize=16)
        plt.show()
    plt.close(fig)
