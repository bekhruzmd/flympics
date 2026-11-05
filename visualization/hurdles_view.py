"""Static plot and animated viewer for logged Hurdles runs; drawing never changes physics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

DISTANCE = 200.0
HURDLE_HEIGHT = 0.5  # in jump heights, as in Hurdles
SAFE_SPEED = 4.5  # fastest take-off with no extra cost, as in Hurdles
NEURAL_CONTROLLERS = {"fly"}
STATUS_NAMES = {"ACCELERATE": "ACCEL"}  # keeps the status line from shifting


def _spans(records: Sequence[dict], flags: Sequence[bool]) -> list[tuple[float, float]]:
    """(start, end) simulation times of each run of steps whose flag is set."""
    spans, start = [], None
    for previous, row, flag in zip([None, *records], records, flags):
        begin = previous["simulation_time"] if previous else 0.0
        if flag and start is None:
            start = begin
        elif not flag and start is not None:
            spans.append((start, begin)); start = None
    if start is not None:
        spans.append((start, records[-1]["simulation_time"]))
    return spans


def air_spans(records: Sequence[dict]) -> list[tuple[float, float]]:
    """(start, end) simulation times of each jump."""
    return _spans(records, [row["airborne"] for row in records])


def coast_spans(records: Sequence[dict]) -> list[tuple[float, float]]:
    """(start, end) simulation times of each stretch of coasting on the ground."""
    coasted = [row["coasts"] > previous for previous, row in zip([0, *(r["coasts"] for r in records)], records)]
    return _spans(records, coasted)


def hurdle_results(records: Sequence[dict], hurdles: Sequence[float]) -> list[tuple[int, bool]]:
    """For each hurdle passed: (record index where it was passed, whether it was hit)."""
    results, previous_hits, upcoming = [], 0, list(hurdles)
    for index, row in enumerate(records):
        passed = 0
        while upcoming and upcoming[0] <= row["position"]:
            upcoming.pop(0); passed += 1
        new_hits = row["hits"] - previous_hits
        # Two hurdles passed in one step is rare; any new hits go to the later ones.
        for k in range(passed):
            results.append((index, k >= passed - new_hits))
        previous_hits = row["hits"]
    return results


def save_hurdles_plot(records: Sequence[dict], hurdles: Sequence[float], output: str | Path) -> None:
    """Distance and speed over simulation time, with jumps and coasting shaded and hits marked."""
    if not records:
        raise ValueError("The plot needs at least one timestep.")
    times = [r["simulation_time"] for r in records]
    fig, (track, speed) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    track.plot(times, [r["position"] for r in records], color="tab:green", label="athlete position")
    for i, hurdle in enumerate(hurdles):
        track.axhline(hurdle, color="tab:gray", linewidth=0.6, alpha=0.5, label="hurdles" if i == 0 else None)
    track.axhline(DISTANCE, color="tab:orange", linestyle="--", label="finish")
    track.set(ylabel="metres", title="Flympics — 200 m Hurdles")
    speed.plot(times, [r["velocity"] for r in records], color="tab:blue", linewidth=1)
    for i, (start, end) in enumerate(air_spans(records)):
        speed.axvspan(start, end, color="tab:cyan", alpha=0.25, label="jump" if i == 0 else None)
    for i, (start, end) in enumerate(coast_spans(records)):
        speed.axvspan(start, end, color="tab:olive", alpha=0.2, label="coast" if i == 0 else None)
    speed.axhline(SAFE_SPEED, color="tab:gray", linestyle="--", linewidth=1, label="safe take-off speed")
    hits = [index for index, hit in hurdle_results(records, hurdles) if hit]
    if hits:
        track.plot([times[i] for i in hits], [records[i]["position"] for i in hits], "x",
                   color="tab:red", markersize=8, mew=2, label="hit")
        speed.plot([times[i] for i in hits], [records[i]["velocity"] for i in hits], "x",
                   color="tab:red", markersize=8, mew=2, label="hit")
    speed.set(xlabel="simulation time (s)", ylabel="m/s")
    track.legend(loc="lower right")
    if speed.get_legend_handles_labels()[0]:
        speed.legend(loc="lower right", fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _fly_drawing():
    """Diagrammatic fly, the same as Sprint's: an illustration, not gait data."""
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Ellipse

    fly = DrawingArea(70, 48, 0, 0)
    for x in (23, 33, 43):
        for direction in (-1, 1):
            fly.add_artist(plt.Line2D([x, x - 7], [24, 24 + direction * 17], color="#343d35", linewidth=1.5))
    for y, angle in ((33, 22), (15, -22)):
        fly.add_artist(Ellipse((29, y), 35, 15, angle=angle, facecolor="#d5e8e4", edgecolor="#839c96", alpha=0.85))
    fly.add_artist(Ellipse((31, 24), 29, 14, facecolor="#655448"))
    fly.add_artist(Ellipse((47, 24), 17, 16, facecolor="#363d34"))
    fly.add_artist(Ellipse((52, 28), 6, 6, facecolor="#c95a47"))
    return fly


def create_hurdles_view(records: Sequence[dict], hurdles: Sequence[float], controller_name: str):
    """Build the track view from measured logs.

    Hurdles turn red when hit and green when cleared. Jump height is drawn to scale with
    the hurdles; the fly drawing itself is not. Activity is dimensionless modeled rate
    (0-1), not firing Hz.
    """
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox

    if not records:
        raise ValueError("The hurdles viewer needs at least one timestep.")
    fig = plt.figure(figsize=(11, 6), facecolor="#f5f5ef", layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1])
    track = fig.add_subplot(grid[0, :])
    history = fig.add_subplot(grid[1, 0])
    card = fig.add_subplot(grid[1, 1])
    fig.suptitle(f"FLYMPICS / 200 m Hurdles / {controller_name} controller", fontsize=16)

    track.set(xlim=(-4, DISTANCE + 4), ylim=(-0.5, 1.7), xlabel="Distance (m)", yticks=[])
    track.spines[["left", "right", "top"]].set_visible(False)
    track.axhline(0, color="#b4bcac", linewidth=2)
    for line in (0, DISTANCE):
        track.axvline(line, color="#b4bcac", linestyle=":")
    track.text(0, 1.45, "START", fontsize=9)
    track.text(DISTANCE, 1.45, "FINISH", ha="right", fontsize=9)
    bars = track.vlines(hurdles, 0, HURDLE_HEIGHT, colors="#6b6f66", linewidth=3)
    status = track.text(0.02, 0.02, "", transform=track.transAxes, fontsize=11)
    athlete = AnnotationBbox(_fly_drawing(), (0, 0.15), frameon=False)
    track.add_artist(athlete)

    outcome_at = dict(enumerate(hurdle_results(records, hurdles)))  # hurdle number -> (record index, hit)

    times = np.array([r["simulation_time"] for r in records])
    velocities = np.array([r["velocity"] for r in records])
    history.set(xlim=(0, max(times[-1], 1)), ylim=(0, max(0.6, velocities.max() * 1.2)),
                xlabel="Simulation time (s)", ylabel="Speed (m/s)", title="Athlete speed")
    history.spines[["right", "top"]].set_visible(False)
    speed_line, = history.plot([], [], color="#35765a", linewidth=1.5)

    neural = controller_name in NEURAL_CONTROLLERS
    keys = ("sensory_mean", "intrinsic_mean", "motor_extensor", "motor_flexor")
    labels = ("Sensory input", "VNC interneurons", "Extensor output", "Flexor output")
    colors = ("#4089aa", "#8b71b3", "#3d946d", "#d19543")
    card.set(xlim=(0, 1.23), ylim=(-0.7, 3.7), yticks=range(4), yticklabels=labels,
             xticks=[0, 0.5, 1], xlabel="Mean modeled activity (0-1)",
             title="Circuit activity · MANC nerve cord" if neural else "Circuit activity · unavailable")
    card.invert_yaxis()
    card.spines[["right", "top"]].set_visible(False)
    card.barh(range(4), [1] * 4, color="#e7e9e2", height=0.5)
    activity_bars = card.barh(range(4), [0] * 4, color=colors, height=0.5)
    values = [card.text(1.04, i, "", va="center", fontsize=10) for i in range(4)]
    note = fig.supxlabel("Selected VNC populations, not a whole-brain map. Rates and action mapping are modeled.",
                         fontsize=9, color="#60665d")
    if not neural:
        note.set_text("Baseline controller: no neural model. Activity bars are not applicable.")

    def update(index):
        row = records[index]
        athlete.xy = athlete.xybox = (row["position"], 0.15 + row["height"])
        bars.set_color([
            ("#c0392b" if outcome_at[n][1] else "#3d946d") if n in outcome_at and outcome_at[n][0] <= index
            else "#6b6f66"
            for n in range(len(hurdles))])
        speed_line.set_data(times[:index + 1], velocities[:index + 1])
        ending = ""
        if index == len(records) - 1:
            ending = "  |  FINISHED" if row["position"] >= DISTANCE else "  |  TIME LIMIT"
        state = "AIR" if row["airborne"] else STATUS_NAMES.get(row["action"], row["action"])
        status.set_text(f"{row['simulation_time']:5.1f} s   |   {row['position']:6.2f} / 200 m   |   "
                        f"{row['velocity']:.2f} m/s   |   {state:5s}   |   {row['hits']} hits{ending}")
        for bar, value, key in zip(activity_bars, values, keys):
            reading = row.get(key)
            shown = neural and reading is not None
            bar.set_width(float(reading) if shown else 0)
            value.set_text(f"{reading:.3f}" if shown else "N/A")
        return [athlete, bars, speed_line, status, *activity_bars, *values]

    update(0)
    return fig, update


GIF_FPS = 10
# Slower than Sprint's 5x: a jump lasts 0.8 s, which would be one or two frames at 5x.
GIF_SPEEDUP = 2.5


def gif_frames(records: Sequence[dict]) -> list[int]:
    """Record indices for a GIF_SPEEDUP x real-time GIF at GIF_FPS, always ending on the last step."""
    dt = records[0]["simulation_time"]  # the first record is at t = dt
    stride = max(1, round(GIF_SPEEDUP / (GIF_FPS * dt)))
    return sorted(set([0, *range(stride - 1, len(records), stride), len(records) - 1]))


def save_hurdles_gif(fig, update, records: Sequence[dict], output: str | Path) -> None:
    """Draw each frame on the canvas and let Pillow write the GIF (much faster than PillowWriter)."""
    from PIL import Image

    images = []
    for index in gif_frames(records):
        update(index)
        fig.canvas.draw()
        size = fig.canvas.get_width_height()
        images.append(Image.frombuffer("RGBA", size, fig.canvas.buffer_rgba()).convert("RGB"))
    images[0].save(output, save_all=True, append_images=images[1:], duration=1000 // GIF_FPS, loop=0)


def animate_hurdles(records: Sequence[dict], hurdles: Sequence[float], controller_name: str, *,
                    show: bool = True, output: str | Path | None = None) -> None:
    """Display a logged run or export a GIF at 2.5x speed. Physics stays fixed."""
    from matplotlib.animation import FuncAnimation

    fig, update = create_hurdles_view(records, hurdles, controller_name)
    if output is not None:
        fig.set_dpi(85)
    # Lay the figure out once; re-running constrained layout on every frame doubles the cost.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    if output is not None:
        save_hurdles_gif(fig, update, records, output)
        fig.savefig(Path(output).with_suffix(".activity.png"), dpi=130)
    if show:
        playback = FuncAnimation(fig, update, frames=len(records), interval=50, repeat=False)
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
