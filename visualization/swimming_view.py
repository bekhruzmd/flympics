"""Static plot and animated viewer for logged Swimming runs; drawing never changes physics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

DISTANCE = 100.0
POOL = 50.0
NEURAL_CONTROLLERS = {"fly"}


def _glide_spans(records: Sequence[dict]) -> list[tuple[float, float]]:
    """(start, end) simulation times of each run of consecutive glide steps."""
    spans, start = [], None
    for previous, row in zip([None, *records], records):
        begin = previous["simulation_time"] if previous else 0.0
        if row["gliding"] and start is None:
            start = begin
        elif not row["gliding"] and start is not None:
            spans.append((start, begin)); start = None
    if start is not None:
        spans.append((start, records[-1]["simulation_time"]))
    return spans


def _turn_time(records: Sequence[dict]) -> float | None:
    return next((r["simulation_time"] for r in records if r["turns"] > 0), None)


def save_swimming_plot(records: Sequence[dict], output: str | Path) -> None:
    """Distance and speed over simulation time, with the turn and the glides marked."""
    if not records:
        raise ValueError("The plot needs at least one timestep.")
    times = [r["simulation_time"] for r in records]
    fig, (track, speed) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    track.plot(times, [r["position"] for r in records], color="tab:green", label="distance swum")
    track.axhline(POOL, color="tab:gray", linestyle=":", label="turn wall")
    track.axhline(DISTANCE, color="tab:orange", linestyle="--", label="finish")
    track.set(ylabel="metres", title="Flympics — 100 m Swimming")
    speed.plot(times, [r["velocity"] for r in records], color="tab:blue", linewidth=1)
    for i, (start, end) in enumerate(_glide_spans(records)):
        speed.axvspan(start, end, color="tab:cyan", alpha=0.2, label="glide" if i == 0 else None)
    turn = _turn_time(records)
    for axis in (track, speed):
        if turn is not None:
            axis.axvline(turn, color="tab:gray", linestyle=":", label="turn" if axis is speed else None)
    speed.set(xlabel="simulation time (s)", ylabel="m/s")
    track.legend(loc="lower right")
    if speed.get_legend_handles_labels()[0]:
        speed.legend(loc="upper right")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _fly_drawing(facing: int):
    """Diagrammatic fly, facing right (+1) or left (-1): an illustration, not gait data."""
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Ellipse

    def x(value: float) -> float:
        return value if facing > 0 else 70 - value

    fly = DrawingArea(70, 48, 0, 0)
    for leg in (23, 33, 43):
        for direction in (-1, 1):
            fly.add_artist(plt.Line2D([x(leg), x(leg - 7)], [24, 24 + direction * 17],
                                      color="#343d35", linewidth=1.5))
    for y, angle in ((33, 22), (15, -22)):
        fly.add_artist(Ellipse((x(29), y), 35, 15, angle=angle * facing,
                               facecolor="#d5e8e4", edgecolor="#839c96", alpha=0.85))
    fly.add_artist(Ellipse((x(31), 24), 29, 14, facecolor="#655448"))
    fly.add_artist(Ellipse((x(47), 24), 17, 16, facecolor="#363d34"))
    fly.add_artist(Ellipse((x(52), 28), 6, 6, facecolor="#c95a47"))
    return fly


def create_swimming_view(records: Sequence[dict], controller_name: str):
    """Build the pool view from measured logs.

    The fly swims out to the 50 m wall and back. Activity is dimensionless modeled rate
    (0-1), not firing Hz.
    """
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox

    if not records:
        raise ValueError("The swimming viewer needs at least one timestep.")
    fig = plt.figure(figsize=(11, 6), facecolor="#eef5f7", layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1])
    pool = fig.add_subplot(grid[0, :])
    history = fig.add_subplot(grid[1, 0])
    card = fig.add_subplot(grid[1, 1])
    fig.suptitle(f"FLYMPICS / 100 m Swimming / {controller_name} controller", fontsize=16)

    pool.set(xlim=(-2, POOL + 2), ylim=(-0.6, 1), xlabel="Position in the pool (m)", yticks=[])
    pool.set_facecolor("#d8ecf3")
    pool.spines[["left", "right", "top"]].set_visible(False)
    for wall in (0, POOL):
        pool.axvline(wall, color="#5d7f8c", linewidth=3)
    for mark in (5, POOL - 5):
        pool.axvline(mark, color="#5d7f8c", linestyle=":", linewidth=1)  # backstroke flags
    pool.text(0.5, 0.72, "START / FINISH", fontsize=9)
    pool.text(POOL - 0.5, 0.72, "TURN", ha="right", fontsize=9)
    status = pool.text(0.02, 0.02, "", transform=pool.transAxes, fontsize=11)
    swimmers = {facing: AnnotationBbox(_fly_drawing(facing), (0, 0.18), frameon=False)
                for facing in (1, -1)}
    for swimmer in swimmers.values():
        pool.add_artist(swimmer)

    times = np.array([r["simulation_time"] for r in records])
    velocities = np.array([r["velocity"] for r in records])
    history.set(xlim=(0, max(times[-1], 1)), ylim=(0, max(0.6, velocities.max() * 1.2)),
                xlabel="Simulation time (s)", ylabel="Speed (m/s)", title="Swimmer speed")
    history.spines[["right", "top"]].set_visible(False)
    speed_line, = history.plot([], [], color="#2f6f8f", linewidth=1.5)

    neural = controller_name in NEURAL_CONTROLLERS
    keys = ("stroke_phase", "grip", "sensory_mean", "intrinsic_mean", "motor_extensor", "motor_flexor")
    labels = ("Arm (stroke phase)", "Grip", "Sensory input", "VNC interneurons",
              "Extensor output", "Flexor output")
    colors = ("#2f6f8f", "#5aa6c8", "#4089aa", "#8b71b3", "#3d946d", "#d19543")
    card.set(xlim=(0, 1.23), ylim=(-0.7, len(keys) - 0.3), yticks=range(len(keys)),
             yticklabels=labels, xticks=[0, 0.5, 1], xlabel="Value (0-1)",
             title="Stroke and circuit activity" if neural else "Stroke · circuit activity unavailable")
    card.invert_yaxis()
    card.spines[["right", "top"]].set_visible(False)
    card.barh(range(len(keys)), [1] * len(keys), color="#e2e8ea", height=0.5)
    bars = card.barh(range(len(keys)), [0] * len(keys), color=colors, height=0.5)
    values = [card.text(1.04, i, "", va="center", fontsize=10) for i in range(len(keys))]
    note = fig.supxlabel("Selected VNC populations, not a whole-brain map. Rates and action mapping are modeled.",
                         fontsize=9, color="#60665d")
    if not neural:
        note.set_text("Baseline controller: no neural model. Circuit bars are not applicable.")

    def update(index):
        row = records[index]
        facing = 1 if row["lap"] == 1 else -1
        x = row["position"] if facing > 0 else DISTANCE - row["position"]
        for direction, swimmer in swimmers.items():
            swimmer.set_visible(direction == facing)
            swimmer.xy = swimmer.xybox = (x, 0.18)
        speed_line.set_data(times[:index + 1], velocities[:index + 1])
        ending = ""
        if index == len(records) - 1:
            ending = "  |  FINISHED" if row["position"] >= DISTANCE else "  |  TIME LIMIT"
        state = "GLIDE" if row["gliding"] else row["action"]
        status.set_text(f"{row['simulation_time']:5.1f} s   |   lap {row['lap']}   |   "
                        f"{row['position']:.2f} / 100 m   |   {row['velocity']:.2f} m/s   |   {state}{ending}")
        for bar, value, key in zip(bars, values, keys):
            available = neural or key in ("stroke_phase", "grip")
            reading = row.get(key)
            shown = available and reading is not None
            bar.set_width(float(reading) if shown else 0)
            value.set_text(f"{reading:.3f}" if shown else "N/A")
        return [*swimmers.values(), speed_line, status, *bars, *values]

    update(0)
    return fig, update


GIF_FPS = 10
GIF_SPEEDUP = 5


def gif_frames(records: Sequence[dict]) -> list[int]:
    """Record indices for a GIF_SPEEDUP x real-time GIF at GIF_FPS, always ending on the last step."""
    dt = records[0]["simulation_time"]  # the first record is at t = dt
    stride = max(1, round(GIF_SPEEDUP / (GIF_FPS * dt)))
    return sorted(set([0, *range(stride - 1, len(records), stride), len(records) - 1]))


def save_swimming_gif(fig, update, records: Sequence[dict], output: str | Path) -> None:
    """Draw each frame on the canvas and let Pillow write the GIF.

    Matplotlib's PillowWriter re-saves the figure at a new dpi on every frame, which made a
    44 s race take minutes; drawing once per frame at the figure's own dpi is several times faster.
    """
    from PIL import Image

    images = []
    for index in gif_frames(records):
        update(index)
        fig.canvas.draw()
        size = fig.canvas.get_width_height()
        images.append(Image.frombuffer("RGBA", size, fig.canvas.buffer_rgba()).convert("RGB"))
    images[0].save(output, save_all=True, append_images=images[1:], duration=1000 // GIF_FPS, loop=0)


def animate_swimming(records: Sequence[dict], controller_name: str, *, show: bool = True,
                     output: str | Path | None = None) -> None:
    """Display a logged run or export a compact 5x-speed GIF. Physics stays fixed."""
    from matplotlib.animation import FuncAnimation

    fig, update = create_swimming_view(records, controller_name)
    if output is not None:
        fig.set_dpi(85)
    # Lay the figure out once; re-running constrained layout on every frame doubles the cost.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    if output is not None:
        save_swimming_gif(fig, update, records, output)
        fig.savefig(Path(output).with_suffix(".activity.png"), dpi=130)
    if show:
        playback = FuncAnimation(fig, update, frames=len(records), interval=100, repeat=False)
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
