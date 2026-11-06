"""Show an Archery end to a pretrained FlyVis model of the fly's motion-vision system.

Exploratory and optional: needs the separate vision environment (see docs/vision.md).
Nothing here feeds the Archery controllers yet; it asks one question first: do the
real-connectome motion detectors (T4/T5) in FlyVis pick up the crosswind?

What the fly "sees" is a modeling choice, written here: a grass speckle field that drifts
sideways with the recorded wind, and the target, which shifts as the fly changes its aim.
FlyVis (Lappalainen et al., Nature 2024, MIT licence) supplies the 45,669-neuron visual
network, built from the fly's real optic-lobe wiring and trained on optic flow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.run_archery import run

SIZE = 64            # rendered frame, pixels
DT = 0.01            # FlyVis integration step (s)
SIM_DT = 0.1         # Archery step (s): 10 FlyVis frames per step
WIND_PIXELS = 0.1    # speckle drift in pixels per FlyVis frame per unit of wind (T4/T5 range)
AIM_PIXELS = 30.0    # target shift in pixels per unit of aim


def render(records: list[dict], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Frames (n, SIZE, SIZE) in [0, 1] and the wind at each frame."""
    rng = np.random.default_rng(seed)
    texture = rng.random((SIZE, 4 * SIZE)) < 0.25            # sparse bright grass specks
    texture = 0.25 + 0.6 * texture
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    frames, winds, offset = [], [], 0.0
    steps = round(SIM_DT / DT)
    for record in records:
        for _ in range(steps):
            offset += WIND_PIXELS * record["wind"]
            frame = np.roll(texture, int(round(offset)), axis=1)[:, :SIZE].copy()
            # Target: dark and light rings, centred where the current aim puts it.
            cx = SIZE / 2 - AIM_PIXELS * record["aim_offset"]
            r = np.hypot(xx - cx, yy - SIZE / 2)
            disc = r < 14
            frame[disc] = np.where((r[disc] // 3.5) % 2 == 0, 0.05, 0.95)
            frames.append(frame)
            winds.append(record["wind"])
    return np.asarray(frames, dtype=np.float32), np.asarray(winds)


def simulate(frames: np.ndarray):
    import torch
    import flyvis
    from flyvis import NetworkView
    from flyvis.datasets.rendering import BoxEye
    from flyvis.utils.activity_utils import LayerActivity

    network = NetworkView(flyvis.results_dir / "flow/0000/000").init_network()
    movie = BoxEye(extent=15, kernel_size=13)(torch.tensor(frames, device=flyvis.device)[None])[0]
    with torch.no_grad():
        state = network.fade_in_state(1.0, DT, movie[[0]])
        responses = network.simulate(movie[None], DT, initial_state=state).cpu()
    return LayerActivity(responses, network.connectome, keepref=True)


CALIBRATION = Path(__file__).resolve().parents[1] / "models" / "flyvis_wind_eye.json"


def horizontal_signal(activity) -> np.ndarray:
    """(T4a + T5a) - (T4b + T5b), averaged over all 721 eye columns, per frame."""
    h = {t: activity[t][0].numpy().mean(axis=-1) for t in ("T4a", "T4b", "T5a", "T5b")}
    return (h["T4a"] + h["T5a"]) - (h["T4b"] + h["T5b"])


def calibrate(levels=tuple(np.round(np.arange(-4, 4.01, 0.25), 2)), repeats: int = 3,
              block_steps: int = 10, seed: int = 0, output: Path = CALIBRATION) -> dict:
    """Steady FlyVis response to grass drifting at each wind level (no target in view).

    Each level is shown in 1 s blocks, in shuffled order, repeats times; the response is
    the mean over the second half of each block, after the network has adapted.
    """
    rng = np.random.default_rng(seed)
    order = [float(level) for _ in range(repeats) for level in rng.permutation(levels)]
    records = [{"wind": level, "aim_offset": 99.0} for level in order for _ in range(block_steps)]
    frames, _ = render(records)
    signal = horizontal_signal(simulate(frames))
    frames_per_block = block_steps * round(SIM_DT / DT)
    blocks: dict[float, list[float]] = {}
    for b, level in enumerate(order):
        blocks.setdefault(level, []).append(float(signal[b * frames_per_block + frames_per_block // 2:(b + 1) * frames_per_block].mean()))
    table = {
        "version": 1,
        "source": "FlyVis pretrained network flow/0000/000 (Lappalainen et al., Nature 2024)",
        "readout": "(T4a + T5a) - (T4b + T5b), mean over 721 columns, second half of 1 s blocks",
        "scene": f"grass specks drifting {WIND_PIXELS} px per 10 ms frame per unit of wind, no target",
        "wind": [float(level) for level in levels],
        "signal": [float(np.mean(blocks[float(level)])) for level in levels],
        "signal_sd": [float(np.std(blocks[float(level)])) for level in levels],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2) + "\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="rule")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "vision")
    parser.add_argument("--calibrate", action="store_true", help=f"Write the wind calibration to {CALIBRATION}")
    args = parser.parse_args()
    if args.calibrate:
        table = calibrate()
        print(f"Wrote {CALIBRATION}: " + ", ".join(f"{w:+.2f}:{s:.4f}" for w, s in zip(table["wind"], table["signal"])))
        return

    records = run(args.controller, args.seed)
    frames, winds = render(records)
    # Horizontal motion: T4/T5 a and b prefer opposite horizontal directions.
    signal = horizontal_signal(simulate(frames))
    settle = round(0.5 / DT)  # ignore the first half second while the network settles
    r = float(np.corrcoef(signal[settle:], winds[settle:])[0, 1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    time = np.arange(len(winds)) * DT
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    top.plot(time, winds, color="tab:blue")
    top.set(ylabel="recorded crosswind", title=f"Archery seed {args.seed}: what the FlyVis motion detectors see")
    bottom.plot(time, signal, color="tab:red")
    bottom.set(xlabel="time (s)", ylabel="T4/T5 a minus b\n(mean activity)")
    bottom.text(0.01, 0.9, f"correlation with wind: r = {r:.2f}", transform=bottom.transAxes)
    fig.savefig(args.output_dir / "archery_eye.png", dpi=150)
    plt.close(fig)
    np.save(args.output_dir / "archery_eye_frames.npy", frames[::10])
    summary = {"controller": args.controller, "seed": args.seed, "frames": len(frames),
               "correlation_with_wind": round(r, 3), "cell_types": ["T4a", "T4b", "T5a", "T5b"]}
    (args.output_dir / "archery_eye.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
