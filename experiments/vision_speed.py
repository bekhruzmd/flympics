"""Calibrate FlyVis on a drawn, downward-facing view of the Sprint track."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain.speed_eye import SPEED_CALIBRATION
from experiments.vision_archery import DT, SIZE, horizontal_signal, simulate

SPEEDS = np.linspace(0.0, 8.0, 21)
PIXELS_PER_BODY_LENGTH = 5.0
FRAMES_PER_SPEED = round(1.0 / DT)


def render(speed: float, texture: np.ndarray) -> np.ndarray:
    """Sample a fixed-height ground view with forward along the image's x-axis."""
    # The camera advances to the right, so the ground moves left across the eye.
    x = np.arange(SIZE) + np.arange(FRAMES_PER_SPEED)[:, None] * (
        speed * PIXELS_PER_BODY_LENGTH * DT
    )
    left = x.astype(int)
    fraction = x - left
    return ((1.0 - fraction[:, None, :]) * texture.T[left].transpose(0, 2, 1)
            + fraction[:, None, :] * texture.T[left + 1].transpose(0, 2, 1)).astype(np.float32)


def calibrate(output: Path = SPEED_CALIBRATION) -> dict:
    rng = np.random.default_rng(0)
    repeats = 3
    samples = [[] for _ in SPEEDS]
    # Reuse Archery's FlyVis simulation. Small batches of consecutive speed
    # blocks bound memory, and the first half of each block is discarded.
    for repeat in range(repeats):
        texture = 0.25 + 0.6 * (rng.random((SIZE, 2 * SIZE)) < 0.25)
        for group in np.array_split(rng.permutation(len(SPEEDS)), 3):
            frames = np.concatenate([render(SPEEDS[i], texture) for i in group])
            signal = horizontal_signal(simulate(frames)).reshape(len(group), FRAMES_PER_SPEED)
            for i, block in zip(group, signal):
                samples[i].append(float(block[FRAMES_PER_SPEED // 2:].mean()))
        print(f"Measured all {len(SPEEDS)} speeds on texture {repeat + 1}/{repeats}", flush=True)

    table = {
        "version": 1,
        "source": "FlyVis pretrained network flow/0000/000 (Lappalainen et al., Nature 2024)",
        "readout": "(T4a + T5a) - (T4b + T5b), mean over 721 columns and the last 0.5 s of each block",
        "scene": "Drawn downward view of ground specks, moving left at 5 px per body length",
        "frame_dt_seconds": DT,
        "block_seconds": FRAMES_PER_SPEED * DT,
        "texture_repeats": repeats,
        "seed": 0,
        "speed_bl_per_s": SPEEDS.tolist(),
        "signal": np.mean(samples, axis=1).tolist(),
        "signal_sd": np.std(samples, axis=1).tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    return table


def plot(table: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.errorbar(table["speed_bl_per_s"], table["signal"], yerr=table["signal_sd"], fmt="o-",
                capsize=3, color="tab:blue", label="Mean ± SD across three textures")
    ax.set(xlabel="Ground speed (body lengths/s)", ylabel="Horizontal T4/T5 activity",
           title="FlyVis response to drawn Sprint ground motion")
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPEED_CALIBRATION)
    parser.add_argument("--plot", type=Path, default=Path("results/vision/speed_eye.png"))
    args = parser.parse_args()
    plot(calibrate(args.output), args.plot)
    print(f"Calibration written to {args.output}; plot written to {args.plot}")


if __name__ == "__main__":
    main()
