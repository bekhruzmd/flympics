"""Compare lesioned MANC circuits with the unchanged control on the 100 m sprint.

Hypotheses (fixed before running):
  H1  Without sensory neurons no input reaches the circuit; the fly never accelerates.
  H2  Removing the top-k interneuron drivers of extensors lowers extensor - flexor,
      so the trained policy scores worse than control and than k random interneurons.
  H3  Removing the top-k interneuron drivers of flexors raises the difference, so
      the untrained policy may begin to accelerate.
Metric: Sprint score -- finish time, or 60 s + metres remaining; lower is better.

The sprint and the fly controller are deterministic, so seeds only choose the
random interneuron lesions that serve as a size-matched control.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from brain.ablation import ablate, top_drivers
from brain.connectome import ConnectomeCircuit, load_manc_front_leg_circuit
from brain.controller import FlyBrainController
from events.sprint import ACCELERATE, Sprint
from experiments.run_sprint import run


class Lesion(TypedDict):
    condition: str
    seed: int | None  # set only for random lesions
    neurons: list[int]


class SprintOutcome(TypedDict):
    score: float
    finished: bool
    time: float
    distance: float
    accelerate_fraction: float
    mean_motor_difference: float


class AblationResult(SprintOutcome):
    condition: str
    seed: int | None
    policy: str
    n_ablated: int
    neurons: list[int]


def conditions(circuit: ConnectomeCircuit, k: int, seeds: list[int]) -> list[Lesion]:
    named = [
        ("control", []),
        ("no_sensory", circuit.indices("sensory")),
        ("no_intrinsic", circuit.indices("intrinsic")),
        (f"extensor_drivers_top{k}", top_drivers(circuit, "motor_extensor", k)),
        (f"flexor_drivers_top{k}", top_drivers(circuit, "motor_flexor", k)),
        ("no_extensor", circuit.indices("motor_extensor")),
        ("no_flexor", circuit.indices("motor_flexor")),
    ]
    result = [Lesion(condition=name, seed=None, neurons=[int(i) for i in neurons])
              for name, neurons in named]
    intrinsic = circuit.indices("intrinsic")
    for seed in seeds:
        chosen = np.random.default_rng(seed).choice(intrinsic, size=k, replace=False)
        result.append(Lesion(condition=f"random_intrinsic_{k}", seed=seed,
                             neurons=sorted(int(i) for i in chosen)))
    return result


def evaluate(controller: FlyBrainController, seed: int) -> SprintOutcome:
    records = run(seed=seed, controller=controller)
    final = records[-1]
    finished = final["distance_to_finish"] == 0
    score = final["simulation_time"] if finished else Sprint().max_time + final["distance_to_finish"]
    return SprintOutcome(
        score=round(score, 8),
        finished=finished,
        time=final["simulation_time"],
        distance=final["position"],
        accelerate_fraction=sum(r["action"] == ACCELERATE for r in records) / len(records),
        mean_motor_difference=float(np.mean([r["motor_difference"] for r in records])),
    )


def run_experiment(k: int = 10, seeds: tuple[int, ...] = tuple(range(50))) -> list[AblationResult]:
    circuit = load_manc_front_leg_circuit()
    rows: list[AblationResult] = []
    for spec in conditions(circuit, k, list(seeds)):
        lesioned = ablate(circuit, spec["neurons"])
        seed = 7 if spec["seed"] is None else spec["seed"]
        for policy, controller in (("untrained", FlyBrainController(circuit=lesioned)),
                                   ("trained", FlyBrainController.trained(circuit=lesioned))):
            rows.append(AblationResult(condition=spec["condition"], seed=spec["seed"], policy=policy,
                                       n_ablated=len(spec["neurons"]), neurons=spec["neurons"],
                                       **evaluate(controller, seed)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=10, help="interneurons removed in targeted/random lesions")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)),
                        help="one random interneuron lesion per seed (size-matched control)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = run_experiment(args.k, tuple(args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"ablation_k{args.k}"
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "neurons"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'condition':<24}{'seed':>5}  {'policy':<10}{'removed':>8}{'score':>10}{'accel':>7}  mean ext-flex")
    for row in rows:
        if row["seed"] is not None:
            continue  # random lesions are summarized below
        seed = "" if row["seed"] is None else row["seed"]
        print(f"{row['condition']:<24}{seed:>5}  {row['policy']:<10}{row['n_ablated']:>8}"
              f"{row['score']:>10.2f}{row['accelerate_fraction']:>7.0%}  {row['mean_motor_difference']:+.6f}")
    for policy in ("untrained", "trained"):
        random = [row for row in rows if row["seed"] is not None and row["policy"] == policy]
        if not random:
            continue
        differences = [row["mean_motor_difference"] for row in random]
        print(f"random_intrinsic_{args.k} ({policy}): {sum(row['finished'] for row in random)}/{len(random)} "
              f"finished; mean ext-flex {min(differences):+.6f} to {max(differences):+.6f}")
    print(f"Logs: {stem}.json/.csv (every random lesion is listed there)")


if __name__ == "__main__":
    main()
