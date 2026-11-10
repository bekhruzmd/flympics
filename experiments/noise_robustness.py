"""How much neural noise do the evolution-trained Archery and Steeplechase flies tolerate?

Gaussian noise (std = noise) is added to every neuron's drive on every substep
(RateNetwork(noise=...)). Each run is seeded: controller.reset(seed) reseeds the noise,
so every number here is reproducible.

    python -m experiments.noise_robustness            # writes results/noise_robustness/

Three measurements:
  1. toy:      a bare threshold decision "x > 1e-5" with noise added to x.
  2. readout:  how much the noise moves extensor - flexor (the value the thresholds cut),
               replaying the noiseless run's observations through a noisy network.
  3. events:   score vs noise for trained, untrained and rule controllers.
               Archery: mean points per end on held-out winds 1000-1099 (max 60).
               Steeplechase: the course is fixed, so seeds 1000-1099 only seed the noise;
               mean race score (s, lower is better) and how many of the 100 races finish.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from brain.connectome import load_manc_front_leg_circuit
from brain.controller import ARCHERY_POLICY, STEEPLECHASE_POLICY, FlyBrainController
from controllers.archery_baselines import ArcheryRuleController
from controllers.steeple_baselines import SteepleRuleController
from experiments import run_archery, run_steeplechase

NOISE_LEVELS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = tuple(range(1000, 1100))
TOY_MARGIN = 1e-5     # the toy decision value sits this far above its threshold
TOY_DRAWS = 100_000


def toy_flip_rates(levels=NOISE_LEVELS, threshold: float = 1e-5, draws: int = TOY_DRAWS) -> dict:
    """Fraction of decisions "go right if x > threshold" that flip when noise is added to x."""
    x = threshold + TOY_MARGIN
    out = {}
    for noise in levels:
        noisy = x + np.random.default_rng(0).normal(0.0, noise, draws) if noise else np.full(draws, x)
        out[noise] = float(np.mean(noisy <= threshold))
    return out


def make_controllers(event: str, circuit, noise: float, policies: dict[str, Path]) -> dict:
    untrained = FlyBrainController.archery if event == "archery" else FlyBrainController.steeplechase
    controllers = {name: FlyBrainController.from_policy(path, circuit, noise=noise) for name, path in policies.items()}
    controllers["untrained"] = untrained(circuit=circuit, noise=noise)
    controllers["rule"] = ArcheryRuleController() if event == "archery" else SteepleRuleController()
    return controllers


def readout_noise(controller_factory, observations: list[dict], levels, seeds) -> dict:
    """Std of (noisy - noiseless) extensor - flexor, and the fraction of decisions that
    differ, along one fixed observation sequence (open loop: the noise cannot change
    what the fly sees, only what it decides)."""
    def trace(noise, seed):
        controller = controller_factory(noise)
        controller.reset(seed)
        actions, diffs = [], []
        for observation in observations:
            actions.append(controller.act(observation))
            diffs.append(controller.last_summary["motor_difference"])
        return np.array(actions), np.array(diffs)

    clean_actions, clean_diffs = trace(0.0, None)
    out = {}
    for noise in levels:
        if not noise:
            continue
        runs = [trace(noise, seed) for seed in seeds]
        out[noise] = {
            "difference_noise_std": float(np.std([d - clean_diffs for _, d in runs])),
            "decision_flip_rate": float(np.mean([a != clean_actions for a, _ in runs])),
        }
    return out


def archery_scores(controller, seeds=SEEDS) -> dict:
    ends = [run_archery.summary(run_archery.run(seed=s, controller=controller)) for s in seeds]
    return {"mean": float(np.mean([e["points"] for e in ends])),
            "min": float(min(e["points"] for e in ends)),
            "timeouts": int(sum(e["timeouts"] for e in ends))}


def steeple_scores(controller, seeds=SEEDS) -> dict:
    races = [run_steeplechase.run(seed=s, controller=controller) for s in seeds]
    scores = [run_steeplechase.score(r) for r in races]
    return {"mean": float(np.mean(scores)), "worst": float(max(scores)),
            "finished": int(sum(r[-1]["position"] >= 600.0 for r in races)),
            "hits": float(np.mean([r[-1]["hits"] for r in races])),
            "splashes": float(np.mean([r[-1]["splashes"] for r in races]))}


def threshold_gaps(path: Path) -> dict:
    thresholds = json.loads(Path(path).read_text())["thresholds"]
    return {"thresholds": thresholds, "gap": thresholds[1] - thresholds[0],
            "distance_from_zero": min(abs(t) for t in thresholds)}


def measure(levels=NOISE_LEVELS, seeds=SEEDS, policies: dict | None = None, log=print) -> dict:
    """policies maps event -> {controller name: policy JSON}; defaults to the saved trained flies."""
    circuit = load_manc_front_leg_circuit()
    policies = policies or {"archery": {"trained": ARCHERY_POLICY}, "steeplechase": {"trained": STEEPLECHASE_POLICY}}
    report = {"noise_levels": list(levels), "seeds": f"{seeds[0]}-{seeds[-1]}",
              "toy": {"threshold": 1e-5, "margin": TOY_MARGIN, "draws": TOY_DRAWS,
                      "flip_rate": {str(k): v for k, v in toy_flip_rates(levels).items()}}}
    for event, scorer in (("archery", archery_scores), ("steeplechase", steeple_scores)):
        event_report = {"thresholds": {n: threshold_gaps(p) for n, p in policies[event].items()},
                        "scores": {}, "readout": {}}
        for noise in levels:
            for name, controller in make_controllers(event, circuit, noise, policies[event]).items():
                if name == "rule" and noise:
                    # Non-neural: noise cannot touch it, so its score is the noiseless one.
                    event_report["scores"].setdefault(name, {})[str(noise)] = event_report["scores"][name]["0.0"]
                    continue
                event_report["scores"].setdefault(name, {})[str(noise)] = result = scorer(controller, seeds)
                log(f"{event:12s} {name:10s} noise {noise:7.0e}  {result}")
        # Open-loop readout diagnostic on one noiseless run of each neural fly.
        for name in [*policies[event], "untrained"]:
            def factory(noise, name=name):
                return make_controllers(event, circuit, noise, policies[event])[name]
            if event == "archery":
                records = run_archery.run(seed=seeds[0], controller=factory(0.0))
            else:
                records = run_steeplechase.run(seed=seeds[0], controller=factory(0.0))
            observations = observations_of(event, records, seeds[0])
            event_report["readout"][name] = {str(k): v for k, v in
                                             readout_noise(factory, observations, levels, seeds[:20]).items()}
        report[event] = event_report
    return report


def observations_of(event: str, records: list[dict], seed: int) -> list[dict]:
    """Replay the recorded actions in a fresh event to recover the observations the fly saw."""
    if event == "archery":
        from events.archery import Archery
        env = Archery()
    else:
        from events.steeplechase import Steeplechase
        env = Steeplechase()
    observation, observations = env.reset(seed), []
    for record in records:
        observations.append(dict(observation))
        observation = env.step(record["action"])
    return observations


# Reference categorical palette, slots 1-4 (blue, orange, aqua, yellow), fixed order.
COLORS = {"trained": "#2a78d6", "untrained": "#eb6834", "rule": "#1baf7a", "noise-trained": "#eda100"}
STYLES = {"trained": ("o", "-"), "untrained": ("s", "--"), "rule": ("", ":"), "noise-trained": ("^", "-")}


def plot(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    levels = report["noise_levels"]
    # Noise 0 cannot sit on a log axis: draw it one decade left of the smallest level, labelled "0".
    xs = [levels[1] / 10 if n == 0 else n for n in levels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    panels = (("archery", "mean points per end (max 60), seeds 1000-1099", "higher is better"),
              ("steeplechase", "mean race score (s), 100 noise seeds", "lower is better"))
    for axis, (event, ylabel, note) in zip(axes, panels):
        for name, by_noise in report[event]["scores"].items():
            marker, line = STYLES.get(name, ("o", "-"))
            ys = [by_noise[str(n)]["mean"] for n in levels]
            axis.plot(xs, ys, marker=marker, linestyle=line, linewidth=2, markersize=8,
                      color=COLORS.get(name, "#6250d6"), label=name)
        axis.set_xscale("log")
        axis.set_xticks(xs, ["0"] + [f"1e{int(np.log10(n))}" for n in levels[1:]])
        axis.set(xlabel="noise std on neuron drive", ylabel=ylabel, title=f"{event.title()} ({note})")
        axis.grid(True, which="major", color="#e6e5e0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[1].invert_yaxis()  # up is better in both panels
    fig.suptitle("Score vs Gaussian noise on every neuron's drive (rule is non-neural: unaffected)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_csv(report: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event", "controller", "noise", "mean", "detail"])
        for event in ("archery", "steeplechase"):
            for name, by_noise in report[event]["scores"].items():
                for noise, result in by_noise.items():
                    detail = {k: v for k, v in result.items() if k != "mean"}
                    writer.writerow([event, name, noise, round(result["mean"], 3), json.dumps(detail)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "noise_robustness")
    parser.add_argument("--archery-policy", type=Path, action="append", default=[],
                        help="Extra archery policy as NAME=PATH (e.g. noise-trained=models/x.json)")
    parser.add_argument("--steeplechase-policy", type=Path, action="append", default=[],
                        help="Extra steeplechase policy as NAME=PATH")
    args = parser.parse_args()
    policies = {"archery": {"trained": ARCHERY_POLICY}, "steeplechase": {"trained": STEEPLECHASE_POLICY}}
    for event, extra in (("archery", args.archery_policy), ("steeplechase", args.steeplechase_policy)):
        for item in extra:
            name, _, path = str(item).partition("=")
            policies[event][name] = Path(path)
    report = measure(policies=policies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save_csv(report, args.output_dir / "scores.csv")
    plot(report, args.output_dir / "score_vs_noise.png")
    print(f"Wrote {args.output_dir / 'summary.json'}, scores.csv and score_vs_noise.png")


if __name__ == "__main__":
    main()
