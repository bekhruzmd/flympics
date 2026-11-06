"""Teach the Archery fly with food: reward-modulated learning on the real connection strengths.

The fly senses the crosswind through a FlyVis-calibrated eye (brain/eye.py), feels its own
aim and bow draw, and learns arrow by arrow from food (brain/reward_learning.py). It starts
from the untrained Archery settings, so everything it gains comes from the food.

    python -m experiments.train_archery_food --ends 3000 --drift 50
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from brain.archery_io import ARCHERY_ACTIONS, ARCHERY_INITIAL_THRESHOLDS, ArcheryDecoder, initial_archery_encoder_weights
from brain.connectome import load_manc_front_leg_circuit
from brain.controller import FlyBrainController
from brain.dynamics import RateNetwork
from brain.eye import ARCHERY_EYE_INPUTS, ArcheryEyeEncoder, FlyVisWindEye
from brain.reward_learning import DRIFT_LIMITS, LearningSettings, RewardModulatedFly, food, tuned_circuit
from brain.sensory import save_encoder_weights
from events.archery import Archery
from experiments.es_training import fingerprint
from experiments.train_archery import evaluate

MODELS = Path(__file__).resolve().parents[1] / "models"
TRAIN_SEEDS = range(0, 500)          # winds the fly practises on, in order, repeating
VALIDATION_SEEDS = tuple(range(1000, 1100))  # never practised on


def practise_end(fly: RewardModulatedFly, seed: int, learn: bool = True) -> tuple[int, list[float]]:
    """One end of six arrows; food after every arrow (learned from unless learn=False)."""
    event = Archery()
    observation = event.reset(seed)
    fly.reset()
    meals = []
    while not event.is_finished():
        arrow, wind = event.arrow, observation["wind"]
        observation = event.step(ARCHERY_ACTIONS[fly.act(observation)])
        if event.arrow != arrow:  # an arrow just landed (or timed out)
            meal = food(event.points[-1] if event.landings[-1] is not None else None)
            if learn:
                fly.reward(meal, wind)
            else:
                fly.clear_traces()
            meals.append(meal)
    return event.total_points, meals


def controller_for(circuit, fly: RewardModulatedFly, eye: FlyVisWindEye) -> FlyBrainController:
    """The learned settings in an ordinary controller, with no jitter, for fair evaluation.

    Uses the same neuron gain the fly practised with (this experiment's modeling choice).
    """
    tuned = tuned_circuit(circuit, fly.multipliers)
    controller = FlyBrainController(circuit=tuned, encoder=ArcheryEyeEncoder(tuned, fly.inputs, eye),
                                    decoder=ArcheryDecoder(tuned, list(fly.thresholds)))
    controller.network = RateNetwork(tuned, leak=fly.settings.leak, gain=fly.settings.gain)
    return controller


def evaluate_with_own_noise(fly: RewardModulatedFly, seeds=VALIDATION_SEEDS) -> dict:
    """Held-out winds, with the fly's own neural noise on and learning frozen: how it
    actually performs, since it practised (and decides) with that noise."""
    points = [practise_end(fly, s, learn=False)[0] for s in seeds]
    return {"validation_seeds": f"{seeds[0]}-{seeds[-1]}", "mean_points": round(float(np.mean(points)), 3),
            "min_points": int(min(points)), "max_points": int(max(points))}


def resting_thresholds(circuit, encoder, gain: float, half_band: float = 0.003) -> tuple[float, float]:
    """Readout thresholds centred on the untrained circuit's resting output (aim straight,
    no wind, bow drawn), so the fly starts with no bias to the left or right."""
    network = RateNetwork(circuit, gain=gain)
    for _ in range(40):
        state = network.step(encoder.encode({"aim_offset": 0.0, "wind": 0.0, "draw": 1.0}))
    rest = float(state[circuit.indices("motor_extensor")].mean() - state[circuit.indices("motor_flexor")].mean())
    return rest - half_band, rest + half_band


def drift_report(circuit, multipliers: np.ndarray) -> dict:
    change = np.abs(multipliers - 1.0)
    coo = circuit.weights.tocoo()
    by_pair: dict[str, list[float]] = {}
    for post, pre, c in zip(coo.row, coo.col, change):
        by_pair.setdefault(f"{circuit.roles[pre]} -> {circuit.roles[post]}", []).append(float(c))
    return {
        "mean_change_percent": round(100 * float(change.mean()), 2),
        "max_change_percent": round(100 * float(change.max()), 2),
        "connections_changed_over_10_percent": int((change > 0.10).sum()),
        "connections": int(change.size),
        "mean_change_percent_by_pathway": {k: round(100 * float(np.mean(v)), 2) for k, v in sorted(by_pair.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ends", type=int, default=3000, help="Practice ends (six arrows each)")
    parser.add_argument("--drift", choices=sorted(DRIFT_LIMITS), default="50")
    parser.add_argument("--noise", type=float, default=LearningSettings.noise)
    parser.add_argument("--weight-rate", type=float, default=LearningSettings.weight_rate)
    parser.add_argument("--input-rate", type=float, default=LearningSettings.input_rate)
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Neuron gain for this experiment. The rest of Flympics uses 3, which saturates "
                             "this circuit; 1 lets the eye's signal reach the motor readout (a modeling choice)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Policy JSON (default models/archery_food_<drift>.json)")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "archery_food")
    args = parser.parse_args()

    circuit = load_manc_front_leg_circuit()
    frozen = fingerprint(circuit)
    eye = FlyVisWindEye()
    start_inputs = initial_archery_encoder_weights(circuit)  # aim, eye, draw columns
    settings = LearningSettings(drift=args.drift, noise=args.noise, weight_rate=args.weight_rate,
                                input_rate=args.input_rate, gain=args.gain)
    start_encoder = ArcheryEyeEncoder(circuit, start_inputs, eye)
    thresholds = resting_thresholds(circuit, start_encoder, args.gain) if args.gain != 3.0 else ARCHERY_INITIAL_THRESHOLDS
    fly = RewardModulatedFly(circuit, start_encoder, thresholds, settings, args.seed)

    before = evaluate(controller_for(circuit, fly, eye), VALIDATION_SEEDS)
    before_noisy = evaluate_with_own_noise(fly)
    print(f"Before practice (held-out winds): {before_noisy['mean_points']} points per end with its own noise, "
          f"{before['mean_points']} without", flush=True)
    started = time.perf_counter()
    history = []
    seeds = list(TRAIN_SEEDS)
    for end in range(1, args.ends + 1):
        points, meals = practise_end(fly, seeds[(end - 1) % len(seeds)])
        history.append({"end": end, "points": points, "food": round(sum(meals), 3),
                        "usual_food": round(fly.baseline or 0.0, 4),
                        "mean_change_percent": round(100 * float(np.abs(fly.multipliers - 1).mean()), 3)})
        if end % 250 == 0:
            recent = np.mean([h["points"] for h in history[-250:]])
            print(f"end {end:5d}: last 250 ends average {recent:5.1f} points; connections moved "
                  f"{history[-1]['mean_change_percent']:.2f}% on average", flush=True)
    elapsed = time.perf_counter() - started
    if fingerprint(circuit) != frozen:
        raise RuntimeError("The MANC circuit changed in memory during training.")

    after = evaluate(controller_for(circuit, fly, eye), VALIDATION_SEEDS)
    after_noisy = evaluate_with_own_noise(fly)
    reference = json.loads((MODELS / "archery.json").read_text())
    output = args.output or MODELS / f"archery_food_{args.drift}.json"
    encoder_path = output.with_name(f"{output.stem}_encoder.csv")
    multipliers_path = output.with_name(f"{output.stem}_connections.csv")
    save_encoder_weights(encoder_path, circuit, fly.inputs, ARCHERY_EYE_INPUTS)
    coo = circuit.weights.tocoo()
    with multipliers_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pre", "post", "manc_weight", "multiplier"])
        for pre, post, w, m in zip(coo.col, coo.row, coo.data, fly.multipliers):
            writer.writerow([circuit.neuron_ids[pre], circuit.neuron_ids[post], repr(float(w)), repr(float(m))])
    policy = {
        "version": 1,
        "event": "archery",
        "method": "Reward-modulated learning (node perturbation) from food after each arrow, on MANC "
                  "connection strengths (multiplier within the drift limit) and input weights; wind sensed "
                  "through a FlyVis-calibrated eye; thresholds, wiring map and neuron model fixed",
        "inputs": list(ARCHERY_EYE_INPUTS),
        "actions": list(ARCHERY_ACTIONS),
        "thresholds": list(map(float, fly.thresholds)),
        "encoder": encoder_path.name,
        "connections": multipliers_path.name,
        "eye_calibration": "flyvis_wind_eye.json",
        "settings": vars(settings) | {"thresholds_from": "untrained resting output +/- 0.003" if args.gain != 3.0 else "Archery defaults", "ends": args.ends, "seed": args.seed, "drift_limits": DRIFT_LIMITS[args.drift],
                                       "train_seeds": f"{seeds[0]}-{seeds[-1]}, repeating", "food": "points / 10, +1 for a bullseye"},
        "training_seconds": round(elapsed, 1),
        "before_practice": {"with_own_noise": before_noisy, "without_noise": before},
        "after_practice": {"with_own_noise": after_noisy, "without_noise": after},
        "for_comparison": {"evolution_trained_wind_number_fly": reference["trained"], "untrained_wind_number_fly": reference["untrained"]},
        "drift": drift_report(circuit, fly.multipliers),
    }
    output.write_text(json.dumps(policy, indent=2) + "\n")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / f"history_{args.drift}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(history)
    import matplotlib.pyplot as plt
    points = np.array([h["points"] for h in history], dtype=float)
    window = max(1, min(100, len(points) // 10))
    smooth = np.convolve(points, np.ones(window) / window, mode="valid")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, layout="constrained")
    top.plot(np.arange(window, len(points) + 1), smooth, color="tab:green")
    top.axhline(before_noisy["mean_points"], color="tab:gray", linestyle="--", label=f"before, held-out winds ({before_noisy['mean_points']})")
    top.axhline(after_noisy["mean_points"], color="tab:green", linestyle=":", label=f"after, held-out winds ({after_noisy['mean_points']})")
    top.set(ylabel=f"points per end\n({window}-end average)", title=f"Archery fly learning from food (drift limit {args.drift})")
    top.legend(loc="lower right")
    bottom.plot([h["mean_change_percent"] for h in history], color="tab:purple")
    bottom.set(xlabel="practice end", ylabel="mean connection change (%)")
    fig.savefig(args.results_dir / f"learning_{args.drift}.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"before": before_noisy["mean_points"], "after": after_noisy["mean_points"],
                      "before_no_noise": before["mean_points"], "after_no_noise": after["mean_points"],
                      "evolution_trained": reference["trained"]["mean_points"], "drift": policy["drift"]["mean_change_percent"],
                      "seconds": policy["training_seconds"], "policy": str(output)}))


if __name__ == "__main__":
    main()
