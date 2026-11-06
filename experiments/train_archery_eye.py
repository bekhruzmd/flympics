"""Train the Archery fly with the evolution strategy, sensing wind through the FlyVis eye.

Identical to experiments.train_archery (same method, settings, seeds and neuron model)
except for one input: instead of being handed the wind as a number, the fly gets the
FlyVis-calibrated T4/T5 motion signal (brain/eye.py). Comparing the two answers whether
this circuit can aim using a real visual signal.
"""
from __future__ import annotations

from brain.archery_io import ARCHERY_ACTIONS, ARCHERY_INITIAL_THRESHOLDS, initial_archery_encoder_weights
from brain.controller import MODELS_DIR
from brain.eye import ARCHERY_EYE_INPUTS, ArcheryEyeEncoder
from events.archery import Archery
from experiments.es_training import EventSpec, cli
from experiments.train_archery import TRAIN_SEEDS, evaluate, fitness, teacher

POLICY = MODELS_DIR / "archery_eye.json"

SPEC = EventSpec(
    name="archery_eye", make_event=Archery, encoder_class=ArcheryEyeEncoder, inputs=ARCHERY_EYE_INPUTS,
    actions=ARCHERY_ACTIONS, initial_weights=initial_archery_encoder_weights,
    initial_thresholds=ARCHERY_INITIAL_THRESHOLDS, train_seeds=TRAIN_SEEDS, fitness=fitness, teacher=teacher,
    evaluate=evaluate, policy=POLICY, encoder=POLICY.with_name("archery_eye_encoder.csv"),
)


def main() -> None:
    cli(SPEC, __doc__, generations=60, imitation_generations=150, pairs=24, weight_sigma=0.3, threshold_sigma=0.02, lr=0.05)


if __name__ == "__main__":
    main()
