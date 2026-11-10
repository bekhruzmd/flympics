## What changed

Adds the **Always-Accelerate** experiment. The fly's sensory encoder is trained to always choose ACCELERATE in Sprint v1. Only the encoder learns; everything else in the brain stays frozen.

- **`brain/sensory.py`**: adds `LearnedSensoryEncoder`, which gives each of the 24 sensory neurons its own distance weight and velocity weight (a 24×2 matrix `E`). Also adds `load_encoder_weights` / `save_encoder_weights` for a validated CSV format.
- **`brain/controller.py`**:
  - `FlyBrainController` takes a new optional `encoder_weights=` argument.
  - A new `FlyBrainController.always_accelerate()` loads the trained weights.
  - When it merged with upstream, this combined cleanly with the new `circuit=` argument from the ablation PR, so both can be used together.
- **`experiments/train_always_accelerate.py`**: on-policy training. Each epoch:
  1. The fly runs a full sprint with its current weights.
  2. Every step is scored with a squared hinge loss (extensor − flexor must clear the threshold by a margin).
  3. The gradient is backpropagated through time through the frozen network to `E`.
  4. Adam updates `E`, with gradient clipping.
- **`models/sprint_encoder_init.csv`**: the untrained starting weights, which reproduce the original fly exactly. **`models/sprint_encoder.csv`**: the trained weights.
- **`experiments/run_sprint.py`** and **`visualization/sprint_view.py`**: new `--controller accelerate` option.
- **`docs/training-plan.md`** (how training works, results, known limits) and **`docs/brain-uml.md`** (class, sequence and activity diagrams).
- **`tests/test_always_accelerate.py`**: 16 tests (see Checklist).

## Data vs. modeled assumptions

**From MANC data (unchanged):** the 135-neuron front-leg circuit, its 2770 edges in `W`, and which neurons are sensory, extensor and flexor.

**Modeling choices:**
- The task-to-neuron encoding: tonic drive 0.45, inputs scaled as `distance/100` and `velocity/8`, and one learned linear weight per sensory neuron per input.
- The teacher signal. In Sprint v1, force is constant and drag only slows the fly, so ACCELERATE is the best action in every state.
- The loss target: a 0.035 margin above the decoder's 0.015 threshold.

**Frozen and checked:** `W`, gain 3.0, leak 0.35, the 2 substeps, the decoder threshold 0.015 and the motor indices. `fingerprint()` hashes these before and after training and raises an error if anything changed. Connectivity, matrix orientation and normalization are not touched.

## Hypothesis and metric (for experiments)

**Hypothesis:** training only the 48 sensory input weights, with the connectome frozen, is enough for the fly to accelerate on every step.

**Metrics:** the fraction of steps where it accelerates, plus distance and finish time, compared with the always-ACCELERATE reference (100 m in 18.7 s, 187 steps).

**Results (default settings):**

| epoch | loss | distance | time | ACCELERATE |
|---:|---:|---:|---:|---:|
| 1 (untrained) | 2.48e-3 | 3.1 m | 60 s (timeout) | 0.7% |
| 10 | 1.11e-3 | 100 m | 24.9 s | 55% |
| 14 | 2.8e-6 | 100 m | 18.7 s | 100% |
| 300 | 1.75e-6 | 100 m | 18.7 s | 100% |

The trained fly matches the reference exactly.

**Known limits:**
- The loss never reaches 0. The network starts from all zeros, so the first step's difference stops at about 0.032. That is above the threshold, so the fly still accelerates, but it is below the 0.05 target. As a result, training always uses every epoch.
- Three sensory neurons have no outgoing synapses, so their weights never change.
- **Seeds:** Sprint v1 has no randomness (`reset()` ignores the seed), so every run is identical. Retraining reproduces `models/sprint_encoder.csv` byte for byte.

## Checklist
- [x] `python -m pytest -q` passes: 45 tests after merging upstream `main`, which includes the 16 new ones
- [x] Runs are seeded and reproducible. The sprint is deterministic, and the retrained weights are byte-identical to the committed file.
- [x] No hardcoded actions or edited scores. The behavior comes only from gradient training of `E`, and the decoder threshold is unchanged.
- [x] Docs updated: `docs/training-plan.md` and `docs/brain-uml.md`
