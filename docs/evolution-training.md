# Evolution-strategy training (Archery, Steeplechase, Hurdles)

```powershell
.venv/Scripts/python -m experiments.train_archery
.venv/Scripts/python -m experiments.train_steeplechase
.venv/Scripts/python -m experiments.train_hurdles_es
```

Each writes `models/<event>.json` (policy and results) and `models/<event>_encoder.csv`
(trained encoder weights), plus `results/<event>_training/history.csv/.png`.

## What learns and what is frozen

| Part | Status |
|------|--------|
| Encoder weights `E` (24 × 3 for Archery, 24 × 5 for Steeplechase, 24 × 3 for Hurdles) | **trained** |
| Decoder band thresholds on extensor − flexor (two per event) | **trained** |
| Tonic drive 0.45 and input normalisation | frozen |
| MANC connectome `W`, gain 3.0 (or 1.0 modeling choice in Hurdles), leak 0.35, 2 substeps, motor populations | frozen |

Training hashes the connectome before and after and stops with an error if it changed.

## Why an evolution strategy

The Always-Accelerate experiment back-propagates through the rate network because its
teacher gives a target action at every step. Here the score (points, race time) depends
on discrete actions, jumps and ring boundaries, so it has no useful gradient. A
black-box method fits.

We use **OpenAI-ES** (Salimans et al., 2017):

1. Sample `pairs` Gaussian noise vectors `ε`; evaluate `θ + σε` and `θ − σε`
   (antithetic sampling halves the variance), plus `θ` itself.
2. Replace each fitness by its centred rank in [−0.5, 0.5]. Ranks ignore the scale of the
   fitness and are robust to plateaus.
3. Estimate the gradient as `Σ (rank(θ+σε) − rank(θ−σε)) ε / (2 · pairs · σ)` and take an
   Adam step.
4. Keep the best parameter vector ever evaluated; that is what gets saved.

### Outcome alone is not enough

Run on the event outcome alone, the ES gets stuck in the first local optimum it finds:

- **Archery:** the untrained fly releases the moment the bow is drawn, ignoring the
  wind (about 42 points). Steering needs the right sign on the aim error *and* a release
  band centred on zero at the same time, and no single perturbation gets both. 150
  generations reached only 44.7 points on held-out winds.
- **Steeplechase:** the untrained fly never moves. The first improvement any
  perturbation finds is "always accelerate" (238.4 s, every barrier hit), and 300
  generations never left it.

The circuit is not the limit: random encoder weights give motor differences that
correlate up to 0.75 with the archery aim error.

### Two stages

**Stage 1, imitation with teacher forcing.** Record trajectories from the rule baseline:
one clean run per training seed, plus two runs per seed where 30 % of actions are random,
so the data also shows mistakes and recoveries. Label every visited state with the
rule's band. For a candidate encoder `E`, replay just the recurrent network along those
fixed observation sequences (`brain.evolution.teacher_forced_differences`). No event
stepping is needed, so a generation takes seconds. The two thresholds are then **fitted
exactly** (`fit_band_thresholds`: sort the motor differences and pick the cut points
with the best agreement), so the ES only has to search `E`. The score is
class-balanced agreement with the rule; balancing keeps the rare JUMP steps from being
ignored. The stage starts from the best of the initial weights and 48 random encoders
(σ 0.5).

**Stage 2, outcome.** Starting from the imitation result and its fitted thresholds, run
the ES on the real event fitness. Every candidate races closed-loop on its own
decisions; all candidates (× training seeds) are simulated together by
`brain.evolution.BatchedRateNetwork`, a dense batched copy of `RateNetwork`. Weights use
σ = 0.3; thresholds use σ = 0.02 and a proportionally smaller step. Because the best
vector ever evaluated is kept and the stage-1 result is the first one evaluated, this
stage never ends below stage 1 on the training fitness.

The rule baseline knows the event's physics constants; the fly does not. Stage 1 copies
the rule's behaviour, and stage 2 is free to beat it.

Defaults: 24 pairs (population 49). Archery: 150 imitation + 60 outcome generations.
Steeplechase: 150 + 200. ES seed 0.

`tests/test_band_brains.py` checks that both the batched rollout and the teacher-forced
replay reproduce the ordinary `FlyBrainController`. Every number in the policy JSON is
measured again with the ordinary controller after training.

## Stage-2 fitness

- **Archery:** mean over 12 training winds (seeds 0–11) of the end's total
  `10 − |landing| / ring_width` per arrow (points without the rounding, so small
  improvements count), with −2 for a timed-out arrow. Reported results use real points on
  100 held-out winds (seeds 1000–1099).
- **Steeplechase:** minus the race score. The course is fixed, so this is one race per
  candidate. There is no held-out course: the result says the circuit can be tuned to this
  race, not that it learned steeplechase in general.
- **Hurdles:** minus the race time (higher is better) over 12 training layouts (seeds 0–11).
  Reported results evaluate on the benchmark course and 50 held-out seeds (1000–1049).
  At default gain 3.0, recurrent saturation drives motor populations near 0.995, so ES
  converges to always accelerating (58.35 s, 0 jumps). As a modeling choice, `--gain 1.0`
  desaturates the recurrence, allowing sensory input to modulate motor differences; the
  trained fly takes an initial jump (59.25 s on benchmark, 1 jump, 2 coasts; 57.52 s mean,
  1.0 jumps on held-out seeds) while still hitting the hurdles due to narrow clearance windows.

## Noise robustness

The trained thresholds sit within ~1e-5 of zero. How the flies hold up under neural
noise is measured in [noise-robustness.md](noise-robustness.md).

## Caveats

This is task-specific tuning of artificial input and output mappings around a fixed,
connectome-derived rate model. It is not a claim about how flies learn, and the per-neuron
encoder weights are not biological synapses.

## Connection tuning at gain 1.5 (Archery and Steeplechase)

The same method as the connection-tuned Hurdles fly (`docs/hurdles.md`): besides the
encoder and thresholds, training scales every existing MANC connection's strength by a
multiplier kept within x0.5–2. The wiring map, every sign and the MANC data stay fixed;
gain 1.5 is a modeling choice from `docs/saturation.md`. Each event was trained twice with
the same budget (150 imitation generations, then 250 with 64 antithetic pairs, ES seed 0):
once with the connections frozen and once with them tuned.

```sh
python -m experiments.train_archery --gain 1.5 --pairs 64 --generations 250 --policy models/archery_gain1.5_long.json
python -m experiments.train_archery --gain 1.5 --pairs 64 --generations 250 --drift 2x --policy models/archery_connections_gain1.5_drift2x_long.json
python -m experiments.train_steeplechase --gain 1.5 --pairs 64 --generations 250 --policy models/steeplechase_gain1.5_long.json
python -m experiments.train_steeplechase --gain 1.5 --pairs 64 --generations 250 --drift 2x --policy models/steeplechase_connections_gain1.5_drift2x_long.json
```

**Archery** (held-out winds 1000–1099, points per end out of 60):

| Fly | Mean points | Tens | Connections changed |
|---|---:|---:|---|
| Rule baseline | 59.92 | 99% | – |
| Gain 1.5, connections within x0.5–2 | **58.38** | 76% | mean 19.4%, max 89.9% |
| Gain 1.5, encoder only | 57.78 | 70% | none |
| FlyVis eye fly, gain 3 (`archery_eye.json`) | 55.57 | 62% | none |
| Trained fly, gain 3 (`archery.json`) | 54.95 | 53% | none |

End by end on the same winds, tuning the connections adds 0.60 points (95% interval
±0.28; better on 52% of ends, worse on 24%, tied on 24%).

**Steeplechase** (the one fixed course; score in seconds, lower is better):

| Fly | Score | Clears | Hits | Splashes |
|---|---:|---:|---:|---:|
| Rule baseline | 121.3 | | | |
| Gain 1.5, connections within x0.5–2 | **137.2** | 12 | 0 | 4 |
| Gain 1.5, encoder only | 139.4 | 12 | 0 | 4 |
| Trained fly, gain 3 (`steeplechase.json`) | 141.0 | | | |

The Steeplechase course is fixed, so these scores say the circuit can be tuned to this one
race, not that it learned steeplechase in general, and there are no held-out courses to
test the 2.2 s gap on.

Limits, as for Hurdles: one training run per setting, so the training's own randomness is
not measured, and the tuned strengths are a model fitted to a game, not a claim about real
synapses. The viewer races both as "Connection-tuned fly".

## Connection tuning at gain 1.5 (400 m and Swimming)

Same method and budget (250 generations, 64 antithetic pairs, ES seed 0), but with no
imitation stage: the stage fits three bands, and these events have two actions. Both races
are fixed, so there are no held-out courses.

```sh
python -m experiments.train_pacing_es --gain 1.5 --pairs 64 --generations 250 --threshold-sigma 0.02 --policy models/pacing_gain1.5_long.json
python -m experiments.train_pacing_es --gain 1.5 --pairs 64 --generations 250 --threshold-sigma 0.02 --drift 2x --policy models/pacing_connections_gain1.5_drift2x_long.json
python -m experiments.train_swimming_es --gain 1.5 --pairs 64 --generations 250 --threshold-sigma 0.02 --policy models/swimming_gain1.5_long.json
python -m experiments.train_swimming_es --gain 1.5 --pairs 64 --generations 250 --threshold-sigma 0.02 --drift 2x --policy models/swimming_connections_gain1.5_drift2x_long.json
```

**400 m** (seed 7; seconds, lower is better): rule 123.0, random 129.7, connections within
x0.5–2 **183.1**, encoder only 198.1, earlier trained fly 198.1. Encoder-only training found
nothing better than its start; tuning the connections (mean change 8.9%, max 38.6%) cut
15 s. It still trails both baselines. The viewer races it as "Connection-tuned fly".

**Swimming** (100 m, 120 s limit): neither trained fly finishes. Encoder only learned a
rhythm (15 strokes, 38.5 m, from 1 stroke and 17.3 m untrained); with the connections tuned
it did worse (0 strokes, 21.2 m). Without an imitation stage the search starts from a fly
that stalls after one stroke and gets little signal. Not shown in the viewer. A two-action
imitation stage is the obvious next step.
