# Evolution-strategy training (Archery, Steeplechase)

```powershell
.venv/Scripts/python -m experiments.train_archery
.venv/Scripts/python -m experiments.train_steeplechase
```

Each writes `models/<event>.json` (policy and results) and `models/<event>_encoder.csv`
(trained encoder weights), plus `results/<event>_training/history.csv/.png`.

## What learns and what is frozen

| Part | Status |
|------|--------|
| Encoder weights `E` (24 × 3 for Archery, 24 × 5 for Steeplechase) | **trained** |
| Decoder band thresholds on extensor − flexor (two per event) | **trained** |
| Tonic drive 0.45 and input normalisation | frozen |
| MANC connectome `W`, gain 3.0, leak 0.35, 2 substeps, motor populations | frozen |

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

## Noise robustness

The trained thresholds sit within ~1e-5 of zero. How the flies hold up under neural
noise is measured in [noise-robustness.md](noise-robustness.md).

## Caveats

This is task-specific tuning of artificial input and output mappings around a fixed,
connectome-derived rate model. It is not a claim about how flies learn, and the per-neuron
encoder weights are not biological synapses.
