# Noise robustness of the evolution-trained flies

The trained Archery and Steeplechase flies decide by cutting extensor − flexor at two
thresholds that sit within about 1e-5 of zero (`models/archery.json`:
8.8e-6 and 1.26e-5, a gap of 3.9e-6; `models/steeplechase.json`: −4.0e-5 and −1.1e-5).
Real neurons are noisy. Does a little noise break these flies?

```powershell
.venv/Scripts/python -m experiments.noise_robustness
```

writes `results/noise_robustness/summary.json`, `scores.csv` and `score_vs_noise.png`
(about 6 minutes). Every run is seeded, so the numbers below reproduce exactly.

## The noise model

`RateNetwork(circuit, noise=σ)` adds independent Gaussian noise N(0, σ²) to every
neuron's drive, before the tanh, on every substep:

```
drive  = gain · W @ state + stimulus + N(0, σ²)
state += leak · (tanh(drive) − state)
```

- Off by default (`noise=0`). With noise off no random numbers are drawn, so the update is
  bit-for-bit the old one (`tests/test_noise.py` checks this against a verbatim copy of the
  old update, and checks that both trained flies still give exactly the scores saved in
  their policy files).
- `FlyBrainController.reset(seed)` reseeds the noise, so a run with a given seed is
  reproducible (also tested). `FlyBrainController.from_policy / band_event / archery /
  steeplechase` take `noise=`.
- `BatchedRateNetwork`, `batched_rollout` and `teacher_forced_differences` take the same
  option, ready for training with noise (part 2).

The noise is on the **drive**, not on the readout. That turns out to matter (below).

## 1. Toy check: a bare threshold

Decision "go right if x > 1e-5" with x = 2e-5 (a margin of 1e-5), noise added to x,
100 000 draws, `np.random.default_rng(0)`:

| noise σ | 0 | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|---|
| decisions flipped | 0 % | 0 % | 15.9 % | 46.1 % | 49.8 % |

If the flies behaved like this, they would be a coin flip from σ = 1e-4.

## 2. Event scores

Archery: mean points per end (6 arrows, max 60) on the held-out winds 1000–1099; the
seed sets both the wind and the noise. Steeplechase: the course is fixed, so seeds
1000–1099 only change the noise; mean race score in seconds (lower is better) and
races finished out of 100. The rule baselines are non-neural, so noise cannot touch them.

| noise σ | 0 | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|---|---|---|
| Archery trained | **54.95** | 54.95 | 54.88 | 54.98 | 54.30 | 46.31 | 17.47 |
| Archery untrained | 41.42 | 41.42 | 41.42 | 41.42 | 41.43 | 41.45 | 41.39 |
| Archery rule | 59.92 | – | – | – | – | – | – |
| Steeplechase trained (s) | **141.0** | 141.0 | 140.9 | 141.0 | 142.7 | 651.5 | 825.8 |
| Steeplechase trained, finished | 100 | 100 | 100 | 100 | 100 | 0 | 0 |
| Steeplechase untrained (s) | 836.9 | 836.9 | 836.9 | 836.9 | 836.9 | 836.9 | 836.8 |
| Steeplechase rule (s) | 121.3 | – | – | – | – | – | – |

Archery at σ = 1e-1 also times out 328 of 600 arrows.

![score vs noise](../results/noise_robustness/score_vs_noise.png)

(`results/` is not tracked; the script regenerates the plot.)

**Finding: the flies do not break at tiny noise.** Both trained flies stay within 0.1
point / 0.1 s of their noiseless scores up to σ = 1e-4, lose a little at 1e-3 (Archery
−0.65 points, Steeplechase +1.7 s), and collapse between 1e-3 and 1e-2. At σ = 1e-2 the
trained Archery fly is still better than the untrained one (46.3 vs 41.5); the trained steeplechaser no longer finishes. At 1e-1
the trained Archery fly is far *worse* than the untrained one.

## 3. Why the toy check is too pessimistic

Two things stand between the noise and the score.

**The circuit shrinks the noise before the readout.** Replaying the noiseless run's
observations (seed 1000) through a noisy network, 20 noise seeds, the noise on
extensor − flexor is a fixed fraction of σ:

| fly | readout noise std / σ | readout noise at σ = 1e-3 |
|---|---|---|
| Archery trained | 0.101 | 1.0e-4 |
| Archery untrained | 0.083 | 8.3e-5 |
| Steeplechase trained | 0.021 | 2.1e-5 |
| Steeplechase untrained | 0.011 | 1.1e-5 |

Saturated neurons pass very little of a drive perturbation (tanh′ ≈ 0), the leak
low-passes it over substeps, and the readout averages each motor population. So the
saturation that forces the tiny thresholds in the first place also protects them: noise
of 1e-4 on the drive is only ~1e-5 (Archery) or ~2e-6 (Steeplechase) at the readout.

**Closed loop forgives flipped decisions.** Even after that shrinkage, σ = 1e-3 flips a
lot of single decisions in the same open-loop replay — but the score barely moves:

| noise σ | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|---|---|
| Archery trained, decisions flipped | 0 % | 0.5 % | 3.4 % | 30 % | 54 % |
| Steeplechase trained, decisions flipped | 0.03 % | 0.4 % | 4.6 % | 26 % | 54 % |

At σ = 1e-3, nearly a third of Archery decisions differ, but a stray LEFT/RIGHT moves the aim
by one small step that the next decisions undo; Steeplechase mostly dithers between
COAST and ACCELERATE, which averages out. Only once roughly half the decisions are random
(σ = 1e-2) do the scores collapse.

(Steeplechase flips a few decisions even at σ = 1e-6, where the readout noise is only
~2e-8, so a few of its steps sit that close to a threshold. They do not change the race.)

## What didn't work, and caveats

- **The untrained flies look "robust", but only because they do nothing useful.** The
  untrained Archery fly's motor difference sits far from its ±0.015 thresholds, so it
  always releases straight away; the untrained steeplechaser never starts. Flat lines
  here are not a virtue.
- **The threshold gap is not on the same scale as σ.** My first plot drew the gap between
  the thresholds as a line on the noise axis; that compares readout units with drive
  units, so I removed it. The fair comparison is with the readout noise in section 3: at
  σ = 1e-4 the Archery readout noise (1.0e-5) is already larger than the threshold gap
  (3.9e-6), yet only 3.4 % of decisions flip, because most steps are far from both.
- **Only one kind of noise.** Independent, additive, per neuron and substep, on the
  drive. Not tested: noise on synapses or weights, multiplicative noise, noise on the
  readout itself, or noisy observations.
- **Steeplechase has one fixed course.** Its 100 seeds only vary the noise, so it says
  nothing about other courses.
- The readout diagnostic is open loop on a single run (seed 1000) and 20 noise seeds.

## Next: training with noise (part 2)

Not done yet. Plan: add a `--noise` option to `experiments/es_training.py` that passes σ
to both the teacher-forced imitation stage and the batched rollouts (with a fresh noise
seed every generation), retrain both flies at σ = 1e-3 or 1e-2, then rerun this script
with `--archery-policy noise-trained=...` / `--steeplechase-policy noise-trained=...` and
compare the score curves and the threshold gaps.
