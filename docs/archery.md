# Archery

Adjust the aim against a gusty crosswind, then release. One end is six arrows at a
10-ring target. Only horizontal aim is modelled.

```powershell
.venv/Scripts/python -m experiments.run_archery --controller rule
.venv/Scripts/python -m experiments.train_archery
.venv/Scripts/python -m experiments.run_archery --controller trained
```

## Rules (`events/archery.py`)

Each step is 0.1 s. Every arrow starts aimed at the centre with the bow undrawn.

| Action | Effect |
|--------|--------|
| `LEFT` / `RIGHT` | Move the aim 0.04 m (`aim_rate · dt`) at the target face |
| `HOLD` | Keep the aim |
| `RELEASE` | Shoot. Ignored until the bow is fully drawn (`draw_time` 1 s) |

The arrow lands at `aim + drift_per_wind · wind + jitter`. The wind is a per-arrow base
wind (sd 2 m/s) plus a gust that keeps changing (an Ornstein–Uhlenbeck process with
correlation time 1.5 s), clipped to ±4 m/s. Each m/s of wind pushes the arrow 0.12 m, so
the aim that cancels the wind keeps moving. The release jitter has sd 0.02 m.

Holding costs: after `steady_time` (4 s) the aim starts to wander, and an arrow not
released within `shot_clock` (10 s) scores 0.

Points are `10 − floor(|landing| / 0.061)`, at least 0. The **score is total points**
(0–60), so higher is better, unlike the race events. The seed fixes the wind, gusts,
sway and jitter, so the same seed always gives the same end.

Observation: `aim_offset`, `wind`, `draw` (0–1), `hold_time`, `arrows_left`.

## Baselines (`controllers/archery_baselines.py`)

| Name | What it does |
|------|--------------|
| `rule` | Aims at `−drift_per_wind · wind`, releases within 0.025 m of it. Knows the drift constant |
| `always_release` | Shoots at the centre as soon as the bow is drawn; ignores the wind |
| `always_hold` | Never shoots: every arrow times out |
| `random` | Random LEFT/RIGHT/HOLD, releases with probability 0.1 per step |
| `fly` | The untrained connectome fly |
| `trained` | The fly trained by `experiments.train_archery` |

## The fly's adapters (`brain/archery_io.py`)

- **Encoder:** `stimulus[sensory] = 0.45 + E @ [aim / 0.5, wind / 4, draw]`, aim and wind
  signed and clipped to ±1. `E` has shape (24, 3).
- **Decoder:** the Sprint extensor − flexor readout in three bands:
  `LEFT` (lowest) | `RELEASE` | `RIGHT` (highest). Before the bow is drawn `RELEASE` acts
  as a hold.

The ordering means a monotone mapping is enough: aim too far right → push the difference
down (LEFT); aim too far left → push it up (RIGHT); in between → release.

## Training

See [evolution-training.md](evolution-training.md).
