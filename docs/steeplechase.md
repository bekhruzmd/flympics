# Steeplechase

Hurdles, water pits and stamina combined: a 600 m race over 12 barriers, every third of
which is a water jump.

```powershell
.venv/Scripts/python -m experiments.run_steeplechase --controller rule
.venv/Scripts/python -m experiments.train_steeplechase
.venv/Scripts/python -m experiments.run_steeplechase --controller trained
```

## Rules (`events/steeplechase.py`)

Each step is 0.1 s. The actions are the Hurdles ones: `ACCELERATE`, `COAST`, `JUMP`.

- **Running:** as in Sprint and Hurdles, `v ← v + dt · (force − 0.16 v)`. Only
  `ACCELERATE` on the ground pushes.
- **Barriers** at 25, 75, …, 575 m. A jump lasts 0.8 s along a 1.0-high arc; the barrier
  is cleared when the arc is above 0.5 as the fly passes it. A hit keeps 30 % of the
  speed and costs 0.04 stamina. Take-off keeps 92 % of the speed.
- **Water jumps** are barriers 3, 6, 9 and 12 (125, 275, 425, 575 m). A 3 m pit lies just
  behind each one. Touching the ground in the pit (landing short, or running through after
  a hit) keeps 50 % of the speed and costs 0.04 stamina. Clearing it needs a fast,
  well-timed take-off (about 6 m/s or more).
- **Stamina** runs from 1 to 0. Accelerating drains `0.008 · dt · (v / 5)²`; coasting or
  flying recovers `0.02 · dt`; each jump costs 0.01. The push is
  `1.25 · (0.25 + 0.75 · stamina)`, so a fly that runs flat out from the gun is crawling
  by the end.

Score: race time when finished, otherwise `240 + metres left`. Lower is better. The
course is fixed; the seed changes nothing.

Observation: `distance_to_barrier` (capped at 40 m), `water_next` (0/1), `velocity`,
`stamina`, `distance_to_finish`.

## Baselines (`controllers/steeple_baselines.py`)

| Name | What it does |
|------|--------------|
| `rule` | Holds 5 m/s, speeds up to 6.2 m/s over the last 30 m before water, kicks over the last 80 m, times each jump from the physics. Knows the event's constants |
| `always_accelerate` | Never jumps: hits every barrier, splashes in every pit, burns out |
| `always_coast` | Never starts |
| `random` | Jumps with p 0.02, coasts with p 0.3 per step |
| `fly` | The untrained connectome fly |
| `trained` | The fly trained by `experiments.train_steeplechase` |

## The fly's adapters (`brain/steeplechase_io.py`)

- **Encoder:** `stimulus[sensory] = 0.45 + E @ [d_barrier / 40, water, v / 8, stamina,
  d_finish / 600]`, all clipped to [0, 1]. `E` has shape (24, 5).
- **Decoder:** extensor − flexor in three bands, as in Hurdles:
  `COAST` | `ACCELERATE` | `JUMP`.

## Training

See [evolution-training.md](evolution-training.md).
