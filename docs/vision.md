# Giving the fly real eyes (exploratory)

`experiments/vision_archery.py` shows an Archery end to FlyVis, a 45,669-neuron model of
the fruit fly's motion-vision system built from its real optic-lobe wiring and trained on
optic flow (Lappalainen et al., *Nature* 2024; MIT licence; https://github.com/TuragaLab/flyvis).
It is optional and lives outside the core project: it needs PyTorch and a one-time download
of the pretrained networks (about 9 MB), installed into a separate `.venv-vision`
(see `requirements-vision.txt`).

## What is real and what is modeled

- **Real:** the FlyVis visual network (optic-lobe connectivity from the fly connectome,
  pretrained by its authors). We use network `flow/0000/000` unchanged.
- **Modeled here:** what the fly sees. A 64 x 64 grey scene of grass specks that drift
  sideways with the recorded crosswind, plus the target, which shifts as the fly changes
  its aim. FlyVis turns that into the 721-column hexagonal eye input.
- **Not connected yet:** the eye's output does not feed the 135-neuron MANC leg circuit.
  The two come from different datasets (brain versus nerve cord), so any link between
  them will be a labelled modeling choice.

## Result: the real motion detectors read the crosswind

The readout is the horizontal motion signal of the direction-selective cells,
(T4a + T5a) - (T4b + T5b), averaged over all 721 columns.

- **Controlled wind blocks** (1 s each, grass only, no target, 0.1 px per 10 ms frame per
  unit of wind): the signal steps down steadily from strong leftward to strong rightward
  wind; correlation with wind r = -0.88, against at most |r| = 0.29 for time-shifted
  wind traces.
- **Too-fast drift** (1.5 px per frame per unit): direction still registers, but strength
  does not (response peaks at speed 1 and fades); fast motion is outside the range these
  cells prefer.
- **A real Archery end** (rule controller, seed 7): r = -0.68 between the signal and the
  recorded wind (`results/vision/archery_eye.png`). The target's own sideways shifts add
  spikes when the aim changes, so this number mixes two cues.

The sign is a convention (which way the "a" cells point), not a mistake.

## Training the fly to aim with its eye

`brain/eye.py` turns a calibration of FlyVis (`models/flyvis_wind_eye.json`, from
`python -m experiments.vision_archery --calibrate`: the T4/T5 signal for grass drifting at
33 wind strengths from -4 to 4) into the Archery fly's wind sense. The fly still feels its
own aim and bow draw. FlyVis itself is not run during training.

**Evolution strategy** (`python -m experiments.train_archery_eye`): the same method,
settings, seeds and neuron model (gain 3) as the wind-number Archery fly; connectome frozen.

| Archery fly (100 held-out winds) | Mean points / 60 | Bullseye rate |
|---|---:|---:|
| Untrained | 41.7 | 0.18 |
| Trained, handed the wind as a number | 55.0 | 0.53 |
| Trained, sensing the wind through the FlyVis eye | **55.6** | **0.62** |
| Same eye-fly, eye blindfolded (always reports no wind) | 41.9 | 0.18 |
| Same eye-fly, eye's left and right swapped | 24.3 | 0.08 |

Numbers are on the current Archery event, where the wind has its own random stream per
arrow so every controller faces identical winds (re-measured; training unchanged).
The blindfold and swap tests show the trained fly really uses the visual signal. Its
readout thresholds are again very close together (4.4e-6 and 1.4e-5), because the circuit
saturates at gain 3.

**Food-reward learning** (`python -m experiments.train_archery_food`): one fly learning
arrow by arrow from food (node perturbation, a critic per wind strength, MANC connection
strengths within +/-50%, signs kept). At gain 3 the circuit is saturated (motor neurons at
about 99.5% activity; the eye's effect on the readout is about 2e-6), so nothing can be
learned. At gain 1 (a modeling choice for this experiment only) 20,000 practice ends gave
28.4 -> 31.1 points on held-out winds with the fly's own noise, but the practice curve
stayed flat, so this is not convincing learning. Reported as a negative result.

## Sprint speed eye

The scene is drawn. A downward-facing camera moves over speckled ground at 21
speeds from 0 to 8 body lengths/s. The calibration uses the same horizontal T4/T5
readout as Archery, measured during the last half of each one-second block on
three textures. Mean activity rose at every sampled speed (Pearson r = 0.96).

![Horizontal T4/T5 activity versus ground speed, with error bars across textures](speed_eye.png)

The eye-to-leg link is modeled. The game maps one velocity unit to one body
length/s in the scene. The encoder interpolates the saved T4/T5 response,
subtracts the response at rest, and scales by the largest absolute change.
That activity and the distance to finish drive the leg circuit. Calibration
uses FlyVis in `.venv-vision`. Training and replay use the table in the normal
`.venv`.

```sh
FLYVIS_ROOT_DIR="$HOME/flyvis-data" .venv-vision/bin/python -m experiments.vision_speed
.venv/bin/python -m experiments.train_sprint_eye --imitation-generations 0 --threshold-sigma 0.00001
.venv/bin/python -m experiments.train_sprint_eye --compare
```

The evolution strategy ran for 60 generations from a threshold of `3e-5`, near
the circuit's saturated motor differences. The initial eye policy covered
10.16 m in 60 s. The saved policy finished the 100 m race.

| Sprint controller (seed 7) | Elapsed time | Distance |
|---|---:|---:|
| Untrained Sprint fly | 60.0 s (timed out) | 3.12 m |
| Existing trained Sprint fly | 18.7 s | 100 m |
| Trained speed-eye fly | 18.7 s | 100 m |
| Same speed-eye fly, blindfolded | 18.7 s | 100 m |

The eye did not improve Sprint performance. The trained fly accelerated on
every step both with vision and with the eye signal held at zero. Sprint has
fixed drag and no cost for accelerating, so that action sequence already gives
the fastest possible time.

## Next steps

1. Tune the MANC connection strengths of the eye-fly with the evolution strategy at
   +/-20%, +/-50% and 2x drift.
2. Revisit food-reward learning with a learned critic of aim error rather than points.
