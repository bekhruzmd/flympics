# Flympics — Mentor MVP

Flympics puts **connectome-derived fruit-fly neural controllers** into virtual Olympic events. This mentor MVP contains exactly one deliberately small event: a 100 m Sprint. Its job is a scientifically honest, runnable foundation for student developers—not a polished game or a claim to have simulated a fruit-fly brain.

## What this MVP proves

```text
Sprint environment → observation → artificial sensory encoder
       → MANC-connectome-derived sparse rate network → artificial motor decoder
       → ACCELERATE/COAST action → Sprint environment
```

Run `--controller fly` to execute that whole loop. The original `fly` controller is untuned; `trained` uses a motor threshold fitted to this sprint. A poor score remains a valid result. Each controller only sees observations and returns a legal action, so `Sprint` knows nothing about neural implementation details.

## What this is not

A connectome is a structural wiring diagram, not an executable brain. In this repository:

- **Biological connectivity data:** sparse edges and annotations come from the real, public MANC v1.2.1 adult Drosophila ventral-nerve-cord release.
- **Modeled neural dynamics:** `RateNetwork` is a leaky, bounded, unsigned rate model. Its gain, leak, normalization, and activation function are engineering assumptions, not fitted biology.
- **Artificial sensory encoding:** virtual distance and velocity stimulate selected annotated front-leg proprioceptive neurons. Flies do not access this virtual observation as written.
- **Artificial motor decoding:** relative activity of annotated front-leg extensor/flexor motor populations maps to a game action. It is not a claim that they issue an Olympic sprint command.

Call it a **connectome-derived neural model** or **connectome-constrained neural controller**, never a simulation of a whole fruit-fly brain or behavior.

## Why this data and model

The FlyWire FAFB whole-brain release is scientifically important, but its proofread connection table is 852 MB and raw synapses are 9.5 GB. A whole brain is counterproductive for this MVP. FlyVis is a valuable PyTorch connectome-constrained visual-system model, but is task-trained and vision-specific. FlyGym/NeuroMechFly is a serious MuJoCo embodied-fly framework; both are intentionally beyond scope.

MANC v1.2.1 is a better fit: it has ~23,650 VNC neurons, explicit sensory, intrinsic, descending, and motor annotations, a public ~83 MB weighted edge list, and CC BY 4.0 reuse. The vendored compact circuit was deterministically extracted from it: 24 front-leg proprioceptive sensory neurons, 96 direct VNC intrinsic targets, and up to eight unique front-leg tibia extensor/flexor motor neurons in each output population. It contains 135 neurons and 2,770 directed edges. See [data/README.md](data/README.md) and `data/manc_front_leg_v1/MANIFEST.txt` for exact hashes and selection.

The graph is SciPy CSR sparse storage. No network access is needed after checkout; raw source data are only needed to regenerate the compact artifact.

## Setup (macOS)

```bash
cd flympics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

To open the local website with the animated fly and circuit activity card:

```bash
python -m experiments.serve_view
```

Visit **http://localhost:8000**. Select a controller, play/pause, change playback
speed, or scrub the timeline. The page shows actual Python simulation output;
the browser does not simulate neural dynamics. The interactive brain widget uses an
illustrative fiber map with selectable regions. Brain regions are context only;
the four nerve-cord populations light up using recorded mean activity, synchronized
with playback and scrubbing. Traveling dots follow a readable subset of real
connections in the nerve-cord map and the selected neuron’s edges in the inspector.
Their brightness uses enhanced activity contrast; travel timing illustrates direction,
not measured spikes. Pausing freezes signals, scrubbing repositions them, and reduced
motion uses stationary markers. Baselines show no neural activity. The expandable
circuit inspector shows all 135 neurons using per-step recorded activity; select
a neuron to inspect its annotated role and connections. Its layout is schematic,
not anatomical. The task goal applies to the controller; individual neuron goals,
reward learning, and dopamine are not modeled. Runs are computed when the
server starts (seed 7 by default; change with `--seed`). Stop with Ctrl+C.
The server binds only to your own computer and requires no additional packages.

```bash
python -m experiments.run_sprint --controller random
python -m experiments.run_sprint --controller rule
python -m experiments.run_sprint --controller fly
```

To watch the fly on the track with a circuit activity card:

```bash
python -m experiments.run_sprint --controller fly --view
```

Press Space to pause/resume the Matplotlib window. The viewer plays the computed
run at the fixed simulation timestep; rendering cannot change the result.
The card shows mean modeled activity (0–1) for sensory neurons, VNC interneurons,
and extensor/flexor outputs. These are selected nerve-cord populations, not a map
of a whole fly brain. High activity is not necessarily useful behavior: both
motor populations can saturate while their difference commands COAST.

For a shareable 5x-speed animation and a final activity-card image:

```bash
python -m experiments.run_sprint --controller fly --gif
open results/sprint_fly_seed7.gif
```

Each run has deterministic `0.1 s` steps (not display-frame-rate steps), accepts `--seed`, and writes JSON, CSV, and a deliberately crude PNG plot to `results/`. Use `--no-plot` for headless runs. Logs include position, velocity, distance, action, and—for fly runs—summarized sensory and motor activity without dumping every neuron.

## Train the 100 m sprint

```bash
python -m experiments.train_sprint
python -m experiments.run_sprint --controller trained --no-plot
python -m experiments.serve_view
```

The website selects **Trained fly** by default; compare it with **Untrained fly**
and the rule/random baselines. Restart an existing server after training.

Training evaluates 34 motor thresholds against the actual sprint score and saves
`models/sprint.json`, including every trial and the original controller's result.
The checked-in policy completes 100 m in **18.7 s**, compared with the original
controller's **3.1 m at the 60 s timeout**. Retraining is deterministic.

This is task-specific calibration of the artificial motor decoder. The sensory
encoder, sparse connectivity, and neural dynamics stay fixed. Both motor
populations nearly saturate; lowering the threshold lets their small positive
difference sustain acceleration. Neural activity still determines each action.

With no fatigue or acceleration penalty, continuous acceleration is optimal in
this environment. The trained policy matches that reference; this is a small
first task, not evidence of general locomotion learning or biological training.
Different random seeds do not provide independent neural-policy validation here:
the sprint and fly controller are deterministic. Refit and reevaluate if physics,
connectivity, or dynamics change. Other events remain future work.

## Tests

```bash
python -m pytest -q
```

Tests do not need network access. They cover Sprint reset/steps/termination, legal actions, controller runs, encoder and neural dimensions, motor decoding, seeded reproducibility, and connectome loading.

## Architecture

```text
events/                 OlympicEvent interface; Sprint implementation
controllers/            Controller interface; random and rule baselines
brain/connectome.py     real-data-derived sparse-circuit loader
brain/dynamics.py       modeled neural dynamics (replaceable)
brain/sensory.py        artificial observation → stimulation adapter
brain/motor.py          artificial activity → game-action adapter
brain/controller.py     closed-loop fly controller
experiments/            reproducible CLI experiment and structured logs
visualization/          intentionally basic Matplotlib output
data/                   provenance and compact circuit artifact
```

## Extending Flympics

Add an event by implementing `OlympicEvent` (`reset`, `observe`, `step`, `is_finished`, `score`) under `events/`; do not modify `brain/`. Add a controller through `Controller.reset` and `Controller.act`. Replace `SensoryEncoder` or `MotorDecoder` independently to experiment with mappings. These seams are intentional student-owned work.

## Future ideas (not implemented)

Hurdles, archery, swimming, connectome ablation (“brain surgery”), evolutionary sensory/motor mappings, personalized fly training, student-trained-fly tournaments, replay, and neural-activity visualization are deliberately left for students.

## Sources and attribution

- [Male CNS/MANC download portal](https://male-cns.janelia.org/download/) — official data access and neuPrint guidance.
- [Marin et al., MANC motor-circuit study](https://elifesciences.org/articles/96084/figures) — MANC annotations and motor-circuit context.
- [Dorkenwald et al., adult whole-brain connectome](https://www.nature.com/articles/s41586-024-07558-y) and [FlyWire/Codex](https://codex.flywire.ai/api/download) — whole-brain reference and data context.
- [FlyVis](https://github.com/TuragaLab/flyvis) and [FlyGym / NeuroMechFly](https://flygym.readthedocs.io/latest/installation.html) — relevant frameworks considered but not reused.

MANC data are CC BY 4.0. Preserve source attribution and citation when redistributing this derived circuit.
