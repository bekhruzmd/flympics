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

## Setup (macOS / Linux)

```bash
cd flympics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Setup (Windows, PowerShell)

```powershell
cd flympics
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. Use Python 3.10 to 3.12. After activation, every `python -m ...` command below is the same on all systems. The `--view` window needs a desktop session; the browser viewer works anywhere.

## Run

To open the local website with the 3D fly:

```bash
python -m experiments.serve_view
```

Visit **http://localhost:8000**. Pick an event (Sprint, Hurdles or Swimming), then
pull a vial to load a controller's run. Press Play (or 4x replay) and drag the tape
ruler to scrub. Each vial fills with the distance its run has covered, so after
watching all four the rack compares them side by side. The page replays actual
Python simulation output; the browser does not simulate anything. The fly's body is
the NeuroMechFly micro-CT scan (FlyGym, Apache-2.0); its six-leg gait and wingbeat
are hand-made animation, not output of the circuit. Distances are read as
1 sim metre = 1 body length (about 2.5 mm; see `docs/fly-speed-data.md`). Fly mode
only changes the picture: flight is not part of the race physics. Real fruit flies
cannot swim, so Swimming is a game model. Runs are computed when the server starts
(seed 7 by default; change with `--seed`). Stop with Ctrl+C.
The server binds only to your own computer and requires no additional packages.

To host the page without Python (for example on Vercel), write the recorded runs to a
static file and serve `visualization/web/` as a plain static site:

```bash
python -m experiments.export_view
```

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

## Circuit ablations

```bash
python -m experiments.ablate_sprint
python -m experiments.ablate_sprint --k 20 --seeds 0 1 2 3 4
```

This compares lesioned circuits with the unchanged control, under both the
untrained and trained motor thresholds. Conditions remove all sensory,
interneuron, extensor, or flexor neurons; the top-k interneurons by summed
input weight onto extensors or flexors; and, as a size-matched control, k
random interneurons per seed (50 seeds by default). Results go to
`results/ablation_k{k}.json/.csv`. Hypotheses and metric are stated in
`experiments/ablate_sprint.py`.

The extensor-only and flexor-only lesions are expected by construction: the
decoder reads extensor minus flexor, so silencing one side fixes the sign of the
difference. They check the wiring of the experiment, not the circuit. The
informative rows are the interneuron lesions.

Results with k = 10 and seeds 0–49 (the defaults):

- No sensory neurons: no activity enters the circuit; the fly never accelerates.
- Top-10 flexor drivers removed: even the untrained threshold finishes (18.7 s).
- All 96 interneurons removed: both thresholds finish. The few direct
  sensory-to-motor edges favor extensors, so in the intact model the
  interneurons are what keep extensor minus flexor near zero.
- Top-10 extensor drivers removed: the trained policy fails (mean difference
  −0.353). This is only partly specific: 21 of 50 random 10-interneuron lesions
  also broke the trained policy, and 6 of 50 lowered the difference at least as
  much. 38 of the 50 random lesions included at least one top driver; the six
  that included three lowered the difference most on average (−0.131, against
  +0.005 for lesions with none).

`brain.ablation.ablate` returns a lesioned *copy* of the circuit: removed neurons
lose every incoming and outgoing edge, and their modeled rate is clamped to 0,
even if the artificial encoder stimulates them. Selected edges can also be cut.
Surviving weights are not renormalized, and the vendored circuit is unchanged.
The decoder still averages over all annotated motor neurons, so a silenced motor
neuron contributes 0 activity.

Ablation is a modeling intervention on a fixed rate model, not a prediction of a
lesioned fly. The sprint and fly controller are deterministic; seeds only choose
the random lesions. The trained threshold is tiny (about 1e-5), so small activity
changes can flip actions: compare targeted lesions against the random control.

## More events

Each event has its own guide: [Hurdles](docs/hurdles.md), [Swimming](docs/swimming.md),
[Archery](docs/archery.md) and [Steeplechase](docs/steeplechase.md).

- **Archery:** six arrows in a gusty crosswind. Aim LEFT/RIGHT against the wind, then
  RELEASE once the bow is drawn. Score is total points (0–60, higher is better).
- **Steeplechase:** 600 m over 12 barriers, four of them water jumps with a pit behind,
  with a stamina budget that punishes running flat out.

```bash
python -m experiments.run_archery --controller rule
python -m experiments.run_steeplechase --controller rule
python -m experiments.train_archery
python -m experiments.train_steeplechase
python -m experiments.run_archery --controller trained
python -m experiments.run_steeplechase --controller trained
```

Both flies are trained with an evolution strategy (OpenAI-ES) on the sensory encoder
weights and the decoder band thresholds only; the connectome and dynamics stay frozen.
Outcome-only training gets stuck, so it first imitates the rule baseline, then optimises
the event outcome. See [docs/evolution-training.md](docs/evolution-training.md).

Checked-in results (`models/archery.json`, `models/steeplechase.json`):

| Archery, 100 held-out winds | Mean points / 60 | 10s |
|---|---:|---:|
| Rule (knows the drift constant) | 59.92 | 99 % |
| **Trained fly** | **55.21** | 56 % |
| Always release at the centre | 42.79 | 19 % |
| Untrained fly | 41.95 | 19 % |

| Steeplechase, fixed course | Time | Barriers cleared | Splashes |
|---|---:|---:|---:|
| Rule | 121.3 s | 12 | 0 |
| **Trained fly** | **141.0 s** | 12 | 4 |
| Random | 196.5 s | 0 | 4 |
| Always accelerate | 238.4 s | 0 | 4 |
| Untrained fly | did not start | 0 | 0 |

The trained archery fly steers against the wind and only then releases. The trained
steeplechase fly clears every barrier at a safe, steady pace, but never builds enough
speed for the water jumps and finishes with most of its stamina unused.

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
brain/ablation.py       experiment-only lesioned circuit copies
experiments/            reproducible CLI experiment and structured logs
visualization/          intentionally basic Matplotlib output
data/                   provenance and compact circuit artifact
```

## Extending Flympics

Add an event by implementing `OlympicEvent` (`reset`, `observe`, `step`, `is_finished`, `score`) under `events/`; do not modify `brain/`. Add a controller through `Controller.reset` and `Controller.act`. Replace `SensoryEncoder` or `MotorDecoder` independently to experiment with mappings. These seams are intentional student-owned work.

## Future ideas (not implemented)

Personalized fly training, student-trained-fly tournaments, replay, and neural-activity visualization are deliberately left for students.

## Sources and attribution

- [Male CNS/MANC download portal](https://male-cns.janelia.org/download/) — official data access and neuPrint guidance.
- [Marin et al., MANC motor-circuit study](https://elifesciences.org/articles/96084/figures) — MANC annotations and motor-circuit context.
- [Dorkenwald et al., adult whole-brain connectome](https://www.nature.com/articles/s41586-024-07558-y) and [FlyWire/Codex](https://codex.flywire.ai/api/download) — whole-brain reference and data context.
- [FlyVis](https://github.com/TuragaLab/flyvis) and [FlyGym / NeuroMechFly](https://flygym.readthedocs.io/latest/installation.html) — relevant frameworks considered but not reused.

MANC data are CC BY 4.0. Preserve source attribution and citation when redistributing this derived circuit.
