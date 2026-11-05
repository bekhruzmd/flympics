# 100 m Swimming: implementation plan

The third Flympics event. The fly swims 100 m in a 50 m pool, with one turn, and
chooses between two actions on every step: PULL and RECOVER. Fast swimming needs a
steady rhythm of full strokes, plus a glide after the dive and after the turn.

The event follows the `OlympicEvent` interface in [events/base.py](../events/base.py)
and the conventions of [events/sprint.py](../events/sprint.py) and
[events/hurdles.py](../events/hurdles.py): 1-D track, fixed `dt`, deterministic, lower
score is better. Like the other events, it is a game model, not fly biomechanics.

## Scope

**In scope:**

- The event itself.
- A base rule controller and simple reference strategies that show the event isn't
  trivially solved.
- A results file for every run, written to `results/`.
- A simple encoder, decoder and set of starting encoder weights (a CSV) for the fly
  brain, built the same way as the Hurdles ones.

**Out of scope:**

- **Training.** The fly brain is not trained in this project. The untrained fly only
  has to swim a full race without errors.
- **The connectome and the network dynamics.** `RateNetwork` stays unchanged.
- **Sprint and Hurdles.** Their behaviour and tests stay unchanged.

## Why this event

Sprint rewards doing one thing all the time. Hurdles rewards timing a rare action.
Swimming tests **rhythm**: the fly has to switch between two actions on a steady
cycle.

- **Always PULL** makes one stroke and then stalls with its arm at the end of the
  stroke.
- **Always RECOVER** glides off the dive and slowly stops.
- **Switching every step** never builds grip, so its pulls barely push.
- **Random presses** mix short strokes with costly recoveries.

## Rules

The race is **fixed**: 100 m in a 50 m pool, one turn. There are no layouts and the
seed does not change anything about the race. It only seeds random controllers.

The state has three parts: position, speed `v`, and arm angle `a` from 0 (forward,
ready to pull) to 1 (stroke finished).

| Rule | Value | Reason |
|------|-------|--------|
| Distance, pool | 100 m in a 50 m pool | One turn |
| `dt` | 0.05 s | Same as Hurdles |
| `max_time` | 120 s | A good race takes about 44 s |
| PULL, `a < 1` | `a` rises over `pull_time` = 0.6 s (12 steps). Thrust `thrust · grip · max(0, 1 − v/hand_speed)`, with `thrust` = 5.0 and `hand_speed` = 4.0 m/s | A stroke pushes less when the fly is already fast |
| Grip | `grip = min(1, pull_steps · dt / grip_time)`, `grip_time` = 0.25 s, so full after 5 steps. `pull_steps` counts the PULL steps in a row and resets on any RECOVER | Long strokes beat quick taps |
| PULL, `a = 1` | No thrust | Always-PULL stalls |
| RECOVER, `a > 0` | `a` falls over `recover_time` = 0.4 s (8 steps). Extra drag `recovery_drag · v`, `recovery_drag` = 0.5 | Recovery has a cost |
| RECOVER, `a = 0` | Glide: body drag drops from `swim_drag` = 0.08 to `glide_drag` = 0.02 | Gliding is worth it at high speed |
| Drag | `c·v² + linear_drag·v`, `linear_drag` = 0.15 | A glide fades out completely |
| Update | `v = max(0, v + dt · (thrust − drag))`, then `position += dt · v` | Semi-implicit Euler, as in Sprint |
| Start | Dive entry at `dive_speed` = 4.0 m/s | |
| Turn at 50 m | `v = turn_keep · v + push_off`, `turn_keep` = 0.2, `push_off` = 3.5 m/s | A second glide |
| Score | Finish time if the fly finishes; otherwise `max_time + metres left` | Same as Sprint and Hurdles |

The arm moves in whole steps (12 to pull through, 8 to recover), counted with
integers, so `a` is exactly 0 or 1 at the ends of its range.

### Observation

```python
{
    "distance_to_wall": next_wall - position,   # the turn wall, then the finish wall
    "velocity":         v,
    "stroke_phase":     a,                      # 0 = arm forward, 1 = stroke finished
}
```

Grip is not observed. It follows from the fly's own recent actions.

## Steps

### 1. Event

- `events/swimming.py`: `Swimming(OlympicEvent)`, `PULL`, `RECOVER`, `LEGAL_ACTIONS`.
  Every rule is a dataclass field. Counters `strokes` and `turns`, and `grip`,
  `gliding` and `lap` for logging.
- `tests/test_swimming.py`: grip builds and resets, pulling at `a = 1` gives no
  thrust, glide versus recovery drag, the turn, the score for a race that doesn't
  finish, illegal actions and determinism.

### 2. Baseline controllers

- `controllers/swim_baselines.py`:
  - `SwimRuleController(v_break=2.6)`, the base controller. It pulls through the full
    range and recovers fully. After the dive and after the turn it glides until its
    speed drops below `v_break`. `v_break=None` turns gliding off (full strokes only).
  - `AlwaysController(action)`, `ChoppyController` (switches every step) and
    `RandomController(p=0.5)`.
- `tests/test_swim_baselines.py` locks in the ranking: rule with glide < full strokes
  < random < switching every step < Always-PULL and Always-RECOVER (neither finishes).

### 3. Results file

- `experiments/run_swimming.py`, a copy of `run_hurdles.py`. Options:
  `--controller`, `--seed`, `--output-dir`.
- Writes `results/swimming_<controller>_seed<n>.csv` and `.json`, one row per step:
  `step`, `simulation_time`, `position`, `lap`, `velocity`, `distance_to_wall`,
  `stroke_phase`, `grip`, `gliding`, `strokes`, `turns`, `action`. The `fly`
  controller adds the motor readout columns.

### 4. Fly brain encoder, decoder and starting weights (untrained)

- `brain/swimming_io.py`:
  - `SwimmingEncoder`: features `[distance_to_wall / 50, v / 4, stroke_phase]`,
    clipped to [0, 1], plus the 0.45 tonic drive. Input names `wall`, `velocity`,
    `stroke_phase`.
  - `SwimmingDecoder`: extensor − flexor > threshold means PULL, otherwise RECOVER.
- `models/swimming_encoder_init.csv`: 24 sensory neurons × 3 inputs = 72 rows, every
  neuron with weights (0.90, 0.25, 0.45). Never overwritten.
- `FlyBrainController.swimming(path=...)` and `--controller fly` in the runner.

### 5. User guide

- `docs/swimming.md`, like [hurdles.md](hurdles.md).

## Prototype results

A throwaway simulation of these rules:

| Policy | Score |
|--------|------:|
| Rule, glide until 2.6 m/s | **44.15 s** |
| Full strokes, no glide | 45.35 s |
| Random, p = 0.5 | 69.95 s |
| Switch every step | 110.10 s |
| Always-PULL | did not finish |
| Always-RECOVER | did not finish |

Without grip, random presses finished within about 2 s of the best policy and short
choppy strokes matched full ones. Grip is what makes rhythm matter.

The implemented event reproduces these scores exactly (random: 69.55 s at seed 7,
68.9 s mean over seeds 0–49). The untrained fly does not finish: it pulls for about 7
steps, then its motor output settles into RECOVER and it glides to 18.8 m (score
201.25). That is expected before training. As in Hurdles, the network saturates, and
here a rhythm needs the output to switch on and off every 8–12 steps.

## Open questions

- The glide only saves about 1.2 s. A bigger gap between `swim_drag` and
  `glide_drag` would make breakout timing matter more.
- Should faster stroking cost energy (fatigue)?
