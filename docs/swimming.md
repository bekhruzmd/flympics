# Running your own Swimming tests

How to try the Swimming strategies and change the race rules. For the event rules and
the design behind them, see [swimming-plan.md](swimming-plan.md).

All commands run from the repository root in PowerShell and use the project's virtual
environment directly. In Git Bash, the commands that pipe a here-string
(`@' ... '@ | ...`) work the same way written as `.venv/Scripts/python - <<'EOF' ... EOF`.

The race is always the same: 100 m in a 50 m pool, one turn. The seed only changes
the `random` strategy.

## The strategies

| Name | What it does |
|------|--------------|
| `rule` | The base controller. Full strokes, and a glide after the dive and the turn until speed drops below 2.6 m/s |
| `full_strokes` | The same full strokes, never glides |
| `random` | Pulls with probability 0.5 on each step |
| `choppy` | Switches action on every step. Grip never builds |
| `always_pull` | Never recovers. Makes one stroke and stalls |
| `always_recover` | Never pulls. Glides off the dive and stops |
| `fly` | The untrained connectome fly brain |

## 1. Run one race

```powershell
.venv/Scripts/python -m experiments.run_swimming --controller rule
```

| Option | Values | Default |
|--------|--------|---------|
| `--controller` | One of the names above | `rule` |
| `--seed` | Any whole number. Only affects `random` | `7` |
| `--output-dir` | Folder for the logs | `results` |
| `--no-plot` | Skip the distance and speed plot | off |
| `--view` | Open the animated viewer. Space pauses and resumes | off |
| `--gif` | Save the animation as a GIF at 5× real time | off |

Every run also saves `swimming_<controller>_seed<n>.png`: distance and speed over
simulation time, with the turn and the glides marked. `--gif` adds `.gif` and
`.activity.png`, the last frame of the viewer. The viewer shows the fly swimming out
to the turn wall and back, its speed, its arm position and grip, and, for `fly`, the
brain's activity.

A GIF takes about 50 s for a finished race and about a minute for a race that runs
to the 120 s limit:

```powershell
.venv/Scripts/python -m experiments.run_swimming --controller rule --gif
```

It prints one line:

```
rule: 100.0 m in 44.15 s, score 44.15; 41 strokes, 1 turn(s), 77 glide steps. Logs: results\swimming_rule_seed7.json/.csv
```

The **score** is the race time when the fly finishes. Otherwise it is
`120 + metres left`, so lower is always better.

| Strategy | Score |
|----------|------:|
| `rule` | 44.15 |
| `full_strokes` | 45.35 |
| `random` (seed 7) | 69.55 |
| `choppy` | 110.10 |
| `always_recover` | 198.85 (did not finish) |
| `fly` (untrained) | 201.25 (did not finish) |
| `always_pull` | 205.33 (did not finish) |

## 2. Compare every strategy

```powershell
foreach ($c in "rule", "full_strokes", "random", "choppy", "always_pull", "always_recover", "fly") {
    .venv/Scripts/python -m experiments.run_swimming --controller $c
}
```

## 3. Change a strategy's settings

Build the controller yourself and pass it to `run()`. For example, to find the best
breakout speed for the rule:

```powershell
@'
from controllers.swim_baselines import SwimRuleController
from experiments.run_swimming import run, score

for v_break in (None, 2.0, 2.4, 2.6, 2.8, 3.0):
    print(f"v_break={v_break}: score {score(run(controller=SwimRuleController(v_break=v_break))):.2f}")
'@ | .venv/Scripts/python -
```

| Strategy | Settings you can change |
|----------|------------------------|
| `SwimRuleController(v_break=2.6)` | `v_break`: glide while faster than this after the dive and the turn. `None` never glides |
| `RandomController(p=0.5)` | `p`: chance of pulling on each step |
| `AlwaysController(action)` | `PULL` or `RECOVER` |
| `ChoppyController()` | None |
| `FlyBrainController.swimming(path=..., motor_threshold=0.015)` | `path`: an encoder weights CSV (default `models/swimming_encoder_init.csv`). `motor_threshold`: how far extensor activity must exceed flexor activity to PULL |

## 4. Change the race rules

Every rule is a setting on `Swimming`. Pass a changed event to `run()`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `distance`, `pool_length` | 100.0, 50.0 | Race length and pool length (m). One turn per pool length |
| `dt` | 0.05 | Seconds per step |
| `max_time` | 120.0 | Time limit (s) |
| `pull_time`, `recover_time` | 0.6, 0.4 | Time to pull the arm through, and to bring it back (s) |
| `thrust`, `hand_speed` | 5.0, 4.0 | Full pull thrust is `thrust · grip · (1 − v / hand_speed)` |
| `grip_time` | 0.25 | Time for grip to build to full during a pull (s) |
| `swim_drag`, `glide_drag` | 0.08, 0.02 | Body drag `c · v²` while swimming and while gliding |
| `linear_drag` | 0.15 | Extra drag `linear_drag · v` at all times |
| `recovery_drag` | 0.5 | Extra drag `recovery_drag · v` while the arm moves forward |
| `dive_speed` | 4.0 | Speed at the start (m/s) |
| `push_off`, `turn_keep` | 3.5, 0.2 | Speed after the turn: `turn_keep · v + push_off` |

For example, to see how much grip matters:

```powershell
@'
from controllers.swim_baselines import ChoppyController, RandomController, SwimRuleController
from events.swimming import Swimming
from experiments.run_swimming import run, score

for grip_time in (0.25, 1e-9):
    for name, controller in (("rule", SwimRuleController()), ("random", RandomController()), ("choppy", ChoppyController())):
        records = run(controller=controller, event=Swimming(grip_time=grip_time))
        print(f"grip_time={grip_time:g} {name:7s} score {score(records, Swimming()):.2f}")
'@ | .venv/Scripts/python -
```

With `grip_time` close to 0, every pull has full grip at once, and random and choppy
strokes get close to the rule.

## 5. Read a race log

Every command-line run writes a CSV and a JSON file with one row per step:

| Column | Meaning |
|--------|---------|
| `step`, `simulation_time` | Step number and time (s) |
| `position`, `lap` | Metres covered, and lap 1 or 2 |
| `velocity` | Speed (m/s) |
| `distance_to_wall`, `stroke_phase` | What the strategy saw. `stroke_phase` is 0 with the arm forward, 1 at the end of a stroke |
| `grip` | Grip on this step's pull, 0 when not pulling |
| `gliding` | Whether this step was a streamlined glide |
| `strokes`, `turns` | Running totals. A stroke is counted at the first step of each pull |
| `action` | `PULL` or `RECOVER` |
| `sensory_mean`, `intrinsic_mean`, `motor_extensor`, `motor_flexor`, `motor_difference` | Only for `fly`: the brain's activity and motor readout |

To plot speed over the race:

```powershell
@'
import pandas as pd
log = pd.read_csv("results/swimming_rule_seed7.csv")
print(log.groupby(log["simulation_time"] // 5 * 5)["velocity"].mean().round(2).to_string())
'@ | .venv/Scripts/python -
```

## 6. Run the automated tests

```powershell
.venv/Scripts/python -m pytest tests/test_swimming.py tests/test_swim_baselines.py tests/test_swimming_brain.py tests/test_run_swimming.py tests/test_swimming_view.py -q
```

These check the event rules, the ranking of the strategies, the fly brain setup, the
CSV and JSON logs, the plot and the animation.
Run them after changing anything in `events/swimming.py`.
