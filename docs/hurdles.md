# Running your own Hurdles tests

How to try the existing Hurdles strategies on courses and settings of your choice.
For the event rules and the design behind them, see [hurdles-plan.md](hurdles-plan.md).

All commands run from the repository root in PowerShell and use the project's
virtual environment directly, so there is no need to activate it. In Git Bash, the
commands that pipe a here-string (`@' ... '@ | ...`) work the same way written as
`.venv/Scripts/python - <<'EOF' ... EOF`.

## The strategies

| Name | What it does |
|------|--------------|
| `rule` | Times each jump using speed and the distance to the next two hurdles. Clears pairs with one jump |
| `rule1` | The same timing rule, but sees only the next hurdle. Fails some pairs |
| `always_run` | Never jumps |
| `always_jump` | Jumps whenever it is on the ground. Never gets going |
| `random` | Jumps at random, 1% of steps (about one jump per hurdle) |
| `fly` | The untrained connectome fly brain |

## 1. Run one race

```powershell
.venv/Scripts/python -m experiments.run_hurdles --controller rule --layout seeded --seed 42
```

| Option | Values | Default |
|--------|--------|---------|
| `--controller` | `rule`, `rule1`, `always_run`, `always_jump`, `random`, `fly` | `rule` |
| `--layout` | `benchmark` (the fixed course) or `seeded` (a random course) | `benchmark` |
| `--seed` | Any whole number. Picks the course when `--layout seeded`, and the random jumps for `random` | `7` |
| `--output-dir` | Folder for the logs | `results` |

It prints one line:

```
rule on seeded: 200.0 m in 42.70 s, score 42.70; 0 hits, 10 jumps. Logs: results\hurdles_seeded_rule_seed42.json/.csv
```

The **score** is the race time when the fly finishes. Otherwise it is
`90 + metres left`, so lower is always better. The same seed always gives the same
course and the same result.

## 2. Compare every strategy on one course

```powershell
foreach ($c in "rule", "rule1", "always_run", "random", "always_jump", "fly") {
    .venv/Scripts/python -m experiments.run_hurdles --controller $c --layout seeded --seed 42
}
```

## 3. Compare strategies over many courses

One course can be lucky or unlucky. This averages each strategy over 50 random
courses (`fly` takes a few seconds):

```powershell
@'
import statistics
from experiments.run_hurdles import run, score

SEEDS = range(50)
for name in ("rule", "rule1", "always_run", "random", "fly"):
    races = [run(name, seed, "seeded") for seed in SEEDS]
    print(f"{name:11s} mean score {statistics.mean(score(r) for r in races):6.2f}"
          f"  mean hits {statistics.mean(r[-1]['hits'] for r in races):5.2f}"
          f"  worst {max(score(r) for r in races):6.2f}")
'@ | .venv/Scripts/python -
```

Change `SEEDS`, for example to `range(100, 200)`, to test on courses the baselines
were never checked on.

## 4. Change a strategy's settings

The command line uses each strategy's default settings. To change them, build the
controller yourself and pass it to `run()`:

```powershell
@'
import statistics
from controllers.hurdle_baselines import RandomJumpController
from experiments.run_hurdles import run, score

for p in (0.005, 0.01, 0.02, 0.05):
    races = [run(seed=seed, layout="seeded", controller=RandomJumpController(p=p)) for seed in range(20)]
    print(f"random p={p}: mean score {statistics.mean(score(r) for r in races):.2f}")
'@ | .venv/Scripts/python -
```

| Strategy | Settings you can change |
|----------|------------------------|
| `RandomJumpController(p=0.01)` | `p`: chance of jumping on each step |
| `HurdleRuleController(look_ahead=2, event=None)` | `look_ahead`: 1 or 2 hurdles. `event`: the race settings it times its jumps for (see section 6) |
| `AlwaysController(action)` | `RUN` or `JUMP` |
| `FlyBrainController.hurdles(path=..., motor_threshold=0.015)` | `path`: an encoder weights CSV. `motor_threshold`: how far extensor activity must exceed flexor activity to JUMP |

## 5. Build your own course

A course is a tuple of hurdle positions in metres. `run()` only uses the benchmark or
seeded courses, so a custom course needs a short loop of its own:

```powershell
@'
from controllers.hurdle_baselines import HurdleRuleController
from events.hurdles import Hurdles

COURSE = (20.0, 21.3, 40.0, 46.0, 80.0, 81.5, 120.0, 150.0, 151.0, 185.0)
for look_ahead in (1, 2):
    event, controller = Hurdles(layout=COURSE), HurdleRuleController(look_ahead=look_ahead)
    event.reset(); controller.reset()
    while not event.is_finished():
        event.step(controller.act(event.observe()))
    print(f"rule look_ahead={look_ahead}: score {event.score():.2f}, "
          f"{event.hits} hits, {event.clears} clears, {event.jumps} jumps")
'@ | .venv/Scripts/python -
```

The same loop works with any strategy from the table in section 4.

A course must follow the layout rules. If it doesn't, `reset()` raises a
`ValueError` naming the problem:

- the first hurdle at 15 m or later, and the last hurdle at least 12 m before the
  200 m finish;
- every gap is either a **pair** (1.0–1.5 m) or at least **6 m**;
- no three hurdles in a row with pair gaps between them.

To check a course without running it:

```powershell
.venv/Scripts/python -c "from events.hurdles import validate_layout; validate_layout((20.0, 23.0), 200.0)"
```

A valid course prints nothing. This example is deliberately invalid, so it ends with:

```
ValueError: Gap of 3.00 m after hurdle 0: use a pair gap (1.0, 1.5) or at least 6.0 m.
```

## 6. Change the race rules

Every rule is a setting on `Hurdles`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `distance` | 200.0 | Track length (m) |
| `dt` | 0.05 | Seconds per step |
| `max_time` | 90.0 | Time limit (s) |
| `acceleration`, `drag` | 1.25, 0.16 | Running physics, same as Sprint |
| `air_time` | 0.8 | Jump length (s) |
| `jump_height`, `hurdle_height` | 1.0, 0.5 | The hurdle is cleared while the jump is above `hurdle_height` |
| `takeoff_keep` | 0.90 | Share of speed kept at take-off |
| `stumble_keep` | 0.30 | Share of speed kept after hitting a hurdle |
| `sensor_range` | 40.0 | Cap on the observed hurdle distances (m) |

For example, to see how much a softer stumble penalty helps never jumping:

```powershell
@'
from controllers.hurdle_baselines import AlwaysController, HurdleRuleController
from events.hurdles import RUN, Hurdles

for keep in (0.3, 0.6, 0.9):
    rules = {"rule": None, "always_run": AlwaysController(RUN)}
    for name in rules:
        event = Hurdles(stumble_keep=keep)
        controller = rules[name] or HurdleRuleController(event=event)
        event.reset(); controller.reset()
        while not event.is_finished():
            event.step(controller.act(event.observe()))
        print(f"stumble_keep={keep} {name:10s} score {event.score():.2f}")
'@ | .venv/Scripts/python -
```

This prints:

```
stumble_keep=0.3 rule       score 42.50
stumble_keep=0.3 always_run score 59.60
stumble_keep=0.6 rule       score 42.50
stumble_keep=0.6 always_run score 49.40
stumble_keep=0.9 rule       score 42.50
stumble_keep=0.9 always_run score 37.00
```

At 0.9, never jumping beats the timing rule. The event only rewards jumping
because hitting a hurdle is expensive.

Pass the same event as `event=` to `HurdleRuleController` so it times its jumps for
those rules. The jump settings especially (`air_time`, `takeoff_keep`, `drag`,
`hurdle_height`) change where its take-off window is.

## 7. Read a race log

Every command-line run writes a CSV and a JSON file with one row per step:

| Column | Meaning |
|--------|---------|
| `step`, `simulation_time` | Step number and time (s) |
| `position`, `velocity` | Metres covered, speed (m/s) |
| `distance_to_next_hurdle`, `distance_to_second_hurdle` | What the strategy saw (capped at 40 m) |
| `height`, `airborne` | Jump height and whether the fly is in the air |
| `hits`, `jumps` | Running totals |
| `action` | `RUN` or `JUMP` |
| `motor_extensor`, `motor_flexor`, `motor_difference`, ... | Only for `fly`: the brain's motor readout |

To list where each hurdle was hit:

```powershell
@'
import pandas as pd
log = pd.read_csv("results/hurdles_seeded_rule1_seed42.csv")
hit_rows = log[log["hits"].diff().fillna(log["hits"]) > 0]
print(hit_rows[["simulation_time", "position", "velocity"]].to_string(index=False))
'@ | .venv/Scripts/python -
```

Run section 1 with `--controller rule1` first so the file exists.

## 8. Run the automated tests

```powershell
.venv/Scripts/python -m pytest tests/test_hurdles.py tests/test_hurdle_baselines.py tests/test_hurdles_brain.py -q
```

These check the event rules, the layout rules, the expected ranking of the
strategies and the fly brain setup. Run them after changing anything in
`events/hurdles.py`. A changed rule can quietly change which strategy wins.
