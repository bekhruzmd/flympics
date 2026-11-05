# Hurdles: rules and running your own tests

For the design behind the rules, see [hurdles-plan.md](hurdles-plan.md).

All commands run from the repository root in PowerShell and use the project's
virtual environment directly, so there is no need to activate it. In Git Bash, the
commands that pipe a here-string (`@' ... '@ | ...`) work the same way written as
`.venv/Scripts/python - <<'EOF' ... EOF`.

## The event in one paragraph

The fly runs 200 m over 7–13 hurdles spaced 8–30 m apart. Each step it chooses:

- **ACCELERATE:** push forward;
- **COAST:** no push, drag slowly brakes it;
- **JUMP:** a 0.8 s jump.

A hurdle is cleared if the fly is high enough when it passes. Taking off too early
means it is already coming down; too late means it is still rising. Either way it
hits the hurdle and loses 70% of its speed. Taking off faster than 4.5 m/s loses
extra speed, so the fastest race accelerates between hurdles, coasts down to
4.5 m/s, then jumps on time. The fly sees the distances to the next two hurdles and
its own speed. The score is the finish time, or `90 + metres left` if time runs
out, so lower is better.

## The strategies

| Name | What it does |
|------|--------------|
| `rule` | Accelerates, coasts down to 4.5 m/s before each hurdle, then jumps in the middle of the window |
| `nocoast` | Jumps in the middle of the window but never coasts, so it takes off too fast |
| `always_accelerate` | Never jumps, so it hits every hurdle |
| `always_coast` | Never moves |
| `always_jump` | Jumps whenever it is on the ground. Never gets going |
| `random` | Jumps 1% of steps, otherwise accelerates |
| `fly` | The untrained connectome fly brain. It accelerates for a few steps, then coasts to a stop, because its output settles in the COAST band. Training is a separate project |

## 1. Run one race

```powershell
.venv/Scripts/python -m experiments.run_hurdles --controller rule --layout seeded --seed 42
```

| Option | Values | Default |
|--------|--------|---------|
| `--controller` | any name from the table above | `rule` |
| `--layout` | `benchmark` (the fixed course) or `seeded` (a random course) | `benchmark` |
| `--seed` | Any whole number. Picks the course when `--layout seeded`, and the actions for `random` | `7` |
| `--output-dir` | Folder for the logs and pictures | `results` |
| `--no-plot` | Skip the distance and speed plot | plot is saved |
| `--view` | Open an animated window of the race (Space pauses) | off |
| `--gif` | Save the animation as a GIF at 2.5× speed. Takes about a minute | off |

It prints the distance, time, score, hits and jumps, and saves:

- `results/hurdles_<layout>_<controller>_seed<N>.csv` and `.json`: one row per step;
- `.png`: distance over time with hurdles and hits marked, and speed over time with
  jumps and coasting shaded and the 4.5 m/s safe speed drawn as a line;
- `.gif` and `.activity.png` with `--gif`.

The same seed always gives the same course and the same result.

## 2. Compare every strategy on one course

```powershell
foreach ($c in "rule", "nocoast", "always_accelerate", "random", "always_coast", "always_jump", "fly") {
    .venv/Scripts/python -m experiments.run_hurdles --controller $c --layout seeded --seed 42 --no-plot
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
for name in ("rule", "nocoast", "always_accelerate", "random", "fly"):
    races = [run(name, seed, "seeded") for seed in SEEDS]
    print(f"{name:18s} mean score {statistics.mean(score(r) for r in races):6.2f}"
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
from controllers.hurdle_baselines import RandomController
from experiments.run_hurdles import run, score

for p_coast in (0.0, 0.1, 0.3):
    races = [run(seed=seed, layout="seeded", controller=RandomController(p_coast=p_coast)) for seed in range(20)]
    print(f"random p_coast={p_coast}: mean score {statistics.mean(score(r) for r in races):.2f}")
'@ | .venv/Scripts/python -
```

| Strategy | Settings you can change |
|----------|------------------------|
| `HurdleRuleController(coast=True, event=None)` | `coast`: whether it brakes before hurdles. `event`: the race settings it plans for (see section 6) |
| `RandomController(p_jump=0.01, p_coast=0.0)` | Chance of jumping and of coasting on each step; otherwise it accelerates |
| `AlwaysController(action)` | `ACCELERATE`, `COAST` or `JUMP` |
| `FlyBrainController.hurdles(path=..., coast_threshold=0.015, jump_threshold=0.30)` | `path`: an encoder weights CSV. The thresholds split the brain's extensor − flexor difference into COAST, ACCELERATE and JUMP |

## 5. Build your own course

A course is a tuple of hurdle positions in metres. `run()` only uses the benchmark or
seeded courses, so a custom course needs a short loop of its own:

```powershell
@'
from controllers.hurdle_baselines import HurdleRuleController
from events.hurdles import Hurdles

COURSE = (20.0, 30.0, 60.0, 70.0, 100.0, 108.0, 140.0, 170.0, 185.0)
for coast in (True, False):
    event, controller = Hurdles(layout=COURSE), HurdleRuleController(coast=coast)
    event.reset(); controller.reset()
    while not event.is_finished():
        event.step(controller.act(event.observe()))
    print(f"rule coast={coast}: score {event.score():.2f}, "
          f"{event.hits} hits, {event.jumps} jumps, {event.coasts} coast steps")
'@ | .venv/Scripts/python -
```

The same loop works with any strategy from the table in section 4.

A course must follow the layout rules. If it doesn't, `reset()` raises a
`ValueError` naming the problem:

- the first hurdle at 15 m or later;
- the last hurdle at least 12 m before the 200 m finish;
- every gap at least 8 m, so each hurdle needs its own jump.

To check a course without running it (a valid course prints nothing):

```powershell
.venv/Scripts/python -c "from events.hurdles import validate_layout; validate_layout((20.0, 25.0), 200.0)"
```

This example has a 5 m gap, so it raises a `ValueError`.

## 6. Change the race rules

Every rule is a setting on `Hurdles`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `distance` | 200.0 | Track length (m) |
| `dt` | 0.05 | Seconds per step |
| `max_time` | 90.0 | Time limit (s) |
| `acceleration`, `drag` | 1.25, 0.16 | Running physics, same as Sprint. Drag is also the only braking, both when coasting and in the air |
| `air_time` | 0.8 | Jump length (s) |
| `jump_height`, `hurdle_height` | 1.0, 0.5 | The hurdle is cleared while the jump is above `hurdle_height` |
| `takeoff_keep` | 0.90 | Share of speed kept at take-off, at or below `safe_speed` |
| `safe_speed` | 4.5 | Fastest take-off with no extra cost (m/s) |
| `overspeed_cost` | 0.40 | Share of speed lost per m/s above `safe_speed` |
| `min_takeoff_keep` | 0.20 | Least share of speed kept at take-off, however fast |
| `stumble_keep` | 0.30 | Share of speed kept after hitting a hurdle |
| `sensor_range` | 40.0 | Cap on the observed hurdle distances (m) |

For example, to see how steep the over-speed cost must be before coasting pays off:

```powershell
@'
from controllers.hurdle_baselines import HurdleRuleController
from events.hurdles import Hurdles

for cost in (0.1, 0.2, 0.3, 0.4, 0.6):
    scores = {}
    for coast in (True, False):
        event = Hurdles(overspeed_cost=cost)
        controller = HurdleRuleController(coast=coast, event=event)
        event.reset(); controller.reset()
        while not event.is_finished():
            event.step(controller.act(event.observe()))
        scores[coast] = event.score()
    print(f"overspeed_cost={cost}: coasting gains {scores[False] - scores[True]:+.2f} s")
'@ | .venv/Scripts/python -
```

Always pass the same event as `event=` to `HurdleRuleController`, so it plans for
those rules. The rule reads `safe_speed`, the take-off cost, `drag` and the jump
settings from it.

## 7. Read a race log

Every command-line run writes a CSV and a JSON file with one row per step:

| Column | Meaning |
|--------|---------|
| `step`, `simulation_time` | Step number and time (s) |
| `position`, `velocity` | Metres covered, speed (m/s) |
| `distance_to_next_hurdle`, `distance_to_second_hurdle` | What the strategy saw (capped at 40 m) |
| `height`, `airborne` | Jump height and whether the fly is in the air |
| `hits`, `jumps`, `coasts` | Running totals |
| `takeoff_keep` | Share of speed kept at the latest take-off |
| `action` | `ACCELERATE`, `COAST` or `JUMP` |
| `motor_extensor`, `motor_flexor`, `motor_difference`, ... | Only for `fly`: the brain's motor readout |

To list each take-off with its speed:

```powershell
@'
import pandas as pd
log = pd.read_csv("results/hurdles_seeded_nocoast_seed42.csv")
takeoffs = log[log["jumps"].diff().fillna(log["jumps"]) > 0]
print(takeoffs[["simulation_time", "position", "velocity", "takeoff_keep"]].to_string(index=False))
'@ | .venv/Scripts/python -
```

Run section 1 with `--controller nocoast` first so the file exists. Compare it with
`rule`: its take-offs keep 90% of speed, while `nocoast`'s keep much less.

## 8. Run the automated tests

```powershell
.venv/Scripts/python -m pytest tests/test_hurdles.py tests/test_hurdle_baselines.py tests/test_hurdles_brain.py tests/test_hurdles_view.py -q
```

These check the event rules, the layout rules, the expected ranking of the
strategies, the fly brain setup and the viewer. Run them after changing anything in
`events/hurdles.py`. A changed rule can quietly change which strategy wins; the
prototype found that an `overspeed_cost` of 0.2 is already too low for coasting to
pay off.
