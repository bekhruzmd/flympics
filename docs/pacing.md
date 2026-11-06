# Running your own 400 m Pacing tests

How to test 400 m Pacing strategies, train neural controllers, and inspect race physics.
For event rules and design, see [pacing-plan.md](pacing-plan.md).

All commands run from the `flympics` directory and use the virtual environment:

```powershell
cd flympics
.\.venv\Scripts\Activate.ps1
```

## The strategies

| Name | Description |
|------|-------------|
| `rule` | Base pacing controller: accelerates to maintain a sustainable cruise speed (3.3 m/s), coasting to conserve energy |
| `always_acc` | Sprints continuously without pacing; burns energy by 66.7 s and crawls under fatigue |
| `always_coast` | Never accelerates; stays at the starting blocks |
| `random` | Accelerates with probability p = 0.5 on each step |
| `sprint_rule` | Applies the 100 m Sprint rule (target 7.0 m/s); exhausts energy early |
| `fly` | The untrained connectome fly controller (threshold 0.015) |
| `trained` | The connectome fly controller fitted to 400 m pacing via motor threshold training |

## 1. Run one race

```powershell
python -m experiments.run_pacing --controller rule
```

| Option | Values | Default |
|--------|--------|---------|
| `--controller` | `rule`, `always_acc`, `always_coast`, `random`, `sprint_rule`, `fly`, `trained` | `rule` |
| `--seed` | Any integer (seeds `random`) | `7` |
| `--output-dir` | Folder for the logs | `results` |
| `--no-plot` | Skip saving the distance/speed/energy plot | off |
| `--view` | Open interactive Matplotlib viewer (Space pauses/resumes) | off |
| `--gif` | Export animated GIF | off |

Output:
```text
rule: 400.0 m in 123.00 s, score 123.00; final energy=0.8, 0 exhausted steps. Logs: results\pacing_rule_seed7.json/.csv
```

## 2. Compare the main strategies

```powershell
foreach ($c in "rule", "trained", "random", "fly") {
    python -m experiments.run_pacing --controller $c --no-plot
}
```

Leaderboard:
| Strategy | Score (lower is better) | Time (s) | Distance (m) | Final Energy | Strategy / Note |
|----------|------------------------:|---------:|-------------:|-------------:|:----------------|
| `rule` | **123.00** | 123.00 | 400.0 | 0.8 | Paced cruise at 3.3 m/s avoiding quadratic drag |
| `random` (seed 7) | 129.70 | 129.70 | 400.0 | 2.8 | Stochastically alternates acceleration and coasting |
| `trained` (fly) | 198.10 | 198.10 | 400.0 | 0.0 | MANC circuit with fitted threshold maintaining forward drive |
| `fly` (untrained) | 635.34 (DNF) | 240.00 | 4.7 | 99.4 | Initial threshold causes circuit to coast early and time out |

## 3. Train the 400 m pacing fly

To train the artificial motor threshold against the 400 m pacing race:

```powershell
python -m experiments.train_pacing
```

This performs a deterministic 34-point grid search over candidate motor thresholds,
recording every trial to `models/pacing.json`. The untrained fly times out at 4.7 m (score 635.34),
while the trained policy achieves full 400 m race completion in 198.10 s.

## 4. Read race logs

Every command-line run writes CSV and JSON logs to `results/`:

| Column | Meaning |
|--------|---------|
| `step`, `simulation_time` | Step number and time in seconds (0.1 s per step) |
| `position` | Distance covered in metres |
| `velocity` | Speed in m/s |
| `energy` | Remaining energy budget (0 to 100) |
| `distance_to_finish` | Remaining metres to the finish line |
| `action` | `ACCELERATE` or `COAST` |
| `exhausted` | Boolean flag indicating whether energy was depleted |
| `sensory_mean`, `intrinsic_mean`, `motor_extensor`, `motor_flexor`, `motor_difference` | Fly controllers only: VNC population activity and readout |

## 5. Automated tests

Run the 400 m pacing test suite:

```powershell
python -m pytest tests/test_pacing.py tests/test_pacing_baselines.py tests/test_pacing_brain.py tests/test_run_pacing.py tests/test_pacing_view.py -v
```
