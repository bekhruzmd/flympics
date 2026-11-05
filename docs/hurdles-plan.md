# 200 m Hurdles v2: implementation plan

The fly runs 200 m over 7–13 randomly spaced hurdles and chooses between three
actions: **ACCELERATE**, **COAST** and **JUMP**. Taking off too fast wastes speed,
and taking off too late hits the hurdle. The best race speeds up between hurdles,
coasts down to a safe speed before each one, and jumps on time.

The event follows the `OlympicEvent` interface in [events/base.py](../events/base.py)
and the conventions of [events/sprint.py](../events/sprint.py): 1-D track, fixed `dt`,
deterministic for a given seed, lower score is better. It is a game model, not fly
biomechanics. How to run it is in [hurdles.md](hurdles.md).

## Status

v2 replaces v1 (RUN/JUMP, close pairs) in the same files. v1 survives only in Git
history, in commits `1205fcb` and `0034ee3` on the `hurdles` branch.

All three steps are built: the event, baselines, runner, viewer and the fly brain's
three-band decoder.

## Why v2

v1 had two problems, both found by simulation:

- **Seeing the second hurdle didn't matter** except for close pairs. When pairs were
  removed, the one-look-ahead and two-look-ahead rules behaved identically.
- **Late jumps were rewarded, not punished.** Any take-off inside the window clears
  the hurdle, and later take-offs keep running under power for longer. On 100
  courses, taking off as late as possible was the fastest strategy.

The root cause is that a jump scales with speed: its length, its take-off window
and its height all stretch in proportion. So going faster never makes a jump harder,
and no rule built only on jump timing makes speed a decision. v2 adds one rule that
penalises speed itself, plus a COAST action to manage it.

## Scope

**In scope:**

- The event itself.
- The baselines that show the event isn't trivially solved.
- The runner, plot and animation.
- A simple, trainable encoder, decoder and set of starting encoder weights for the
  fly brain.

**Out of scope:**

- **Training.**
- **The connectome and network dynamics.**
- **Sprint.** Its behaviour and tests stay unchanged.

## Rules

| Rule | Value | Change from v1 |
|------|-------|----------------|
| Track length | 200 m | |
| `dt`, `max_time` | 0.05 s, 90 s | |
| ACCELERATE | Same as Sprint: `acceleration` 1.25, `drag` 0.16, top speed 7.81 m/s | Was called RUN |
| COAST | No push; drag only. Slows the fly by 0.8% per step | **New** |
| JUMP | Only works on the ground. Starts a jump of `air_time` = 0.8 s (16 steps) | |
| In the air | No push, drag only, so exactly the same slowdown as COAST. Every action is ignored | |
| **Take-off cost** | Speed kept = 0.90 up to `safe_speed` = 4.5 m/s; above it, `0.90 − 0.40 × (v − 4.5)`, never below 0.20 | **New**: was a flat 0.90 |
| Jump height, hurdle height | `y(t) = 4·H·t·(T − t)/T²`, `H` = 1; hurdle at 0.5·H | |
| Clearing a hurdle | Cleared if in the air and `y ≥ h` when passing it. Window: take off 0.146·L to 0.854·L before the hurdle, where `L` is jump length | |
| Hitting a hurdle | Speed × 0.30, no disqualification | |
| Score | Finish time, or `max_time + metres left` | |

What the take-off cost means in practice:

| Speed at take-off | 4.5 m/s | 5.0 | 5.5 | 6.0 | 6.5 | 7.0+ |
|---|---|---|---|---|---|---|
| Speed kept | 90% | 70% | 50% | 30% | 20% | 20% |
| Speed after take-off | 4.05 | 3.50 | 2.75 | 1.80 | 1.30 | ≤ 1.56 |

Arriving faster than 4.5 m/s always leaves the fly slower after the jump. Braking
by coasting is slow: going from 6 m/s to 4.5 m/s takes about 1.8 s and 9.4 m. So a
fly coming off a long gap has to start coasting well before the hurdle.

### Layout

- **Gaps:** each gap is drawn uniformly from **8–30 m**, which gives 7–13 hurdles
  per course.
- **No pairs.** Every gap is wider than any one jump can clear (at most about 3.7 m
  at top speed), so each hurdle takes its own jump.
- **Run-up and run-out:** the first hurdle at 15 m or later, the last at least 12 m
  before the finish.
- `validate_layout` enforces these rules for any course, so every gap must be at least 8 m.
- **Two kinds of course:** `BENCHMARK_LAYOUT`, a fixed course with a mix of short
  and long gaps chosen in step 1, and `generate_layout(seed)`.

### Observation

Unchanged from v1:

```python
{
    "distance_to_next_hurdle":   min(next_x - position, 40.0),
    "distance_to_second_hurdle": min(second_x - position, 40.0),
    "velocity":                  velocity,
}
```

No baseline uses the second hurdle. It is kept so the encoder shape (24 × 3) doesn't
change, and so later training can test whether it helps. See the open questions.

## Prototype results

A throwaway simulation of these rules over seeds 0–49:

| Strategy | Mean score | Hits per race |
|----------|-----------:|--------------:|
| Coast to 4.5 m/s, jump mid-window (`rule`) | **47.70 s** | 0 |
| Same, but coasts only over the last 3 m | 48.83 s | 0 |
| Jump mid-window, never coast (`nocoast`) | 51.78 s | 0 |
| Always accelerate (never jump) | 56.82 s | 9.82 |
| Random: 1% jump, otherwise accelerate | 64.14 s | 9.16 |
| Random: 1% jump, 30% coast, otherwise accelerate | 81.49 s | 9.14 |
| Always coast / always jump | 290 (did not finish) | 0 |

Arrival speed for the coasting rule:

| Target speed at take-off | 4.0 | **4.5** | 4.8 | 5.1 | 5.5 |
|---|---|---|---|---|---|
| Mean score | 50.65 s | **47.70 s** | 48.66 s | 49.72 s | 51.08 s |

How steep the penalty must be, as the coasting rule's gain over `nocoast`:

| `safe_speed` \ `overspeed_cost` | 0.2 | 0.3 | 0.45 |
|---|---|---|---|
| 4.0 | −0.31 s | +2.83 s | +6.96 s |
| 4.5 | +0.17 s | +2.17 s | +5.02 s |
| 5.0 | +0.30 s | +1.44 s | +2.95 s |

What this showed:

- **With `overspeed_cost` 0.2, coasting is pointless.** The extra speed lost at
  take-off is smaller than the time spent braking. At 0.4 coasting gains about 4 s.
- **The best take-off speed is exactly `safe_speed`.** Missing it by 0.3 m/s in
  either direction costs about 1 s, so speed control is a real skill.
- **Late braking costs time:** coasting only over the last 3 m loses 1.1 s.

## Steps

### 1. Event ✅ done

The event gives exactly the prototype's scores on 5 seeded courses. The benchmark
course is `(15, 24, 52, 61, 70.5, 98, 107, 135, 144, 158, 186)`, alternating gaps of
about 9 m and 28 m.

`events/hurdles.py`, replacing v1:

- **Actions:** `ACCELERATE`, `COAST`, `JUMP`; `LEGAL_ACTIONS`.
- **New settings:** `safe_speed` = 4.5, `overspeed_cost` = 0.40, `min_takeoff_keep` = 0.20.
  `takeoff_keep` = 0.90 stays as the cost at or below `safe_speed`.
- **New method:** `takeoff_keep_at(v)` applies the table above. The rule controller
  uses it too.
- **Layout:** `MIN_GAP` = 8, `GAP` = (8, 30). The pair constants and the hurdle-count
  redraw go.
- **Kept from v1:** jump arc, interpolated crossing, stumble, observation, the
  counters (`hits`, `jumps`, `clears`), `airborne`, `height`, and time as `steps × dt`.
- **New counter:** `coasts` (steps spent coasting), for logging.

Tests (`tests/test_hurdles.py`, rewritten):

- same seed and actions give the same run;
- illegal actions and stepping a finished race are rejected;
- ACCELERATE matches Sprint's ACCELERATE exactly;
- COAST and a step in the air slow the fly by exactly the same amount;
- `takeoff_keep_at` gives 0.90 at or below 4.5 m/s, 0.50 at 5.5, and 0.20 at 7.5;
- the timing window has two edges (early and late hits, middle clears);
- actions in the air are ignored;
- a hit applies the stumble;
- the distance cap works;
- 1000 generated layouts pass validation;
- gaps under 8 m and short run-ups or run-outs are rejected.

### 2. Baselines, runner and viewer ✅ done

| Controller | Benchmark | Seeds 0–49: mean score | Mean hits |
|------------|----------:|-----------------------:|----------:|
| `rule` | 48.10 s | 47.70 s | 0.00 |
| `nocoast` | 53.20 s | 51.78 s | 0.00 |
| `always_accelerate` | 58.35 s | 56.82 s | 9.82 |
| `random` | 65.85 s | 63.43 s | 9.04 |
| `always_coast`, `always_jump` | 290 (did not finish) | 290 | 0 |

`controllers/hurdle_baselines.py`:

| Controller | Behaviour |
|------------|-----------|
| `HurdleRuleController(coast=True)` | Jumps at the middle of the window, computed from the speed kept after take-off. While above `safe_speed`, it coasts once the remaining distance to the take-off point is within its braking distance, `(v − safe_speed) / drag`. Otherwise it accelerates. `coast=False` gives `nocoast` |
| `AlwaysController(action)` | Always accelerate, always coast or always jump |
| `RandomController(p_jump=0.01, p_coast=0.0)` | Random actions |

`experiments/run_hurdles.py`:

- **Controller names:** `rule`, `nocoast`, `always_accelerate`, `always_coast`,
  `always_jump`, `random`, `fly`.
- **Records** add `coasts` and the speed kept at the latest take-off.
- **Options** stay the same.

`visualization/hurdles_view.py`:

- **Plot:** coasting stretches shaded on the speed plot, beside the jumps, and a
  dashed line at `safe_speed`.
- **Animation:** the status shows ACCEL, COAST, JUMP or AIR.

Tests:

- **Ranking** on the benchmark course and over seeds 0–49:
  `rule < nocoast < always_accelerate < random < always_coast = always_jump`
  (the last two both fail to finish).
- `rule` takes off within 0.3 m/s of `safe_speed` on at least 90% of its jumps.
- Runs are reproducible.
- The viewer tests from v1, updated.

### 3. Fly brain ✅ done

Result: the untrained fly accelerates for 5 steps, then its output settles in the
COAST band for the rest of the race. It stalls at 1.95 m on the benchmark, scoring
288.05.

- **Encoder:** unchanged. `HurdlesEncoder` keeps its 24 × 3 weights, the same
  starting CSV and the same `(0.90, 0.45, 0.25)` weights.
- **Decoder:** `HurdlesDecoder` reads Sprint's extensor − flexor difference against
  **two thresholds**:

  ```
  difference ≤ coast_threshold (0.015)            → COAST
  coast_threshold < difference ≤ jump_threshold   → ACCELERATE
  difference > jump_threshold (0.30)              → JUMP
  ```

  0.015 is Sprint's default, and the band between it and 0.30 keeps Sprint's meaning
  of "above the threshold means push". The order COAST < ACCELERATE < JUMP follows
  increasing extensor drive. Both thresholds are parameters, to be fitted in
  training.
- **Controller:** `FlyBrainController.hurdles(coast_threshold, jump_threshold)`.

Tests: the decoder's three bands, plus an untrained fly running a full race with
legal actions. The untrained fly is expected to stall: its difference settles near 0
after about 8 steps, which falls in the COAST band. The untrained Sprint fly does the
same, coasting to a standstill.

## Notes for later training

- The network saturates within about 8 steps of all-positive input. Signed encoder
  weights can switch the motor output on in 3–4 steps and off in about 6.
- With three actions, both thresholds have to be fitted.
- COAST and JUMP are each needed only in short, rare stretches, so the actions are
  imbalanced, and JUMP is the rarest.

## Open questions

- Should the second-hurdle observation be removed (encoder 24 × 2) if training shows it doesn't help?
- Should COAST brake harder than drag alone (`coast_drag`), so braking takes less track?
- Should hurdle heights or `safe_speed` vary between seeded courses?
