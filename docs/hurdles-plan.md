# 200 m Hurdles: implementation plan

The second Flympics event. The fly runs 200 m over 10–14 hurdles and chooses
between two actions, RUN and JUMP. Jumping too early, too late or too often costs time.

The event follows the `OlympicEvent` interface in [events/base.py](../events/base.py)
and copies the conventions of [events/sprint.py](../events/sprint.py): 1-D track,
fixed `dt`, deterministic for a given seed, lower score is better. Like Sprint, it
is a game model, not fly biomechanics.

## Scope

**In scope:**

- The event itself.
- The baselines that show the event isn't trivially solved.
- A simple, trainable encoder, decoder and set of starting encoder weights for the
  fly brain, built the same way as Sprint's.

**Out of scope:**

- **Training.** Making the fly good at hurdles is a separate project.
- **The connectome and the network dynamics.** `RateNetwork`'s gain, leak and
  substeps stay unchanged.
- **Sprint.** Sprint's behaviour and its tests stay unchanged.

The untrained fly is not expected to score well. It only has to run a full race
without errors.

## Why this event

In Sprint v1 the best action is always ACCELERATE (see
[training-plan.md](training-plan.md)). Hurdles is built so that every simple
policy loses:

- **Always RUN** hits every hurdle.
- **Always JUMP** never gets going.
- **One look-ahead** can't time a jump over a close pair.

A good score needs the controller to combine distance and velocity at the right
moment.

## Rules

| Rule | Value | Reason |
|------|-------|--------|
| Track length | 200 m | Room for both long gaps and tight pairs |
| Hurdles | 10–14, first one at 15 m or later | |
| `dt` | 0.05 s | At 0.1 s a pair's take-off window can be shorter than one step |
| `max_time` | 90 s | A clean 200 m takes about 32 s; Always-RUN takes about 60 s |
| RUN | Same physics as Sprint ACCELERATE: `acceleration` 1.25, `drag` 0.16, top speed 7.81 m/s | Results can be compared with Sprint |
| JUMP | Only works on the ground. Starts a jump of `air_time` = 0.8 s, counted as 16 whole steps | Whole steps make every landing exact |
| In the air | No push forward, drag still applies, and every action is ignored | The controller can't see whether it is airborne, so spamming JUMP is legal, just slow |
| Take-off cost | Speed × 0.90 at each take-off | Makes jumping too often expensive |
| Jump height | `y(t) = 4·H·t·(T − t) / T²`, with `H` = 1 | |
| Hurdle height | `h` = 0.5·H | The fly is above `h` for the middle 71% of each jump |
| Clearing a hurdle | Cleared only if the fly is in the air **and** `y ≥ h` when it passes the hurdle. The exact crossing time is interpolated within the step | Gives a window with two edges: too early means already coming down, too late means still going up |
| Hitting a hurdle | Speed × 0.30. No disqualification | Real rules allow knocking hurdles. The time is lost through physics, not an added penalty |
| Score | Finish time if the fly finishes; otherwise `max_time + remaining distance` (same as Sprint) | |

### Layout rules

- **Pair:** two hurdles 1.0–1.5 m apart. One jump can clear both, but only with
  good timing.
- **Spacing:** a pair never touches a third hurdle. Every other gap is at least 6 m.
  The generator draws "close" gaps of 6–9 m and "far" gaps of 20–35 m.
- **Run-up and run-out:** at least 15 m before the first hurdle and at least 12 m
  after the last.
- These rules guarantee that **two look-ahead observations are enough**. No take-off
  decision depends on a third hurdle.

There are two kinds of layout:

- `BENCHMARK_LAYOUT`: one fixed course used for every scoreboard comparison.
- `generate_layout(seed)`: a seeded random course for training and for checking
  that a controller works on new courses. Each gap is a pair with probability 0.3
  (never two pairs in a row); otherwise it is close or far with equal chance. The
  generator redraws until the course has 10–14 hurdles.

### Observation

```python
{
    "distance_to_next_hurdle":   min(next_x - position, 40.0),
    "distance_to_second_hurdle": min(second_x - position, 40.0),
    "velocity":                  velocity,
}
```

When fewer hurdles are left than the observation needs, the missing distance is set
to the 40 m cap, meaning "nothing nearby". There is no airborne flag and no distance
to the finish.

## Steps

### 1. Event ✅ done

- [events/hurdles.py](../events/hurdles.py) contains:
  - `Hurdles(OlympicEvent)`;
  - `RUN`, `JUMP`, `LEGAL_ACTIONS`;
  - `BENCHMARK_LAYOUT`, `generate_layout(seed)` and `validate_layout(hurdles, distance)`;
  - counters `hits`, `jumps` and `clears`, and properties `airborne` and `height`, for logging.
- [tests/test_hurdles.py](../tests/test_hurdles.py): 16 tests. They cover the
  two-edged timing window, clearing a pair with one jump, the take-off cost and
  stumble, ignored mid-air actions, the distance cap, 1000 generated layouts, the
  rejected layouts, and Always-RUN and Always-JUMP.
### 2. Baselines and runner ✅ done

| Controller | Benchmark score | Hits | Seeds 0–49: mean score | Mean hits |
|------------|----------------:|-----:|-----------------------:|----------:|
| `rule` (2 look-ahead) | 42.50 s | 0 | 42.42 s | 0.00 |
| `rule1` (1 look-ahead) | 47.25 s | 2 | 45.12 s | 1.28 |
| `always_run` | 59.60 s | 13 | 58.54 s | 12.36 |
| `random` (p = 0.01) | 67.25 s | 13 | 65.20 s | 11.60 |
| `always_jump` | 290 (did not finish) | 0 | 290 | 0.00 |

The benchmark course has one 1.2 m pair that both rules clear, and two wider pairs
(1.5 m at 90 m, 1.4 m at 160 m) that only the two-look-ahead rule clears. Race time is
now `steps × dt`, so a race that runs out of time stops at exactly 90 s.

| File | Content |
|------|---------|
| `controllers/hurdle_baselines.py` | `AlwaysController(action)`, `RandomJumpController(p=0.01, seed)` (about 12 jumps per race, roughly one per hurdle, at random times) and `HurdleRuleController(look_ahead=2)`. The rule predicts its speed in the air as `v · 0.90 · (1 − drag · T/2)` and its jump length as `L = v_air · T`. Its take-off window is `[0.146·L, 0.854·L]`, or `[0.146·L, 0.854·L − s]` for a pair `s` apart. It jumps once the next hurdle is at the middle of the window. With `look_ahead=1` it ignores pairs |
| `experiments/run_hurdles.py` | Copy of `run_sprint.py`. Each step's record adds `distance_to_next_hurdle`, `distance_to_second_hurdle`, `height`, `airborne`, `hits` and `jumps`. Command-line options: `--controller`, `--layout benchmark\|seeded`, `--seed`. Writes CSV and JSON to `results/` |
| `BENCHMARK_LAYOUT` | Same rules, with pairs the one-look-ahead rule fails on |
| Tests | Mean score on the benchmark and on seeds 0–49 ranks as `rule(2) < rule(1) < Always-RUN < random < Always-JUMP`. Random jumping is worse than never jumping: at p = 0.01 it averages 66 s against Always-RUN's 60 s, and at p = 0.1 it never finishes. On the benchmark, `rule(2)` hits no hurdles and `rule(1)` hits at least one |

### 3. Fly brain encoder, decoder and starting weights

Kept simple on purpose, so it can serve as the starting point for later training.
It has the same form as Sprint's, and nothing is tuned for performance.

| Part | Plan |
|------|------|
| `brain/hurdles_io.py`: `HurdlesEncoder` | The same form as `LearnedSensoryEncoder`. Features `[d1/40, d2/40, v/8]`, each clipped to [0, 1], where `d1` and `d2` are the distances to the next and second hurdle. Stimulus per sensory neuron: `0.45 + E @ features`. `E` has shape (24, 3). The input names are `next_hurdle`, `second_hurdle` and `velocity` |
| `brain/hurdles_io.py`: `HurdlesDecoder` | The same comparison as `MotorDecoder`: extensor − flexor > threshold → JUMP, otherwise RUN. The threshold is a parameter with Sprint's default of 0.015 |
| `models/hurdles_encoder_init.csv` | The same `pre,post,weight` format as Sprint's, one row per input → sensory edge (72 rows). Every neuron gets (0.90, 0.45, 0.25), following Sprint's (0.90, 0.25). Never overwritten |
| Loading and saving | The existing CSV loading and saving functions are generalised to take a list of input names, with Sprint's list as the default. Alternatively, Hurdles gets its own copies of them |
| `brain/controller.py` | `FlyBrainController` also accepts an `encoder` and a `decoder` from outside. `FlyBrainController.hurdles(path=...)` builds the Hurdles setup. If nothing is passed in, it builds exactly what it builds today |
| `run_hurdles.py` | Adds a `--controller fly` option |
| Tests | The starting CSV loads with shape (24, 3). Encoder output matches the formula above. An untrained fly runs a full benchmark race and a seeded race without errors and returns legal actions. All Sprint tests pass unchanged, including the bit-identical encoder check |

## Prototype results

A throwaway simulation of these rules, averaged over 100 seeded layouts with about
13.6 hurdles each, `dt` = 0.05:

| Policy | Mean score | Hurdles hit per race |
|--------|-----------:|---------------:|
| Always-RUN | 60.1 s | 13.6 |
| Always-JUMP | 290 (did not finish) | 0 |
| Rule, 1 look-ahead | 45.0 s | 1.08 |
| Rule, 2 look-ahead | **43.0 s** | **0.00** |
| Clean 200 m, no hurdles | 31.8 s | – |

What this showed:

- **The first settings were too tight.** With a 0.6 s jump and a hurdle at 0.75
  of jump height, a 1.5 m pair couldn't be cleared below about 6 m/s. The current
  settings give a take-off window of 2–5 steps at normal race speeds.
- **The rule has to use speed after take-off.** If it plans with the speed measured
  before the jump, it overestimates the jump length and comes down on the second
  hurdle of a pair.
- **Always-JUMP never moves.** It takes off at speed 0 and gets no push in the air,
  which is why step 2 adds a random-jump baseline.

## Notes for later training (not addressed here)

- With the current network settings and all-positive input, every neuron saturates
  within about 8 steps and stays there. Signed encoder weights can switch the motor
  output on in 3–4 steps and off in about 6.
- The output's "off" level of about +0.02 is above the 0.015 default threshold, so
  the threshold has to be fitted during training.
- JUMP is the right action in only about 1 step in 50. Training will need to handle
  that imbalance.

## Open questions

- Should there be "choice gaps" of 3–4 m, where clearing both hurdles with one jump
  and jumping twice are both viable?
- Should a hit cost more at higher speed?
- Should hurdle height vary in the seeded courses?
