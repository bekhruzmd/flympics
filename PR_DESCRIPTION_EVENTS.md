## What changed

Adds two new events, **200 m Hurdles** and **100 m Swimming**, each with baselines, a results runner, a plot and animation, and an untrained fly-brain encoder and decoder. Each tests something Sprint doesn't: Hurdles needs **speed control and jump timing**, and Swimming needs **rhythm**.

**Training is out of scope** for both. The untrained fly only has to finish a race run without errors.

### 200 m Hurdles

The fly runs over 7–13 randomly spaced hurdles and chooses **ACCELERATE**, **COAST** or **JUMP** on every step. The best race speeds up between hurdles, coasts down to a safe speed before each one, and jumps on time.

- **`events/hurdles.py`**:
  - ACCELERATE uses Sprint's physics.
  - A jump lasts 0.8 s and clears the hurdle only if the fly is high enough as it passes.
  - Taking off faster than `safe_speed` (4.5 m/s) costs extra speed, and hitting a hurdle keeps only 30% of it.
  - There is a fixed `BENCHMARK_LAYOUT` and seeded `generate_layout(seed)`, and `validate_layout` checks the spacing rules.
- **`controllers/hurdle_baselines.py`**: `HurdleRuleController`, which coasts to `safe_speed` and jumps at the middle of the take-off window, plus `nocoast`, always-action and random strategies.
- **`experiments/run_hurdles.py`** and **`visualization/hurdles_view.py`**: logs, plot and `--view` / `--gif`. Options: `--controller`, `--layout benchmark|seeded`, `--seed`.
- **`brain/hurdles_io.py`** and **`models/hurdles_encoder_init.csv`**:
  - Encoder: 24 × 3 weights on the distances to the next two hurdles and speed.
  - Decoder: three bands. COAST, ACCELERATE or JUMP as extensor − flexor rises past 0.015 and 0.30.
- **Docs:** `docs/hurdles-plan.md` and `docs/hurdles.md`.

The history keeps v1 (RUN/JUMP with close hurdle pairs) in `1205fcb` and `0034ee3`, and the v2 rework in `564119b`. The plan explains why v1 was replaced: seeing the second hurdle didn't matter, and late jumps were rewarded.

### 100 m Swimming

A fixed race in a 50 m pool with one turn. The fly chooses **PULL** or **RECOVER** on every step.

- **`events/swimming.py`**:
  - A pull pushes while the arm still has range (12 steps), with thrust `thrust · grip · (1 − v / hand_speed)`.
  - **Grip** builds over the first 5 steps of an unbroken pull and resets on RECOVER, so full strokes beat quick taps.
  - Recovering brings the arm forward in 8 steps with extra drag. With the arm forward, RECOVER is a streamlined glide with 4× lower drag, which pays off after the dive and the turn.
  - The race is fixed: the seed only seeds random controllers.
- **`controllers/swim_baselines.py`**: `SwimRuleController(v_break=2.6)`, which makes full strokes and glides after the dive and the turn until its speed drops below `v_break`. Also always-PULL, always-RECOVER, switch-every-step and random strategies.
- **`experiments/run_swimming.py`** and **`visualization/swimming_view.py`**: logs, a plot with the glides and the turn marked, and `--view` / `--gif`. The viewer shows the fly swimming out and back with its arm position and grip. The GIF plays at 5× and is written frame by frame with Pillow, which took a full race from about 4 minutes to under 1.
- **`brain/swimming_io.py`** and **`models/swimming_encoder_init.csv`**:
  - Encoder: 24 × 3 weights on wall distance, speed and stroke phase.
  - Decoder: Sprint's extensor − flexor comparison, read as PULL or RECOVER.
- **Docs:** `docs/swimming-plan.md` and `docs/swimming.md`.

### Shared brain changes

- **`brain/sensory.py`**: the encoder CSV functions take a list of input names.
- **`brain/controller.py`**: `FlyBrainController` accepts an outside encoder and decoder, and gains `FlyBrainController.hurdles()` and `FlyBrainController.swimming()`.

Sprint's defaults and behaviour are unchanged.

## Data vs. modeled assumptions

**From MANC data (unchanged):** the 135-neuron front-leg circuit, its connectivity and which neurons are sensory, extensor and flexor.

**Modeling choices:**
- **Both events are game models, not fly biomechanics.** That covers Hurdles' jump arc, take-off cost, stumble and layout rules, and Swimming's grip, hand-speed limit, glide and recovery drag, dive and turn.
- **Encoder inputs:**
  - Hurdles: `d1/40`, `d2/40`, `v/8`, with starting weights (0.90, 0.45, 0.25).
  - Swimming: `d_wall/50`, `v/4`, `stroke_phase`, with starting weights (0.90, 0.25, 0.45).
  - Both use Sprint's 0.45 tonic drive and follow Sprint's starting weights; the stroke-phase weight is untuned.
- **Swimming's `stroke_phase`** stands in for limb-position (proprioceptive) input, so the circuit doesn't have to generate the rhythm on its own.
- **Decoder thresholds:** Hurdles uses 0.015 and 0.30, and Swimming uses Sprint's 0.015. All are parameters to fit in training.

**Unchanged:** the connectome, `RateNetwork` and Sprint. The Sprint tests pass unchanged, including the bit-identical encoder check.

## Hypothesis and metric (for experiments)

**Hypotheses:**
- **Hurdles** can't be solved by a fixed policy: it needs speed control and jump timing together.
- **Swimming:** grip is what makes rhythm matter. Without it, random presses finish within about 2 s of the best controller.

**Metric:** score = finish time, or `max_time + metres left` if the fly doesn't finish (90 s for Hurdles, 120 s for Swimming). Lower is better.

**Hurdles:**

| Controller | Benchmark | Seeds 0–49: mean score | Mean hits |
|---|---:|---:|---:|
| `rule` | 48.10 s | 47.70 s | 0.00 |
| `nocoast` | 53.20 s | 51.78 s | 0.00 |
| `always_accelerate` | 58.35 s | 56.82 s | 9.82 |
| `random` | 65.85 s | 63.43 s | 9.04 |
| `always_coast`, `always_jump` | 290 (did not finish) | 290 | 0 |

**Swimming** (fixed race):

| Controller | Score |
|---|---:|
| `rule` (glide until 2.6 m/s) | **44.15 s** |
| `full_strokes` (never glides) | 45.35 s |
| `random` (p = 0.5) | 68.9 s mean over seeds 0–49 |
| `choppy` (switches every step) | 110.10 s |
| `always_recover` | 198.85 (did not finish) |
| `always_pull` | 205.33 (did not finish) |

**The untrained fly** doesn't finish either race, which is expected before training:
- **Hurdles:** it accelerates for 5 steps, then settles in the COAST band and stalls at 1.95 m (score 288.05).
- **Swimming:** it pulls for about 7 steps, then settles into RECOVER and glides to a stop at 18.8 m (score 201.25).

In both cases the untrained network saturates, and its motor output stops switching.

## Checklist
- [x] `python -m pytest -q` passes: 129 tests, 84 of them new (49 Hurdles, 35 Swimming). The branch merges cleanly into upstream `main`.
- [x] Runs are seeded and reproducible. Hurdles courses come from the seed or the fixed benchmark, the Swimming race is fixed, and random controllers are seeded.
- [x] No hardcoded actions or edited scores. The baselines are transparent rules, and the fly's actions come only from the decoders.
- [x] Docs updated: `docs/hurdles-plan.md`, `docs/hurdles.md`, `docs/swimming-plan.md`, `docs/swimming.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
