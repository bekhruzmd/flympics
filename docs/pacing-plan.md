# 400 m Pacing: implementation plan

The Flympics pacing event. The fly races a longer 400 m course with a strictly limited
energy budget, choosing between two legal actions on every 0.1 s step: `ACCELERATE` and `COAST`.

In short sprints (100 m), continuous acceleration is optimal because fatigue is negligible.
Over 400 m, an athlete with finite energy capacity who sprints flat-out burns out before
the halfway mark and suffers severe aerodynamic drag losses followed by an exhausting
crawl to the finish. Fast completion requires **pacing**: sustaining a steady, aerodynamically
efficient cruising speed that conserves the energy budget across the entire 400 m.

The event follows the `OlympicEvent` interface in [events/base.py](../events/base.py)
and respects all project principles: independent event interface, fixed 0.1 s step,
seeded runs, deterministic semi-implicit Euler physics, and absolute honesty about scope.

## Scope and biological attribution

**What comes from biological data:**
- **Connectivity and annotations:** Sparse edges and cell annotations come from the real,
  peer-reviewed MANC v1.2.1 adult Drosophila ventral nerve cord (VNC) release.
  The vendored 135-neuron circuit contains 24 front-leg proprioceptive sensory neurons,
  96 intrinsic VNC target interneurons, and 15 motor neurons (7 tibia extensor, 8 tibia flexor).
  The synaptic edge counts and weights in `data/manc_front_leg_v1/` are completely unchanged.

**What is a modeled assumption:**
- **Rate dynamics:** `RateNetwork` is an unsigned leaky rate model with gain 1.0, leak 0.1,
  tanh activation, and 2 sub-steps per 0.1 s. It is an engineering model, not biophysical spiking.
- **Sensory encoding:** Virtual distance to finish, velocity, and remaining energy budget are
  linearly scaled and applied as artificial current into the 24 front-leg proprioceptive nodes.
  Fruit flies in nature do not sense Olympic race distances or digital battery counters in their legs.
- **Motor decoding:** Tibia extensor minus flexor mean population rate is compared to a threshold
  to command virtual `ACCELERATE` vs `COAST`. Fruit fly motor neurons do not issue 400 m dash commands.
- **Task physics:** 1-D track, quadratic + linear drag, 100.0-unit energy budget with 1.5/s burn rate
  and 0.05 m/s² residual fatigue acceleration. This is a game benchmark, not whole-fly biomechanics.

## Event rules

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `distance` | 400.0 m | Track length |
| `dt` | 0.1 s | Fixed simulation time step |
| `max_time` | 240.0 s | Race time limit |
| `acceleration` | 1.25 m/s² | Normal acceleration while energy > 0 |
| `drag` | 0.10 s⁻¹ | Linear friction coefficient |
| `drag_quad` | 0.03 m⁻¹ | Aerodynamic quadratic drag coefficient |
| `energy_budget` | 100.0 | Initial energy capacity |
| `burn_rate` | 1.5 s⁻¹ | Energy consumed per second of acceleration |
| `fatigue_accel` | 0.05 m/s² | Exhausted crawl acceleration when energy = 0 |
| `recovery_rate` | 0.0 s⁻¹ | Default strict non-replenishing budget |

### Observations

```python
{
    "distance_to_finish": max(0.0, 400.0 - position),
    "velocity": velocity,
    "energy": energy,
}
```

### Scoring

Finish time if the fly crosses 400 m; otherwise `max_time + (distance - position)`.
Lower is always better.

## Strategies and results

| Controller | Strategy | Score | Time (s) | Distance (m) | Final Energy | Exhausted Steps |
|------------|----------|------:|---------:|-------------:|-------------:|----------------:|
| `rule` | Cruises at 3.3 m/s; coasts when above target | **123.00** | 123.00 | 400.0 | 0.8 | 0 |
| `random` (seed 7) | p = 0.5 per step | 129.70 | 129.70 | 400.0 | 2.8 | 0 |
| `always_acc` | Accelerates 100% of the time; exhausts at 66.7 s | 198.10 | 198.10 | 400.0 | 0.0 | 1315 |
| `sprint_rule` | Targets 7.0 m/s; exhausts early | 198.10 | 198.10 | 400.0 | 0.0 | 1315 |
| `trained` (fly) | Trained motor threshold (8.62e-6) | 198.10 | 198.10 | 400.0 | 0.0 | 1315 |
| `fly` (untrained) | Untrained threshold (0.015); coasts after 4 steps | 635.34 | 240.00 | 4.7 | 99.4 | 0 |
| `always_coast` | Never accelerates | 640.00 | 240.00 | 0.0 | 100.0 | 0 |

### Key observations
1. **Pacing beats flat-out sprinting:** `rule` cruises in 123.0 s with energy to spare, while `always_acc`
   wastes energy fighting high quadratic drag ($F_{drag} \propto v^2$) early on, exhausts its reserves,
   and spends over 130 seconds crawling under fatigue (198.10 s).
2. **Untrained fly result is genuine:** The untrained connectome controller accelerates for only 4 steps
   before interneuron inhibition settles the extensor minus flexor difference below the default 0.015 threshold.
   It coasts to 4.7 m and times out at 240 s (score 635.34).
3. **Explicit training improves performance:** Fitting the motor threshold via grid search (`experiments/train_pacing.py`)
   calibrates the decision threshold to `8.62e-6`, enabling the fly brain to complete the entire 400 m race (198.10 s).
