# Circuit saturation across events

The saturation report compares the untrained MANC rate-network controller across Sprint,
Hurdles, Swimming, Pacing, Steeplechase, and Archery at gains `0.5, 1, 1.5, 2, 3`.
Run one event with:

```powershell
.venv/Scripts/python -m experiments.saturation_report --event sprint
```

Each run writes `results/saturation/<event>.csv` and `.png` and prints a results table.

## Measurements

- At every race step, the report records each group's mean activity and the percentage of
  sensory, intrinsic, extensor, and flexor neurons with activity **greater than 0.95**.
  It averages these percentages over the race.
- For each encoder input, it compares the settled extensor-minus-flexor readout at that
  input's low and high values, holding other inputs at their midpoints. The network resets
  and settles for 50 steps for each value. Archery aim and wind use `[-1, 1]`; draw and
  other event inputs use `[0, 1]`.
- Each event's untrained controller runs a fixed setup (Hurdles uses its benchmark layout;
  Archery uses seed 7). The score follows that event's scoring rule. Archery is higher-is-
  better; other events are lower-is-better.

## Default gain (3.0)

Percentages below are race averages: within each group, the share of neurons above 0.95.
Scores have different units and should only be compared within their event.

| Event | Sensory % | Intrinsic % | Extensor % | Flexor % | Largest input effect | Score |
|---|---:|---:|---:|---:|---:|---:|
| Sprint | 74.6 | 99.2 | 99.2 | 99.2 | 3.13e-6 | 156.9 |
| Hurdles | 74.9 | 99.7 | 99.7 | 99.7 | 1.98e-6 | 288.0 |
| Swimming | 74.9 | 99.8 | 99.8 | 99.8 | 1.98e-6 | 201.2 |
| Pacing | 74.9 | 99.8 | 99.8 | 99.8 | 1.98e-6 | 635.3 |
| Steeplechase | 99.9 | 99.8 | 99.8 | 99.8 | 7.93e-7 | 836.9 |
| Archery | 70.3 | 91.3 | 92.2 | 91.1 | 1.94e-5 | 47 / 60 |

Across all six events, input effects are smaller at gain 3 than at gain 0.5. Intrinsic and
motor groups are almost entirely above 0.95 in five events. Archery remains less saturated
by this threshold and its score rises with gain, so saturation does not imply the same
behavioral outcome for every decoder or event.

## Results

| Event | Plot | Full table (CSV) |
|---|---|---|
| Sprint | [PNG](../results/saturation/sprint.png) | [CSV](../results/saturation/sprint.csv) |
| Hurdles | [PNG](../results/saturation/hurdles.png) | [CSV](../results/saturation/hurdles.csv) |
| Swimming | [PNG](../results/saturation/swimming.png) | [CSV](../results/saturation/swimming.csv) |
| Pacing | [PNG](../results/saturation/pacing.png) | [CSV](../results/saturation/pacing.csv) |
| Steeplechase | [PNG](../results/saturation/steeplechase.png) | [CSV](../results/saturation/steeplechase.csv) |
| Archery | [PNG](../results/saturation/archery.png) | [CSV](../results/saturation/archery.csv) |

## Limits

These are modeled activity values, not recorded biological firing rates. The low/high
input test uses constant stimuli and one 50-step settling period; real event inputs change
over time. One fixed run is reported per gain, so stochastic variation is not estimated.
