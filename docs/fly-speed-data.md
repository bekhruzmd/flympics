# Real fruit fly speeds, and how they map onto Flympics

The Flympics events are game models. This page collects published measurements of how
fast *Drosophila melanogaster* actually walks and flies, so we can say how far the game
is from the real animal and choose a realistic scale if we want one.

Nothing in the simulation uses these numbers yet.

## Measured speeds

| What | Speed | Source | Notes |
|---|---:|---|---|
| Free walking, per-fly average | 7.2 to 44.7 mm/s, most common about 28 mm/s | Mendes et al. 2013, *eLife* | 71 videos of wild-type adults walking straight; sex not reported. |
| Fastest walking ever reported | over 50 mm/s (instantaneous) | Pratt et al. 2024, *Current Biology* | Flies on a miniature treadmill sustained a 40 mm/s belt. The authors say flies may go faster; it was not tested systematically. |
| Free flight, preferred speed in still air | about 1 m/s | Leitch et al. 2021, *PNAS* | Flies released in the Mojave Desert; males and females together. |
| Free flight, first arrivals at traps | 1.4 ± 0.6 m/s | Leitch et al. 2021, *PNAS* | Average groundspeed over 30 traps, so it includes some wind help. |

**About males:** none of these studies reports a male-only top speed. An older genetics
study (Lee 1978, *Japanese Journal of Genetics*) found that males walked faster than
females on average. Males are also the smaller sex; females can be up to about 30% larger.

**Body length:** about 2.5 mm is used below. That is a common figure for the adult fly; a
male is usually a little smaller, so speeds in body lengths per second are, if anything,
slightly underestimated for males.

## In body lengths per second (2.5 mm body)

| Speed | mm/s | Body lengths/s |
|---|---:|---:|
| Slow free walking | 7.2 | 2.9 |
| Typical free walking | 28 | 11.2 |
| Fast free walking | 44.7 | 17.9 |
| Fastest recorded walking | 50 | 20 |
| Preferred flight | 1,000 | 400 |
| Fast flight (first arrivals) | 1,400 | 560 |

Flying is roughly 20 to 40 times faster than running, which is why a Fly mode should be
treated as a cheat in a running event.

## Converting to our races

### Option 1: the race distances are real metres

| Race | Typical walking (28 mm/s) | Fastest walking (50 mm/s) | Preferred flight (1 m/s) |
|---|---:|---:|---:|
| 100 m Sprint | about 60 min | about 33 min | about 1.7 min |
| 200 m Hurdles | about 2 h | about 67 min | about 3.3 min |

Also, a leg-powered hop lifts a fly only millimetres, so the 0.5 m hurdles in
`events/hurdles.py` could not be jumped on foot.

### Option 2: one sim metre is one fly body length (recommended)

This keeps every number in the code as it is and only changes how we describe it. The
100 m Sprint becomes 100 body lengths (25 cm) and the 200 m Hurdles 200 body lengths (50 cm).

| Race | Current sim (`rule`) | Typical walking | Fastest walking | Preferred flight |
|---|---:|---:|---:|---:|
| Sprint, 100 body lengths | 18.9 s | 8.9 s | 5.0 s | 0.25 s |
| Hurdles, 200 body lengths | 48.1 s | 17.9 s | 10.0 s | 0.5 s |

The sim's top speed is `acceleration / drag` = 1.25 / 0.16 ≈ 7.8 sim metres per second,
which in this reading is 7.8 body lengths/s, or about 20 mm/s. That is a slow-to-typical
walking fly, so the current game is already in the right range for a running fly, just on
the slow side. To match a typical walker, the top speed would need to rise to about 11
body lengths/s; to match the fastest recorded walking, about 20.

The 0.5 m hurdles become half a body length (about 1.25 mm) under this reading, which is
small enough for a leg-powered hop.

## Caveats

- All walking numbers come from short, straight walks in the lab, not 100-body-length races.
- Speeds depend on temperature, age, strain and how the fly is motivated.
- The flight numbers are groundspeeds outdoors and can include wind.
- The body-length conversion uses a single typical length, not a measured male length.

## Sources

- Mendes, C. S., Bartos, I., Akay, T., Márka, S., Mann, R. S. (2013). Quantification of gait
  parameters in freely walking wild type and sensory deprived *Drosophila melanogaster*.
  *eLife* 2:e00231. https://elifesciences.org/articles/00231
- Pratt, B. G., et al. (2024). Miniature linear and
  split-belt treadmills reveal mechanisms of adaptive motor control in walking
  *Drosophila*. *Current Biology*. https://doi.org/10.1016/j.cub.2024.08.006
- Leitch, K. J., Ponce, F. V., Dickson, W. B., van Breugel, F., Dickinson, M. H. (2021). The
  long-distance flight behavior of *Drosophila* supports an agent-based model for
  wind-assisted dispersal in insects. *PNAS* 118(17):e2013342118.
  https://doi.org/10.1073/pnas.2013342118
- Lee (1978). Genetic variations of walking and flying ability of *Drosophila
  melanogaster*. *Japanese Journal of Genetics* 53(5):327.
  https://www.jstage.jst.go.jp/article/ggs1921/53/5/53_5_327/_article/-char/en
