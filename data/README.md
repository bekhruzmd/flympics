# Data used by the MVP

`manc_front_leg_v1/` is a compact, derived subset of the **Male Adult Nerve
Cord (MANC) v1.2.1** connectome. It contains 135 annotated neurons and 2,770
directed, weighted connections. It is intentionally small enough for a laptop
and for students to inspect.

## Biological data

MANC reconstructs the adult male Drosophila ventral nerve cord. Its annotations
include sensory neurons, VNC intrinsic neurons, descending neurons, and motor
neurons. This MVP chooses annotated front-leg proprioceptive sensory neurons and
annotated front-leg tibia extensor/flexor motor neurons because they provide an
honest sensory-to-motor *structural* path within a motor-relevant connectome.
It does **not** model a whole brain or claim a sprint circuit exists in a fly.

## Exact extraction

The public raw files are at:

`https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/`

Download `manc_121_meta.feather` and `manc_121_simple_edgelist.feather`, then
run:

```bash
python scripts/extract_manc_front_leg.py /path/to/raw-manc-files
```

The deterministic rule is in the script and the input MD5 hashes are in
`manc_front_leg_v1/MANIFEST.txt`. Edges retain their synapse-count-derived
normalization, then are renormalized within the selected subgraph for stable
rate dynamics. This is a modeling transformation, not a new biological
measurement.

MANC data are released under **CC BY 4.0**. Preserve attribution to the Male
CNS/MANC collaboration and the source papers when redistributing derived data.
