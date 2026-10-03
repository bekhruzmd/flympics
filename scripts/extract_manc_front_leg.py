"""Create the small, deterministic MANC circuit shipped with Flympics.

This is data preparation, not a biological circuit-discovery method. It records the
selection rule so a student can regenerate the CSVs from the exact public release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd


def choose_top(scores: pd.Series, n: int) -> set[str]:
    return set(scores.sort_values(ascending=False, kind="stable").head(n).index.astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path, help="directory containing MANC v1.2.1 feather files")
    parser.add_argument("--output", type=Path, default=Path("data/manc_front_leg_v1"))
    args = parser.parse_args()
    meta = pd.read_feather(args.raw_dir / "manc_121_meta.feather")
    edges = pd.read_feather(args.raw_dir / "manc_121_simple_edgelist.feather")
    meta["manc_121_id"] = meta["manc_121_id"].astype(str)
    edges[["pre", "post"]] = edges[["pre", "post"]].astype(str)

    sensory_all = meta[(meta.super_class == "sensory") &
                       (meta.body_part_sensory == "front_leg") &
                       (meta.cell_function == "proprioception")]
    sensory = set(sensory_all.sort_values("manc_121_id").head(24).manc_121_id)
    intrinsic_all = set(meta.loc[meta.super_class == "ventral_nerve_cord_intrinsic", "manc_121_id"])
    first_hop = edges[edges.pre.isin(sensory) & edges.post.isin(intrinsic_all)]
    intrinsic = choose_top(first_hop.groupby("post")["count"].sum(), 96)

    motor = meta[(meta.super_class == "motor") & (meta.body_part_effector == "front_leg")]
    extensor_all = set(motor[motor.cell_type.str.contains("extensor", case=False, na=False)].manc_121_id)
    flexor_all = set(motor[motor.cell_type.str.contains("flexor", case=False, na=False)].manc_121_id)
    outgoing = edges[edges.pre.isin(intrinsic)]
    extensor = choose_top(outgoing[outgoing.post.isin(extensor_all)].groupby("post")["count"].sum(), 8)
    flexor = choose_top(outgoing[outgoing.post.isin(flexor_all)].groupby("post")["count"].sum(), 8)
    selected = sensory | intrinsic | extensor | flexor
    selected_meta = meta[meta.manc_121_id.isin(selected)].copy()
    roles = {node: "sensory" for node in sensory}
    roles.update({node: "intrinsic" for node in intrinsic})
    roles.update({node: "motor_extensor" for node in extensor})
    roles.update({node: "motor_flexor" for node in flexor})
    selected_meta["role"] = selected_meta.manc_121_id.map(roles)
    selected_meta = selected_meta.sort_values(["role", "manc_121_id"])
    # Existing 'norm' equals count / total postsynaptic input. Keep only selected
    # edges, then rescale by max selected incoming weight for stable rate dynamics.
    selected_edges = edges[edges.pre.isin(selected) & edges.post.isin(selected)].copy()
    selected_edges["weight"] = selected_edges["norm"]
    selected_edges["weight"] /= selected_edges.groupby("post").weight.transform("sum").clip(lower=1e-12)
    selected_edges = selected_edges[["pre", "post", "count", "weight"]].sort_values(["post", "pre"])
    args.output.mkdir(parents=True, exist_ok=True)
    selected_meta[["manc_121_id", "role", "cell_type", "neurotransmitter_predicted"]].to_csv(args.output / "neurons.csv", index=False)
    selected_edges.to_csv(args.output / "edges.csv", index=False, float_format="%.12g")
    source_hashes = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in
                     (args.raw_dir / "manc_121_meta.feather", args.raw_dir / "manc_121_simple_edgelist.feather")}
    with (args.output / "MANIFEST.txt").open("w") as f:
        f.write("MANC v1.2.1 compact front-leg circuit\n")
        f.write("Selection: first 24 sorted front_leg proprioceptive sensory neurons; top 96 VNC intrinsic targets by direct synapse count; up to 8 unique extensor and flexor front-leg motor neurons by selected-intrinsic input.\n")
        for name, digest in source_hashes.items(): f.write(f"{name} md5 {digest}\n")
        f.write(f"neurons {len(selected_meta)}; edges {len(selected_edges)}\n")


if __name__ == "__main__":
    main()
