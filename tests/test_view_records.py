import json
import sys

import pytest

from brain.connectome import load_manc_front_leg_circuit
from experiments import export_view, run_archery, run_hurdles, run_pacing, run_steeplechase, run_swimming
from experiments.serve_view import (
    ARCHERY_CONTROLLERS,
    HURDLES_CONTROLLERS,
    PACING_CONTROLLERS,
    STEEPLECHASE_CONTROLLERS,
    SWIMMING_CONTROLLERS,
    viewer_payload,
)
from events.archery import Archery
from events.hurdles import Hurdles
from events.pacing import Pacing400m
from events.steeplechase import Steeplechase
from events.swimming import Swimming


EVENTS = {
    "pacing": {
        "controllers": PACING_CONTROLLERS,
        "runner": run_pacing.run,
        "score": run_pacing.score,
        "event": Pacing400m,
    },
    "hurdles": {
        "controllers": HURDLES_CONTROLLERS,
        "runner": run_hurdles.run,
        "score": run_hurdles.score,
        "event": Hurdles,
    },
    "swimming": {
        "controllers": SWIMMING_CONTROLLERS,
        "runner": run_swimming.run,
        "score": run_swimming.score,
        "event": Swimming,
    },
    "steeplechase": {
        "controllers": STEEPLECHASE_CONTROLLERS,
        "runner": run_steeplechase.run,
        "score": run_steeplechase.score,
        "event": Steeplechase,
    },
    "archery": {
        "controllers": ARCHERY_CONTROLLERS,
        "runner": run_archery.run,
        "score": run_archery.score,
        "event": Archery,
    },
}


def without_activity(rows):
    return [{key: value for key, value in row.items() if key != "neuron_activity"} for row in rows]


def test_exported_view_payload_matches_runtime_view_payload(tmp_path, monkeypatch):
    path = tmp_path / "runs.json"
    monkeypatch.setattr(sys, "argv", ["export_view.py", "--seed", "7", "--output", str(path)])
    export_view.main()
    assert json.loads(path.read_text()) == viewer_payload(7)


@pytest.mark.parametrize("event_name", sorted(EVENTS))
def test_view_records_cover_every_event_controller_and_final_score(event_name):
    payload = viewer_payload(7)
    event_payload = payload["events"][event_name]
    spec = EVENTS[event_name]
    circuit = load_manc_front_leg_circuit()

    for controller in spec["controllers"]:
        rows = event_payload["runs"][controller]
        assert rows
        assert without_activity(rows) == spec["runner"](controller, 7)
        assert all(row["step"] == idx for idx, row in enumerate(rows, start=1))
        assert [row["simulation_time"] for row in rows]
        times = [row["simulation_time"] for row in rows]
        dt = spec["event"]().dt
        assert all(abs((b - a) - dt) < 1e-9 for a, b in zip(times, times[1:]))
        assert spec["score"](rows) == pytest.approx(spec["score"](rows))

        if "neuron_activity" in rows[0]:
            for row in rows:
                assert len(row["neuron_activity"]) == circuit.size
                assert len(row["neuron_activity"]) == len(circuit.neuron_ids)
                assert all(isinstance(value, (int, float)) for value in row["neuron_activity"])
              
    final_record = event_payload["runs"][spec["controllers"][0]][-1]
    assert spec["score"](event_payload["runs"][spec["controllers"][0]]) == pytest.approx(
        spec["score"](event_payload["runs"][spec["controllers"][0]])
    )
    if event_name == "archery":
        assert final_record["total_points"] == pytest.approx(spec["score"](event_payload["runs"][spec["controllers"][0]]))
