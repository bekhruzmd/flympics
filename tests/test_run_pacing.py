import csv
import json
import sys

import pytest

from experiments import run_pacing
from experiments.run_pacing import run, score

COLUMNS = ["step", "simulation_time", "position", "velocity", "energy",
           "distance_to_finish", "action", "exhausted"]


def run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pacing", *args])
    run_pacing.main()


def test_a_run_is_logged_to_csv_and_json(monkeypatch, tmp_path, capsys):
    output = tmp_path / "results"
    run_cli(monkeypatch, "--controller", "rule", "--seed", "3", "--output-dir", str(output))

    stem = output / "pacing_rule_seed3"
    json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    assert json_path.is_file() and csv_path.is_file()

    logged = json.loads(json_path.read_text())
    expected = run("rule", seed=3)
    assert logged == expected
    assert [row["step"] for row in logged] == list(range(1, len(logged) + 1))

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == COLUMNS
    assert len(rows) == len(expected)
    final = rows[-1]
    assert float(final["position"]) == 400.0
    assert float(final["simulation_time"]) == pytest.approx(score(expected))

    printed = capsys.readouterr().out
    assert "score 123.00" in printed and str(stem) in printed


def test_fly_run_logs_neural_activity(monkeypatch, tmp_path):
    run_cli(monkeypatch, "--controller", "fly", "--output-dir", str(tmp_path))
    with (tmp_path / "pacing_fly_seed7.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    for col in ("sensory_mean", "intrinsic_mean", "motor_extensor", "motor_flexor", "motor_difference"):
        assert col in reader.fieldnames and first[col] != ""


def test_no_plot_skips_png(monkeypatch, tmp_path):
    run_cli(monkeypatch, "--controller", "rule", "--no-plot", "--output-dir", str(tmp_path))
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".csv", ".json"]


def test_gif_flag_invokes_animation(monkeypatch, tmp_path):
    calls = []
    import visualization.pacing_view as view
    monkeypatch.setattr(view, "animate_pacing",
                        lambda records, name, *, show, output: calls.append((records, name, show, output)))
    run_cli(monkeypatch, "--controller", "rule", "--gif", "--no-plot", "--output-dir", str(tmp_path))
    (records, name, show, output), = calls
    assert records == run("rule") and name == "rule" and show is False
    assert output == tmp_path / "pacing_rule_seed7.gif"
