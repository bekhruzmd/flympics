import csv
import json
import sys

import pytest

from experiments import run_swimming
from experiments.run_swimming import run, score

COLUMNS = ["step", "simulation_time", "position", "lap", "velocity", "distance_to_wall",
           "stroke_phase", "grip", "gliding", "strokes", "turns", "action"]


def run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_swimming", *args])
    run_swimming.main()


def test_a_run_is_logged_to_csv_and_json(monkeypatch, tmp_path, capsys):
    output = tmp_path / "results"  # does not exist yet: main() must create it
    run_cli(monkeypatch, "--controller", "rule", "--seed", "3", "--output-dir", str(output))

    stem = output / "swimming_rule_seed3"
    json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    assert json_path.is_file() and csv_path.is_file()

    # The JSON log is exactly the run, one record per step.
    logged = json.loads(json_path.read_text())
    expected = run("rule", seed=3)
    assert logged == expected
    assert [row["step"] for row in logged] == list(range(1, len(logged) + 1))

    # The CSV has the same rows and the documented columns.
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == COLUMNS
    assert len(rows) == len(expected)
    final = rows[-1]
    assert float(final["position"]) == 100.0
    assert float(final["simulation_time"]) == pytest.approx(score(expected))
    assert final["turns"] == "1"
    assert {row["action"] for row in rows} == {"PULL", "RECOVER"}

    # The summary line points to the logs it wrote.
    printed = capsys.readouterr().out
    assert "score 44.15" in printed and str(stem) in printed


def test_fly_run_logs_the_motor_readout(monkeypatch, tmp_path):
    run_cli(monkeypatch, "--controller", "fly", "--output-dir", str(tmp_path))
    with (tmp_path / "swimming_fly_seed7.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    assert reader.fieldnames[:len(COLUMNS)] == COLUMNS
    for column in ("sensory_mean", "intrinsic_mean", "motor_extensor", "motor_flexor", "motor_difference"):
        assert column in reader.fieldnames and first[column] != ""


def test_a_second_run_overwrites_the_same_log(monkeypatch, tmp_path):
    for _ in range(2):
        run_cli(monkeypatch, "--controller", "choppy", "--output-dir", str(tmp_path))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "swimming_choppy_seed7.csv", "swimming_choppy_seed7.json", "swimming_choppy_seed7.png"]
    assert len(json.loads((tmp_path / "swimming_choppy_seed7.json").read_text())) == len(run("choppy"))


def test_no_plot_skips_the_png(monkeypatch, tmp_path):
    run_cli(monkeypatch, "--controller", "rule", "--no-plot", "--output-dir", str(tmp_path))
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".csv", ".json"]


def test_gif_flag_hands_the_logged_run_to_the_animation(monkeypatch, tmp_path):
    # The real export takes most of a minute; check the wiring, and test the GIF itself
    # on a short run in test_swimming_view.py.
    calls = []
    import visualization.swimming_view as view
    monkeypatch.setattr(view, "animate_swimming",
                        lambda records, name, *, show, output: calls.append((records, name, show, output)))
    run_cli(monkeypatch, "--controller", "rule", "--gif", "--no-plot", "--output-dir", str(tmp_path))
    (records, name, show, output), = calls
    assert records == run("rule") and name == "rule" and show is False
    assert output == tmp_path / "swimming_rule_seed7.gif"
