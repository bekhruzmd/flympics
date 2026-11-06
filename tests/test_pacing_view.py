import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from PIL import Image
import pytest

from experiments.run_pacing import run
from visualization.pacing_view import animate_pacing, create_pacing_view, save_pacing_plot


@pytest.fixture(scope="module")
def rule_run():
    return run("rule")


def test_plot_is_saved(rule_run, tmp_path):
    path = tmp_path / "race.png"
    save_pacing_plot(rule_run, path)
    assert path.is_file()
    with Image.open(path) as img:
        assert img.size == (1350, 1050)  # 9 x 7 in at 150 dpi


def test_empty_records_raise():
    with pytest.raises(ValueError, match="needs at least one timestep"):
        save_pacing_plot([], "unused.png")
    with pytest.raises(ValueError, match="needs at least one timestep"):
        create_pacing_view([], "rule")


def test_viewer_creation_and_update(rule_run):
    fig, update = create_pacing_view(rule_run, "rule")
    artists = update(10)
    assert len(artists) > 0
    matplotlib.pyplot.close(fig)


def test_short_gif_export(tmp_path):
    records = run("rule")[:20]
    path = tmp_path / "pacing.gif"
    animate_pacing(records, "rule", show=False, output=path)
    assert path.is_file()
    assert path.with_suffix(".activity.png").is_file()
