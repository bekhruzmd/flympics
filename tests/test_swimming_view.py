import matplotlib

matplotlib.use("Agg")

import pytest
from PIL import Image

from experiments.run_swimming import run
from visualization.swimming_view import (_glide_spans, _turn_time, animate_swimming,
                                         create_swimming_view, gif_frames, save_swimming_plot)


@pytest.fixture(scope="module")
def rule_run():
    return run("rule")


def test_plot_is_saved(rule_run, tmp_path):
    path = tmp_path / "race.png"
    save_swimming_plot(rule_run, path)
    with Image.open(path) as image:
        assert image.size == (1200, 750)  # 8 x 5 in at 150 dpi


def test_plot_marks_the_turn_and_the_glides(rule_run):
    spans = _glide_spans(rule_run)
    assert len(spans) == 2  # after the dive and after the turn
    turn = _turn_time(rule_run)
    assert spans[0][0] == 0.0 and spans[0][1] < turn <= spans[1][0]
    assert _turn_time(run("always_recover")) is None
    with pytest.raises(ValueError):
        save_swimming_plot([], "unused.png")


def test_gif_is_five_times_real_time(rule_run):
    frames = gif_frames(rule_run)
    assert frames[0] == 0 and frames[-1] == len(rule_run) - 1
    # dt = 0.05 s, 10 frames per second: 10 steps (0.5 s of race) per frame.
    assert frames[1:3] == [9, 19]


def test_viewer_follows_the_fly_out_and_back(rule_run):
    fig, update = create_swimming_view(rule_run, "rule")
    out, back = (next(i for i, r in enumerate(rule_run) if r["lap"] == lap and r["position"] > 25)
                 for lap in (1, 2))
    update(out)
    right, left = fig.axes[0].artists
    assert right.get_visible() and not left.get_visible()
    update(back)
    assert left.get_visible() and not right.get_visible()
    assert left.xy[0] == pytest.approx(100 - rule_run[back]["position"])
    matplotlib.pyplot.close(fig)


def test_short_gif_export(tmp_path):
    records = run("fly")[:40]  # 5 frames; a full race takes most of a minute
    path = tmp_path / "swim.gif"
    animate_swimming(records, "fly", show=False, output=path)
    with Image.open(path) as gif:
        assert gif.n_frames == len(gif_frames(records)) == 5
    assert path.with_suffix(".activity.png").is_file()
    with pytest.raises(ValueError):
        create_swimming_view([], "rule")
