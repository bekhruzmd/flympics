import matplotlib

matplotlib.use("Agg")

import pytest
from PIL import Image

from events.hurdles import BENCHMARK_LAYOUT, generate_layout
from experiments.run_hurdles import course, run
from visualization.hurdles_view import (air_spans, animate_hurdles, coast_spans, create_hurdles_view,
                                        gif_frames, hurdle_results, save_hurdles_plot)


@pytest.fixture(scope="module")
def rule_run():
    return run("rule")  # benchmark: clears every hurdle, coasts before each long gap ends


@pytest.fixture(scope="module")
def accelerate_run():
    return run("always_accelerate")  # benchmark: hits every hurdle


def test_course_matches_the_race():
    assert course() == BENCHMARK_LAYOUT
    assert course("seeded", 5) == generate_layout(5)
    with pytest.raises(ValueError):
        course("olympic")


def test_plot_is_saved(rule_run, tmp_path):
    path = tmp_path / "race.png"
    save_hurdles_plot(rule_run, BENCHMARK_LAYOUT, path)
    with Image.open(path) as image:
        assert image.size == (1200, 750)  # 8 x 5 in at 150 dpi
    with pytest.raises(ValueError):
        save_hurdles_plot([], BENCHMARK_LAYOUT, tmp_path / "unused.png")


def test_hits_are_found_from_the_log(rule_run, accelerate_run):
    assert [hit for _, hit in hurdle_results(rule_run, BENCHMARK_LAYOUT)] == [False] * len(BENCHMARK_LAYOUT)
    results = hurdle_results(accelerate_run, BENCHMARK_LAYOUT)
    assert [hit for _, hit in results] == [True] * len(BENCHMARK_LAYOUT)
    # Each hurdle is recorded at the step that passed it.
    for (index, _), hurdle in zip(results, BENCHMARK_LAYOUT):
        assert accelerate_run[index - 1]["position"] < hurdle <= accelerate_run[index]["position"]


def test_jump_and_coast_spans_match_the_counters(rule_run):
    dt = rule_run[0]["simulation_time"]
    assert len(air_spans(rule_run)) == rule_run[-1]["jumps"]
    coasting = sum(end - start for start, end in coast_spans(rule_run))
    assert coasting == pytest.approx(rule_run[-1]["coasts"] * dt)
    assert coast_spans(run("nocoast")) == []


def test_gif_is_two_and_a_half_times_real_time(rule_run):
    frames = gif_frames(rule_run)
    assert frames[0] == 0 and frames[-1] == len(rule_run) - 1
    # dt = 0.05 s, 10 frames per second: 5 steps (0.25 s of race) per frame.
    assert frames[1:3] == [4, 9]


@pytest.mark.parametrize("name, colour", [("rule", "#3d946d"), ("always_accelerate", "#c0392b")])
def test_viewer_draws_jumps_and_hurdle_outcomes(name, colour, rule_run, accelerate_run):
    records = rule_run if name == "rule" else accelerate_run
    fig, update = create_hurdles_view(records, BENCHMARK_LAYOUT, name)
    track = fig.axes[0]
    athlete, hurdles = track.artists[0], track.collections[0]
    if name == "rule":
        jump = next(i for i, row in enumerate(records) if row["height"] > 0.9)
        update(jump)
        assert athlete.xy == pytest.approx((records[jump]["position"], 0.15 + records[jump]["height"]))
        assert "AIR" in track.texts[-1].get_text()
    update(len(records) - 1)
    colours = [tuple(c) for c in hurdles.get_colors()]
    assert colours == [matplotlib.colors.to_rgba(colour)] * len(BENCHMARK_LAYOUT)
    matplotlib.pyplot.close(fig)


def test_short_gif_export(tmp_path):
    records = run("fly")[:20]  # 5 frames; a full race takes about a minute
    path = tmp_path / "hurdles.gif"
    animate_hurdles(records, BENCHMARK_LAYOUT, "fly", show=False, output=path)
    with Image.open(path) as gif:
        assert gif.n_frames == len(gif_frames(records)) == 5
    assert path.with_suffix(".activity.png").is_file()
    with pytest.raises(ValueError):
        create_hurdles_view([], BENCHMARK_LAYOUT, "rule")
