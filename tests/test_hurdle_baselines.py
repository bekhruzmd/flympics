import statistics

import pytest

from controllers.hurdle_baselines import AlwaysController, HurdleRuleController, RandomJumpController
from events.hurdles import BENCHMARK_LAYOUT, JUMP, RUN
from experiments.run_hurdles import CONTROLLERS, run, score

# Best (lowest) score first: timing beats not jumping, and untimed jumping is worse than none.
RANKING = ("rule", "rule1", "always_run", "random", "always_jump")


def test_benchmark_ranking():
    scores = [score(run(name, seed=7)) for name in RANKING]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)


def test_seeded_ranking():
    means = [statistics.mean(score(run(name, seed, "seeded")) for seed in range(50)) for name in RANKING]
    assert means == sorted(means) and len(set(means)) == len(means)


def test_benchmark_needs_the_second_look_ahead():
    assert run("rule", seed=7)[-1]["hits"] == 0
    assert run("rule1", seed=7)[-1]["hits"] >= 1
    assert run("always_run", seed=7)[-1]["hits"] == len(BENCHMARK_LAYOUT)


def test_runs_are_reproducible_and_legal():
    for name in CONTROLLERS:
        records = run(name, seed=3, layout="seeded")
        assert records == run(name, seed=3, layout="seeded")
        assert all(row["action"] in {RUN, JUMP} for row in records)
        assert records[-1]["simulation_time"] <= 90.0 + 1e-9
    assert run("random", seed=3) != run("random", seed=4)


def test_runner_rejects_unknown_layout():
    with pytest.raises(ValueError):
        run("rule", layout="olympic")


def test_controller_arguments_are_checked():
    with pytest.raises(ValueError):
        AlwaysController("ACCELERATE")
    with pytest.raises(ValueError):
        RandomJumpController(p=1.5)
    with pytest.raises(ValueError):
        HurdleRuleController(look_ahead=3)


def test_rule_window_matches_the_jump_arc():
    rule = HurdleRuleController()
    # Hurdle at half the jump height: the arc is above it from 14.6% to 85.4% of the jump.
    assert rule.window == pytest.approx((0.1464, 0.8536), abs=1e-4)
    far = {"distance_to_next_hurdle": 10.0, "distance_to_second_hurdle": 30.0, "velocity": 7.0}
    pair = {"distance_to_next_hurdle": 10.0, "distance_to_second_hurdle": 11.4, "velocity": 7.0}
    # A pair shrinks the window from the far side, so the take-off moves closer to the hurdle.
    assert rule.take_off_distance(pair) == pytest.approx(rule.take_off_distance(far) - 0.7)
    assert HurdleRuleController(look_ahead=1).take_off_distance(pair) == rule.take_off_distance(far)
