import statistics

import pytest

from controllers.hurdle_baselines import AlwaysController, HurdleRuleController, RandomController
from events.hurdles import ACCELERATE, BENCHMARK_LAYOUT, COAST, JUMP, Hurdles
from experiments.run_hurdles import CONTROLLERS, run, score

# Best (lowest) score first: braking before hurdles beats jumping too fast, and both beat not
# jumping; always coasting and always jumping never move, so they tie last.
RANKING = ("rule", "nocoast", "always_accelerate", "random", "always_coast", "always_jump")


def assert_ranked(scores):
    *finishers, coast, jump = scores
    assert finishers == sorted(finishers) and len(set(finishers)) == len(finishers)
    assert finishers[-1] < coast == jump == 290.0


def test_benchmark_ranking():
    assert_ranked([score(run(name, seed=7)) for name in RANKING])


def test_seeded_ranking():
    assert_ranked([statistics.mean(score(run(name, seed, "seeded")) for seed in range(50)) for name in RANKING])


def test_rule_takes_off_near_the_safe_speed():
    speeds = []
    for seed in range(10):
        records = run("rule", seed, "seeded")
        speeds += [before["velocity"] for before, row in zip(records, records[1:]) if row["jumps"] > before["jumps"]]
    near = [abs(v - Hurdles().safe_speed) <= 0.3 for v in speeds]
    assert sum(near) >= 0.9 * len(near)


def test_rule_clears_everything_and_only_rule_coasts():
    rule, nocoast = run("rule", seed=7), run("nocoast", seed=7)
    assert rule[-1]["hits"] == nocoast[-1]["hits"] == 0
    assert rule[-1]["jumps"] == nocoast[-1]["jumps"] == len(BENCHMARK_LAYOUT)
    assert rule[-1]["coasts"] > 0 and nocoast[-1]["coasts"] == 0
    assert run("always_accelerate", seed=7)[-1]["hits"] == len(BENCHMARK_LAYOUT)


def test_runs_are_reproducible_and_legal():
    for name in CONTROLLERS:
        records = run(name, seed=3, layout="seeded")
        assert records == run(name, seed=3, layout="seeded")
        assert all(row["action"] in {ACCELERATE, COAST, JUMP} for row in records)
        assert records[-1]["simulation_time"] <= 90.0 + 1e-9
    assert run("random", seed=3) != run("random", seed=4)


def test_runner_rejects_unknown_layout():
    with pytest.raises(ValueError):
        run("rule", layout="olympic")


def test_controller_arguments_are_checked():
    with pytest.raises(ValueError):
        AlwaysController("RUN")
    with pytest.raises(ValueError):
        RandomController(p_jump=1.5)
    with pytest.raises(ValueError):
        RandomController(p_jump=0.6, p_coast=0.6)


def test_random_controller_mixes_actions():
    controller = RandomController(p_jump=0.2, p_coast=0.3); controller.reset(0)
    actions = [controller.act({}) for _ in range(2000)]
    assert actions.count(JUMP) / 2000 == pytest.approx(0.2, abs=0.03)
    assert actions.count(COAST) / 2000 == pytest.approx(0.3, abs=0.03)


def test_rule_window_and_braking():
    rule = HurdleRuleController()
    # Hurdle at half the jump height: the arc is above it from 14.6% to 85.4% of the jump.
    assert rule.window == pytest.approx((0.1464, 0.8536), abs=1e-4)
    # Faster than safe_speed, the take-off cost shortens the jump.
    assert rule.take_off_distance(6.0) < rule.take_off_distance(4.5)
    # Far from the hurdle it accelerates, inside its braking distance it coasts, then it jumps.
    assert rule.act({"distance_to_next_hurdle": 30.0, "velocity": 6.0}) == ACCELERATE
    assert rule.act({"distance_to_next_hurdle": 8.0, "velocity": 6.0}) == COAST
    assert rule.act({"distance_to_next_hurdle": 1.5, "velocity": 4.5}) == JUMP
    assert HurdleRuleController(coast=False).act({"distance_to_next_hurdle": 8.0, "velocity": 6.0}) == ACCELERATE


def test_rule_follows_the_event_it_is_given():
    strict = Hurdles(safe_speed=3.5)
    rule = HurdleRuleController(event=strict)
    # Braking from 4.5 to 3.5 m/s takes 6.25 m, so at 6 m it is already braking.
    assert rule.act({"distance_to_next_hurdle": 6.0, "velocity": 4.5}) == COAST
    assert HurdleRuleController().act({"distance_to_next_hurdle": 6.0, "velocity": 4.5}) == ACCELERATE
