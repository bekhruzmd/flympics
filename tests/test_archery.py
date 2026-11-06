import pytest

from controllers.archery_baselines import AlwaysController, ArcheryRuleController, RandomController
from events.archery import HOLD, LEFT, RELEASE, RIGHT, Archery
from experiments.run_archery import run, score, summary


def play(event, controller, seed=0):
    observation = event.reset(seed)
    controller.reset(seed)
    while not event.is_finished():
        observation = event.step(controller.act(observation))
    return event


def test_reset_observation_and_illegal_actions():
    event = Archery()
    observation = event.reset(3)
    assert observation["aim_offset"] == 0.0 and observation["draw"] == 0.0
    assert observation["arrows_left"] == 6.0
    assert abs(observation["wind"]) <= event.wind_max
    with pytest.raises(ValueError):
        event.step("SHOOT")


def test_release_is_ignored_until_the_bow_is_drawn():
    event = Archery()
    event.reset(0)
    for _ in range(round(event.draw_time / event.dt)):
        event.step(RELEASE)
    assert event.points == [] and event.draw == 1.0
    event.step(RELEASE)
    assert len(event.points) == 1 and event.arrow == 1
    assert event.observe()["draw"] == 0.0  # the next arrow starts undrawn


def test_aim_moves_and_is_clipped():
    event = Archery()
    event.reset(0)
    event.step(RIGHT)
    assert event.aim == pytest.approx(event.aim_rate * event.dt)
    event.step(LEFT); event.step(LEFT)
    assert event.aim == pytest.approx(-event.aim_rate * event.dt)
    for _ in range(30):
        event.step(LEFT)
    assert event.aim >= -event.max_aim


def test_points_by_ring():
    event = Archery()
    assert event.points_for(0.0) == 10
    assert event.points_for(-0.06) == 10
    assert event.points_for(0.062) == 9
    assert event.points_for(5.0) == 0
    assert event.points_for(None) == 0


def test_holding_times_every_arrow_out():
    event = play(Archery(), AlwaysController(HOLD))
    assert event.points == [0] * 6 and event.landings == [None] * 6
    assert event.time == pytest.approx(6 * event.shot_clock)
    with pytest.raises(RuntimeError):
        event.step(HOLD)


def test_same_seed_same_end_and_seed_changes_wind():
    a, b = play(Archery(), RandomController(), 4), play(Archery(), RandomController(), 4)
    assert a.landings == b.landings
    assert Archery().reset(1)["wind"] != Archery().reset(2)["wind"]


def test_aiming_against_the_wind_beats_ignoring_it():
    seeds = range(30)
    rule = [play(Archery(), ArcheryRuleController(), s).score() for s in seeds]
    centre = [play(Archery(), AlwaysController(RELEASE), s).score() for s in seeds]
    assert sum(rule) / len(rule) > 58
    assert sum(rule) > sum(centre) + 200


def test_run_records_and_score():
    records = run("rule", seed=7)
    shots = [r for r in records if r["shot"]]
    assert len(shots) == 6 and [r["arrow"] for r in shots] == [1, 2, 3, 4, 5, 6]
    assert score(records) == sum(r["points"] for r in shots)
    assert summary(records)["timeouts"] == 0
    assert run("rule", seed=7) == records
