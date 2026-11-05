import statistics

import pytest

from controllers.swim_baselines import (AlwaysController, ChoppyController, RandomController,
                                        SwimRuleController)
from events.swimming import PULL, RECOVER, Swimming


def race(controller, seed: int = 0) -> Swimming:
    event = Swimming()
    event.reset(seed)
    controller.reset(seed)
    while not event.is_finished():
        event.step(controller.act(event.observe()))
    return event


def test_ranking():
    rule = race(SwimRuleController()).score()
    full = race(SwimRuleController(v_break=None)).score()
    random_mean = statistics.mean(race(RandomController(), seed).score() for seed in range(20))
    choppy = race(ChoppyController()).score()
    always_pull = race(AlwaysController(PULL))
    always_recover = race(AlwaysController(RECOVER))
    assert rule < full < random_mean < choppy < 120.0
    assert always_pull.position < 100.0 and always_pull.score() > 120.0
    assert always_recover.position < 100.0 and always_recover.score() > 120.0


def test_rule_scores():
    assert race(SwimRuleController()).score() == pytest.approx(44.15)
    assert race(SwimRuleController(v_break=None)).score() == pytest.approx(45.35)


def test_always_pull_makes_one_stroke():
    event = race(AlwaysController(PULL))
    assert event.strokes == 1
    assert event.stroke_phase == 1.0


def test_rule_strokes_use_the_full_range():
    event, controller = Swimming(), SwimRuleController(v_break=None)
    event.reset(); controller.reset()
    phases = []
    for _ in range(60):
        phases.append(event.observe()["stroke_phase"])
        event.step(controller.act(event.observe()))
    assert max(phases) == 1.0 and min(phases[1:]) == 0.0
    assert event.strokes == 3  # 20-step cycle: 12 pull + 8 recover


def test_rule_glides_after_the_dive_and_the_turn():
    event, controller = Swimming(), SwimRuleController(v_break=2.6)
    event.reset(); controller.reset()
    glides_by_lap = {1: 0, 2: 0}
    while not event.is_finished():
        lap = event.lap
        event.step(controller.act(event.observe()))
        glides_by_lap[lap] += event.gliding
    assert glides_by_lap[1] > 0 and glides_by_lap[2] > 0


def test_random_is_seeded():
    assert race(RandomController(), 3).score() == race(RandomController(), 3).score()


def test_bad_arguments():
    with pytest.raises(ValueError):
        AlwaysController("KICK")
    with pytest.raises(ValueError):
        RandomController(p=1.5)
    with pytest.raises(ValueError):
        SwimRuleController(v_break=-1.0)
