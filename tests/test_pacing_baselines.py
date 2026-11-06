import pytest

from controllers.pacing_baselines import (AlwaysAccelerateController, AlwaysCoastController,
                                          PacingRuleController, RandomPacingController,
                                          SprintPacingController)
from events.pacing import Pacing400m


def race(controller, seed: int = 7) -> Pacing400m:
    event = Pacing400m()
    event.reset(seed)
    controller.reset(seed)
    while not event.is_finished():
        event.step(controller.act(event.observe()))
    return event


def test_baseline_ranking():
    rule = race(PacingRuleController(target_velocity=3.3))
    always_acc = race(AlwaysAccelerateController())
    always_coast = race(AlwaysCoastController())
    sprint_rule = race(SprintPacingController())

    assert rule.score() < always_acc.score()
    assert rule.score() == pytest.approx(123.0)
    assert rule.energy > 0.0

    assert always_acc.score() == pytest.approx(198.1)
    assert always_acc.energy == 0.0
    assert always_acc.exhausted_steps > 1000

    assert sprint_rule.score() == pytest.approx(198.1)
    assert sprint_rule.energy == 0.0

    assert always_coast.position == 0.0
    assert always_coast.score() == pytest.approx(640.0)


def test_random_controller_is_seeded():
    r1 = race(RandomPacingController(), seed=42)
    r2 = race(RandomPacingController(), seed=42)
    r3 = race(RandomPacingController(), seed=99)
    assert r1.score() == r2.score()
    assert r1.steps == r2.steps
    assert r1.score() != r3.score()


def test_controller_invalid_arguments():
    with pytest.raises(ValueError, match="target_velocity must be positive"):
        PacingRuleController(target_velocity=-1.0)
    with pytest.raises(ValueError, match="Probability p must be between 0 and 1"):
        RandomPacingController(p=-0.1)
    with pytest.raises(ValueError, match="Probability p must be between 0 and 1"):
        RandomPacingController(p=1.5)
