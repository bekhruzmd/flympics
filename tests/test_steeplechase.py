import pytest

from controllers.steeple_baselines import AlwaysController, RandomController, SteepleRuleController
from events.steeplechase import ACCELERATE, BARRIERS, COAST, JUMP, WATER, Steeplechase
from experiments.run_steeplechase import run, score


def play(event, controller, seed=0):
    observation = event.reset(seed)
    controller.reset(seed)
    while not event.is_finished():
        observation = event.step(controller.act(observation))
    return event


def test_course_and_first_observation():
    assert len(BARRIERS) == 12 and len(WATER) == 4
    observation = Steeplechase().reset()
    assert observation == {"distance_to_barrier": 25.0, "water_next": 0.0, "velocity": 0.0,
                           "stamina": 1.0, "distance_to_finish": 600.0}


def test_illegal_action():
    event = Steeplechase()
    event.reset()
    with pytest.raises(ValueError):
        event.step("SPRINT")


def test_accelerating_drains_and_coasting_recovers_stamina():
    event = Steeplechase()
    event.reset()
    for _ in range(100):
        event.step(ACCELERATE)
    tired = event.stamina
    assert tired < 1.0
    for _ in range(20):
        event.step(COAST)
    assert event.stamina > tired


def test_never_jumping_hits_every_barrier_and_splashes_in_every_pit():
    event = play(Steeplechase(), AlwaysController(ACCELERATE))
    assert event.hits == 12 and event.clears == 0
    assert event.splashes == 4 and event.dry_pits == 0
    assert event.stamina == 0.0


def test_coasting_never_starts():
    event = play(Steeplechase(), AlwaysController(COAST))
    assert event.position == 0.0 and event.score() == pytest.approx(240.0 + 600.0)


def test_running_into_the_pit_after_a_hit_splashes():
    event = Steeplechase()
    event.reset()
    pit = BARRIERS[min(WATER)]
    event.position, event.velocity, event._next, event._pit = pit - 0.1, 3.0, min(WATER), 0
    event.step(COAST)
    assert event.hits == 1 and event.splashes == 1


def test_a_long_jump_lands_past_the_pit():
    event = Steeplechase()
    event.reset()
    pit = BARRIERS[min(WATER)]
    # Fast take-off 1 m before the barrier: the arc is high over it and lands > 3 m past it.
    event.position, event.velocity, event._next, event._pit = pit - 1.0, 7.5, min(WATER), 0
    event.step(JUMP)
    while event.airborne:
        event.step(COAST)
    assert event.clears == 1 and event.hits == 0
    assert event.dry_pits == 1 and event.splashes == 0


def test_rule_clears_everything_and_beats_the_baselines():
    rule = play(Steeplechase(), SteepleRuleController())
    assert rule.hits == 0 and rule.clears == 12 and rule.splashes == 0
    assert rule.score() < play(Steeplechase(), AlwaysController(ACCELERATE)).score()
    assert rule.score() < play(Steeplechase(), RandomController(), seed=7).score()


def test_run_records_and_score():
    records = run("rule")
    assert records[-1]["position"] == 600.0
    assert score(records) == pytest.approx(records[-1]["simulation_time"])
    assert run("rule", seed=1) == run("rule", seed=2)  # the course ignores the seed
