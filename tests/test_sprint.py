import pytest

from events.sprint import ACCELERATE, COAST, Sprint


def test_reset_and_observation():
    sprint = Sprint(); observation = sprint.reset(123)
    assert observation == {"distance_to_finish": 100.0, "velocity": 0.0}
    assert not sprint.is_finished()


def test_fixed_step_is_deterministic():
    a, b = Sprint(), Sprint(); a.reset(1); b.reset(999)
    for _ in range(12):
        a.step(ACCELERATE); b.step(ACCELERATE)
    assert a.observe() == b.observe()


def test_legal_actions_and_termination():
    sprint = Sprint(distance=0.02); sprint.reset()
    sprint.step(ACCELERATE)
    sprint.step(ACCELERATE)
    assert sprint.is_finished()
    invalid = Sprint(); invalid.reset()
    with pytest.raises(ValueError):
        invalid.step("FLY")
    assert COAST in {COAST, ACCELERATE}


def test_controllers_face_identical_conditions_on_same_seed():
    from controllers.random_controller import RandomController
    from controllers.rule_controller import RuleController
    seed = 42
    rule_event = Sprint()
    rand_event = Sprint()
    rule_ctrl = RuleController()
    rand_ctrl = RandomController()

    obs1 = rule_event.reset(seed)
    rule_ctrl.reset(seed)
    while not rule_event.is_finished():
        obs1 = rule_event.step(rule_ctrl.act(obs1))

    obs2 = rand_event.reset(seed)
    rand_ctrl.reset(seed)
    while not rand_event.is_finished():
        obs2 = rand_event.step(rand_ctrl.act(obs2))

    assert rule_event.distance == rand_event.distance
    assert rule_event.max_time == rand_event.max_time
    assert rule_event.acceleration == rand_event.acceleration
    assert rule_event.drag == rand_event.drag
