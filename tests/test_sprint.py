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
