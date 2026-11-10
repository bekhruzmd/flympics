import pytest

from events.pacing import ACCELERATE, COAST, LEGAL_ACTIONS, Pacing400m


def fresh(**rules) -> Pacing400m:
    event = Pacing400m(**rules)
    event.reset()
    return event


def test_reset_and_observation():
    event = Pacing400m()
    obs = event.reset(42)
    assert obs == {
        "distance_to_finish": 400.0,
        "velocity": 0.0,
        "energy": 100.0,
    }
    assert event.position == 0.0
    assert event.velocity == 0.0
    assert event.energy == 100.0
    assert event.time == 0.0
    assert event.steps == 0
    assert not event.is_finished()


def test_fixed_step_is_deterministic():
    a = Pacing400m()
    b = Pacing400m()
    a.reset(10)
    b.reset(99)
    actions = [ACCELERATE, ACCELERATE, COAST, ACCELERATE, COAST] * 20
    for action in actions:
        a.step(action)
        b.step(action)
    assert a.observe() == b.observe()
    assert a.position == b.position
    assert a.velocity == b.velocity
    assert a.energy == b.energy
    assert a.steps == b.steps
    assert a.time == b.time


def test_illegal_action():
    event = fresh()
    with pytest.raises(ValueError, match="Illegal pacing action"):
        event.step("SPRINT")


def test_stepping_finished_race_raises():
    event = fresh(distance=0.01)
    event.step(ACCELERATE)
    assert event.is_finished()
    with pytest.raises(RuntimeError, match="Cannot step a finished race"):
        event.step(ACCELERATE)


def test_energy_burn_and_exhaustion():
    event = fresh(energy_budget=1.5, burn_rate=1.5, dt=0.1)  # 1.5 energy = 10 steps of burn
    for _ in range(10):
        event.step(ACCELERATE)
    assert event.energy == pytest.approx(0.0)
    assert event.exhausted_steps == 1

    # Next step is under fatigue
    v_before = event.velocity
    event.step(ACCELERATE)
    assert event.energy == 0.0
    assert event.exhausted_steps == 2
    drag = event.drag * v_before + event.drag_quad * v_before * v_before
    expected_v = max(0.0, v_before + event.dt * (event.fatigue_accel - drag))
    assert event.velocity == pytest.approx(expected_v)


def test_coasting_consumes_no_energy():
    event = fresh()
    event.step(ACCELERATE)
    energy_after_accel = event.energy
    for _ in range(15):
        event.step(COAST)
    assert event.energy == energy_after_accel
    assert event.coasts == 15


def test_drag_physics():
    event = fresh(energy_budget=50.0)
    event.velocity = 4.0
    v = event.velocity
    event.step(COAST)
    drag = event.drag * v + event.drag_quad * v * v
    expected = max(0.0, v - event.dt * drag)
    assert event.velocity == pytest.approx(expected)


def test_finish_and_score():
    event = fresh()
    while not event.is_finished():
        event.step(ACCELERATE if event.velocity < 3.3 else COAST)
    assert event.position == 400.0
    assert event.score() == pytest.approx(event.time)
    assert event.score() < event.max_time


def test_unfinished_race_score():
    event = fresh()
    while not event.is_finished():
        event.step(COAST)
    assert event.position == 0.0
    assert event.time == pytest.approx(event.max_time)
    assert event.score() == pytest.approx(event.max_time + event.distance)


def test_controllers_face_identical_conditions_on_same_seed():
    from controllers.pacing_baselines import PacingRuleController, RandomPacingController
    seed = 42
    rule_event = Pacing400m()
    rand_event = Pacing400m()
    rule_ctrl = PacingRuleController()
    rand_ctrl = RandomPacingController()

    obs1 = rule_event.reset(seed)
    rule_ctrl.reset(seed)
    while not rule_event.is_finished():
        obs1 = rule_event.step(rule_ctrl.act(obs1))

    obs2 = rand_event.reset(seed)
    rand_ctrl.reset(seed)
    while not rand_event.is_finished():
        obs2 = rand_event.step(rand_ctrl.act(obs2))

    assert rule_event.distance == rand_event.distance
    assert rule_event.energy_budget == rand_event.energy_budget
    assert rule_event.max_time == rand_event.max_time
    assert rule_event.acceleration == rand_event.acceleration
    assert rule_event.drag == rand_event.drag
    assert rule_event.drag_quad == rand_event.drag_quad
