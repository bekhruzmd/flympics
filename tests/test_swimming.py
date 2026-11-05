import pytest

from events.swimming import PULL, RECOVER, Swimming


def fresh(**rules) -> Swimming:
    event = Swimming(**rules)
    event.reset()
    return event


def run_actions(event: Swimming, actions) -> None:
    for action in actions:
        event.step(action)


def test_reset_observation():
    event = Swimming()
    assert event.reset() == {"distance_to_wall": 50.0, "velocity": 4.0, "stroke_phase": 0.0}
    assert (event.lap, event.turns, event.strokes) == (1, 0, 0)


def test_seed_does_not_change_the_race():
    races = []
    for seed in (None, 0, 1, 99):
        event = Swimming()
        event.reset(seed)
        run_actions(event, ([PULL] * 12 + [RECOVER] * 8) * 20)
        races.append((event.position, event.velocity, event.strokes))
    assert len(set(races)) == 1


def test_grip_builds_over_five_steps_and_resets_on_recover():
    event = fresh()
    grips = []
    for _ in range(7):
        event.step(PULL)
        grips.append(event.grip)
    assert grips == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])
    event.step(RECOVER)
    assert event.grip == 0.0
    event.step(PULL)
    assert event.grip == pytest.approx(0.2)
    assert event.strokes == 2


def test_thrust_matches_the_formula():
    event = fresh(dive_speed=2.0)
    run_actions(event, [PULL] * 5)
    v = event.velocity
    event.step(PULL)  # full grip
    expected = v + event.dt * (5.0 * (1 - v / 4.0) - 0.08 * v * v - 0.15 * v)
    assert event.velocity == pytest.approx(expected)


def test_no_thrust_faster_than_hand_speed():
    event = fresh(dive_speed=5.0)
    run_actions(event, [PULL] * 5)
    v = event.velocity
    event.step(PULL)
    assert event.velocity == pytest.approx(v + event.dt * -(0.08 * v * v + 0.15 * v))


def test_arm_ranges_are_whole_steps():
    event = fresh()
    run_actions(event, [PULL] * 11)
    assert event.stroke_phase < 1.0
    event.step(PULL)
    assert event.stroke_phase == 1.0
    run_actions(event, [RECOVER] * 7)
    assert event.stroke_phase > 0.0
    event.step(RECOVER)
    assert event.stroke_phase == 0.0


def test_pulling_at_the_end_of_the_stroke_gives_no_thrust():
    event = fresh(dive_speed=2.0)
    run_actions(event, [PULL] * 12)
    v = event.velocity
    event.step(PULL)
    assert event.grip == 0.0
    assert event.stroke_phase == 1.0
    assert event.velocity == pytest.approx(v + event.dt * -(0.08 * v * v + 0.15 * v))


def test_glide_loses_less_speed_than_recovery():
    glide = fresh(dive_speed=3.0)
    glide.step(RECOVER)
    assert glide.gliding
    assert glide.velocity == pytest.approx(3.0 - 0.05 * (0.02 * 9 + 0.15 * 3))

    recover = fresh(dive_speed=2.0)
    run_actions(recover, [PULL] * 12)
    v = recover.velocity
    recover.step(RECOVER)
    assert not recover.gliding
    assert recover.velocity == pytest.approx(v - 0.05 * (0.08 * v * v + 0.15 * v + 0.5 * v))


def test_turn_at_the_wall():
    event = fresh()
    while event.turns == 0:
        event.step(PULL if event.steps % 20 < 12 else RECOVER)
    assert event.position >= 50.0
    assert event.lap == 2
    assert event.velocity > 3.5  # 0.2 * v + 3.5 push-off
    assert event.observe()["distance_to_wall"] == pytest.approx(100.0 - event.position)


def test_finish_and_score():
    event = fresh()
    while not event.is_finished():
        event.step(PULL if event.steps % 20 < 12 else RECOVER)
    assert event.position == 100.0
    assert event.turns == 1
    assert event.score() == event.time < 120.0
    with pytest.raises(RuntimeError):
        event.step(PULL)


def test_unfinished_race_score():
    event = fresh()
    while not event.is_finished():
        event.step(RECOVER)
    assert event.position < 100.0
    assert event.time == pytest.approx(120.0)
    assert event.score() == pytest.approx(120.0 + 100.0 - event.position)


def test_illegal_action():
    with pytest.raises(ValueError):
        fresh().step("KICK")
