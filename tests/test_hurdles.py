import pytest

from events.hurdles import (ACCELERATE, BENCHMARK_LAYOUT, COAST, GAP, JUMP, MIN_GAP, Hurdles,
                            generate_layout, validate_layout)
from events.sprint import Sprint


def jump_length(event, velocity):
    """Expected ground covered in one jump: take-off cost, then drag with no push."""
    air_speed = velocity * event.takeoff_keep_at(velocity) * (1 - event.drag * event.air_time / 2)
    return air_speed * event.air_time


def race_one_jump(event, window_fraction):
    """ACCELERATE, then JUMP once when the next hurdle is window_fraction of a jump length away."""
    event.reset()
    jumped = False
    while not event.is_finished():
        observation = event.observe()
        trigger = window_fraction * jump_length(event, observation["velocity"])
        if not jumped and observation["distance_to_next_hurdle"] <= trigger:
            jumped = True
            event.step(JUMP)
        else:
            event.step(ACCELERATE)
    return event


def speed_up(event, steps=40):
    event.reset()
    for _ in range(steps):
        event.step(ACCELERATE)
    return event


def test_reset_and_observation():
    event = Hurdles(); observation = event.reset(123)
    assert observation == {"distance_to_next_hurdle": 15.0, "distance_to_second_hurdle": 24.0, "velocity": 0.0}
    assert event.hurdles == BENCHMARK_LAYOUT
    assert not event.is_finished() and not event.airborne and event.height == 0.0
    assert event.coasts == 0 and event.last_takeoff_keep is None


def test_fixed_step_is_deterministic():
    a, b = Hurdles(seeded=True), Hurdles(seeded=True); a.reset(4); b.reset(4)
    actions = (ACCELERATE, ACCELERATE, COAST, JUMP)
    for i in range(400):
        action = actions[i % 4] if i % 37 == 0 else ACCELERATE
        a.step(action); b.step(action)
    assert a.observe() == b.observe() and (a.hits, a.jumps, a.coasts) == (b.hits, b.jumps, b.coasts)


def test_illegal_action_and_finished_race():
    event = Hurdles(); event.reset()
    with pytest.raises(ValueError):
        event.step("RUN")
    event = Hurdles(max_time=0.1); event.reset()
    event.step(ACCELERATE); event.step(ACCELERATE)
    assert event.is_finished()
    with pytest.raises(RuntimeError):
        event.step(ACCELERATE)


def test_accelerate_matches_sprint():
    hurdles, sprint = Hurdles(), Sprint(dt=0.05)
    hurdles.reset(); sprint.reset()
    for _ in range(40):  # 2 s, well before the first hurdle at 15 m
        hurdles.step(ACCELERATE); sprint.step("ACCELERATE")
    assert hurdles.velocity == sprint.velocity and hurdles.position == sprint.position


def test_coast_and_air_slow_the_fly_equally():
    coaster, flyer = speed_up(Hurdles(layout=(100.0,))), speed_up(Hurdles(layout=(100.0,)))
    flyer.step(JUMP)  # take-off step: take-off cost, then drag
    speed = flyer.velocity
    coaster.velocity = speed
    coaster.step(COAST); flyer.step(COAST)
    assert coaster.velocity == flyer.velocity == pytest.approx(speed * (1 - coaster.dt * coaster.drag))
    assert coaster.coasts == 1 and flyer.coasts == 0  # COAST in the air is ignored


def test_takeoff_cost_grows_above_safe_speed():
    event = Hurdles()
    assert event.takeoff_keep_at(3.0) == event.takeoff_keep_at(4.5) == 0.90
    assert event.takeoff_keep_at(5.5) == pytest.approx(0.50)
    assert event.takeoff_keep_at(7.5) == event.min_takeoff_keep == 0.20
    # Faster than safe_speed always leaves the fly slower after the jump.
    after = [v * event.takeoff_keep_at(v) for v in (4.5, 5.0, 5.5, 6.0)]
    assert after == sorted(after, reverse=True)


def test_takeoff_applies_the_cost_at_the_current_speed():
    event = speed_up(Hurdles(layout=(100.0,)), steps=160)  # about 5.1 m/s at 48 m
    before = event.velocity
    assert before > event.safe_speed
    event.step(JUMP)
    keep = event.takeoff_keep_at(before)
    assert event.last_takeoff_keep == keep
    assert event.velocity == pytest.approx(before * keep * (1 - event.dt * event.drag))


def test_actions_in_the_air_are_ignored():
    event = speed_up(Hurdles(layout=(100.0,)))
    event.step(JUMP)
    velocity = event.velocity
    event.step(JUMP); event.step(ACCELERATE)
    assert event.jumps == 1 and event.airborne
    assert event.velocity == pytest.approx(velocity * (1 - event.dt * event.drag) ** 2)
    while event.airborne:
        event.step(COAST)
    event.step(JUMP)
    assert event.jumps == 2


def test_timing_window_has_two_edges():
    good = race_one_jump(Hurdles(layout=(100.0,)), 0.5)
    assert (good.clears, good.hits) == (1, 0)
    early = race_one_jump(Hurdles(layout=(100.0,)), 1.2)
    assert (early.clears, early.hits) == (0, 1)
    late = race_one_jump(Hurdles(layout=(100.0,)), 0.08)
    assert (late.clears, late.hits) == (0, 1)


def test_hit_applies_stumble():
    event = Hurdles(layout=(15.0,)); event.reset()
    while event.hits == 0:
        before = event.velocity
        event.step(ACCELERATE)
    running = before + event.dt * (event.acceleration - event.drag * before)
    assert event.velocity == pytest.approx(running * event.stumble_keep)


def test_observation_caps_and_runs_out():
    event = Hurdles(layout=(100.0,)); event.reset()
    assert event.observe()["distance_to_next_hurdle"] == event.sensor_range
    assert event.observe()["distance_to_second_hurdle"] == event.sensor_range
    while event.position < 100.0:
        event.step(ACCELERATE)
    assert event.observe()["distance_to_next_hurdle"] == event.sensor_range


def test_simple_strategies_lose():
    always_accelerate = Hurdles(); always_accelerate.reset()
    while not always_accelerate.is_finished():
        always_accelerate.step(ACCELERATE)
    assert always_accelerate.hits == len(BENCHMARK_LAYOUT)
    for action in (COAST, JUMP):
        stuck = Hurdles(); stuck.reset()
        while not stuck.is_finished():
            stuck.step(action)
        assert stuck.position == 0.0 and stuck.score() == stuck.max_time + stuck.distance


def test_generated_layouts_follow_the_rules():
    for seed in range(1000):
        layout = generate_layout(seed)
        validate_layout(layout, 200.0)
        gaps = [b - a for a, b in zip(layout, layout[1:])]
        assert all(GAP[0] <= gap <= GAP[1] for gap in gaps)
        assert 7 <= len(layout) <= 13
    assert generate_layout(5) == generate_layout(5) != generate_layout(6)
    assert Hurdles(seeded=True).reset(5)["distance_to_next_hurdle"] == 15.0
    with pytest.raises(ValueError):
        Hurdles(seeded=True).reset()


@pytest.mark.parametrize("layout", [
    (50.0, 50.0 + MIN_GAP - 1),  # gap too short for its own jump
    (10.0,),                     # run-up too short
    (50.0, 195.0),               # run-out too short
    (),
])
def test_validate_layout_rejects(layout):
    with pytest.raises(ValueError):
        validate_layout(layout, 200.0)
