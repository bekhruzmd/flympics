import pytest

from events.hurdles import (BENCHMARK_LAYOUT, JUMP, RUN, Hurdles, generate_layout, validate_layout)


def jump_length(event, velocity):
    """Expected ground covered in one jump: take-off cost, then drag with no push."""
    air_speed = velocity * event.takeoff_keep * (1 - event.drag * event.air_time / 2)
    return air_speed * event.air_time


def race_one_jump(event, window_fraction, pair_gap=0.0):
    """RUN, then JUMP once when the next hurdle is window_fraction of a jump length away."""
    event.reset()
    jumped = False
    while not event.is_finished():
        observation = event.observe()
        trigger = window_fraction * jump_length(event, observation["velocity"]) + pair_gap
        if not jumped and observation["distance_to_next_hurdle"] <= trigger:
            jumped = True
            event.step(JUMP)
        else:
            event.step(RUN)
    return event


def test_reset_and_observation():
    event = Hurdles(); observation = event.reset(123)
    assert observation == {"distance_to_next_hurdle": 15.0, "distance_to_second_hurdle": 22.0, "velocity": 0.0}
    assert event.hurdles == BENCHMARK_LAYOUT
    assert not event.is_finished() and not event.airborne and event.height == 0.0


def test_fixed_step_is_deterministic():
    a, b = Hurdles(seeded=True), Hurdles(seeded=True); a.reset(4); b.reset(4)
    for i in range(400):
        action = JUMP if i % 37 == 0 else RUN
        a.step(action); b.step(action)
    assert a.observe() == b.observe() and (a.hits, a.jumps) == (b.hits, b.jumps)


def test_illegal_action_and_finished_race():
    event = Hurdles(); event.reset()
    with pytest.raises(ValueError):
        event.step("ACCELERATE")
    event = Hurdles(max_time=0.1); event.reset()
    event.step(RUN); event.step(RUN)
    assert event.is_finished()
    with pytest.raises(RuntimeError):
        event.step(RUN)


def test_jump_in_the_air_is_ignored():
    event = Hurdles(layout=(100.0,)); event.reset()
    for _ in range(40):
        event.step(RUN)
    event.step(JUMP)
    velocity_after_takeoff = event.velocity
    event.step(JUMP)
    assert event.jumps == 1 and event.airborne
    # Only drag acts in the air: no second take-off cost, no push.
    assert event.velocity == pytest.approx(velocity_after_takeoff * (1 - event.dt * event.drag))
    for _ in range(round(event.air_time / event.dt)):
        event.step(JUMP if not event.airborne else RUN)
    assert event.jumps == 2


def test_takeoff_cost():
    event = Hurdles(layout=(100.0,)); event.reset()
    for _ in range(40):
        event.step(RUN)
    before = event.velocity
    event.step(JUMP)
    assert event.velocity == pytest.approx(before * event.takeoff_keep * (1 - event.dt * event.drag))


def test_timing_window_has_two_edges():
    good = race_one_jump(Hurdles(layout=(100.0,)), 0.5)
    assert (good.clears, good.hits) == (1, 0)
    early = race_one_jump(Hurdles(layout=(100.0,)), 1.2)
    assert (early.clears, early.hits) == (0, 1)
    late = race_one_jump(Hurdles(layout=(100.0,)), 0.08)
    assert (late.clears, late.hits) == (0, 1)


def test_one_jump_clears_a_pair():
    # Middle of the shrunken window [0.146 L, 0.854 L - gap].
    event = race_one_jump(Hurdles(layout=(100.0, 101.2)), 0.5, pair_gap=-0.6)
    assert (event.clears, event.hits, event.jumps) == (2, 0, 1)


def test_hit_applies_stumble():
    event = Hurdles(layout=(15.0,)); event.reset()
    while event.hits == 0:
        before = event.velocity
        event.step(RUN)
    running = before + event.dt * (event.acceleration - event.drag * before)
    assert event.velocity == pytest.approx(running * event.stumble_keep)


def test_observation_caps_and_runs_out():
    event = Hurdles(layout=(100.0,)); event.reset()
    assert event.observe()["distance_to_next_hurdle"] == event.sensor_range
    assert event.observe()["distance_to_second_hurdle"] == event.sensor_range
    while event.position < 100.0:
        event.step(RUN)
    assert event.observe()["distance_to_next_hurdle"] == event.sensor_range


def test_baselines_lose_the_way_the_rules_intend():
    always_run = Hurdles(); always_run.reset()
    while not always_run.is_finished():
        always_run.step(RUN)
    assert always_run.hits == len(BENCHMARK_LAYOUT)
    always_jump = Hurdles(); always_jump.reset()
    while not always_jump.is_finished():
        always_jump.step(JUMP)
    assert always_jump.position == 0.0 and always_jump.score() == always_jump.max_time + always_jump.distance


def test_generated_layouts_follow_the_rules():
    for seed in range(1000):
        layout = generate_layout(seed)
        validate_layout(layout, 200.0)
        assert 10 <= len(layout) <= 14
    assert generate_layout(5) == generate_layout(5) != generate_layout(6)
    assert Hurdles(seeded=True).reset(5)["distance_to_next_hurdle"] == 15.0
    with pytest.raises(ValueError):
        Hurdles(seeded=True).reset()


@pytest.mark.parametrize("layout", [
    (50.0, 51.0, 52.2),         # cluster of three
    (50.0, 53.0),               # gap that is neither a pair nor at least 6 m
    (10.0,),                    # run-up too short
    (50.0, 190.0),              # run-out too short
    (),
])
def test_validate_layout_rejects(layout):
    with pytest.raises(ValueError):
        validate_layout(layout, 200.0)
