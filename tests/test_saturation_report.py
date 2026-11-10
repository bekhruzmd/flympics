import importlib

import numpy as np
import pytest

from brain.dynamics import RateNetwork
from experiments import saturation_report as report

NAMES = sorted(report.EVENTS)


@pytest.fixture(scope="module", params=NAMES)
def event_rows(request):
    spec = report.EVENTS[request.param]
    return spec, report.report(spec, gains=(0.5, 3.0))


@pytest.mark.parametrize("name", NAMES)
def test_default_gain_is_unchanged(name):
    fly = report.EVENTS[name].make_fly()
    assert fly.network.gain == 3.0
    assert RateNetwork(fly.circuit).gain == 3.0


@pytest.mark.parametrize("name", NAMES)
def test_gain_3_race_matches_the_plain_fly(name):
    spec = report.EVENTS[name]
    runner = importlib.import_module(f"experiments.run_{name}")
    fly = spec.make_fly()
    fly.network.gain = 3.0
    assert spec.race(fly) == runner.run("fly")


def test_rows_are_bounded_and_complete(event_rows):
    spec, rows = event_rows
    for row in rows:
        for group, _ in report.GROUPS:
            assert 0.0 <= row[f"{group}_mean"] <= 1.0
        for name in spec.inputs:
            assert np.isfinite(row[f"effect_{name}"])
        assert np.isfinite(row["score"])


def test_higher_gain_saturates_and_mutes_inputs(event_rows):
    spec, (low, high) = event_rows
    assert high["intrinsic_mean"] > low["intrinsic_mean"]
    for name in spec.inputs:
        assert abs(high[f"effect_{name}"]) < abs(low[f"effect_{name}"])


@pytest.mark.parametrize("name", NAMES)
def test_input_effect_leaves_the_network_reset(name):
    spec = report.EVENTS[name]
    fly = spec.make_fly()
    report.input_effect(fly, len(spec.inputs), 0)
    assert not fly.network.state.any()
