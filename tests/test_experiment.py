from experiments.run_sprint import run


def test_seeded_random_experiment_is_reproducible():
    assert run("random", seed=4) == run("random", seed=4)


def test_controller_interface_runs_for_each_baseline_and_fly():
    for name in ("random", "rule", "fly"):
        records = run(name, seed=2)
        assert records and records[-1]["simulation_time"] <= 60.0
        assert all(row["action"] in {"COAST", "ACCELERATE"} for row in records)
