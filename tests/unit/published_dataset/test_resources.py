"""The launcher's worker default: the Jobs panel's arithmetic with the published measurement."""

import pytest

from darsia.fluidflower.published_dataset import resources
from darsia.fluidflower.resource_advisor_panel import GIB, resource_advice

MANIFEST = {"plan": {"precision": "float32", "pipeline": "native-phase-n1"},
            "code": {"published_commit": "c" * 40},
            "machine": {"threads": 4, "peak_memory_bytes": int(11.8 * GIB),
                        "peak_memory_basis": "committed memory, one worker"}}


def inventory(cores, available_gib, logical=None):
    return {"cpu": {"physical_cores": cores, "logical_cores": logical or cores},
            "memory": {"total_bytes": int(available_gib * 2 * GIB), "available_bytes": int(available_gib * GIB)}}


def test_formula_takes_the_smaller_of_cores_and_memory_and_the_runs_in_scope():
    advice = resources.default_workers(MANIFEST, runs_in_scope=39, inventory=inventory(32, 100))
    # min(32 / 4 = 8, (100 - 1) / 11.8 = 8) = 8
    assert (advice["core_limit"], advice["ram_limit"], advice["workers"]) == (8, 8, 8)
    assert "measured on the reference machine (committed memory, one worker)" in advice["reason"]
    assert "minus 1 GiB reserve" in advice["reason"]
    assert resources.default_workers(MANIFEST, runs_in_scope=39, threads=8,
                                     inventory=inventory(32, 100))["workers"] == 4
    assert resources.default_workers(MANIFEST, runs_in_scope=39, inventory=inventory(32, 100),
                                     reserve_bytes=40 * GIB)["workers"] == 5


def test_a_laptop_gets_one_worker_or_a_clear_message():
    laptop = resources.default_workers(MANIFEST, runs_in_scope=39, inventory=inventory(4, 14))
    assert laptop["workers"] == 1 and "message" not in laptop
    small = resources.default_workers(MANIFEST, runs_in_scope=39, inventory=inventory(4, 10))
    assert small["workers"] == 0
    assert small["message"].startswith("Not even one worker fits: 10.0 GiB available minus a 1 GiB reserve")
    assert "Nothing starts unless you choose a worker count" in small["message"]


def test_no_run_is_split_over_workers():
    one_run = resources.default_workers(MANIFEST, runs_in_scope=1, inventory=inventory(64, 500))
    assert one_run["workers"] == 1 and "a run never spans workers" in one_run["reason"]
    assert resources.default_workers(MANIFEST, runs_in_scope=0, inventory=inventory(64, 500))["workers"] == 0


def test_without_a_measurement_the_user_chooses():
    manifest = dict(MANIFEST, machine={"threads": 2, "peak_memory_bytes": None})
    advice = resources.default_workers(manifest, runs_in_scope=3, inventory=inventory(8, 64))
    assert advice["workers"] is None and "choose the worker count" in advice["reason"]
    cuda = resources.default_workers(MANIFEST, runs_in_scope=3, backend="cuda", inventory=inventory(8, 64))
    assert cuda["workers"] is None and "VRAM" in cuda["reason"]


def test_the_jobs_panel_rules_are_unchanged_by_default():
    machine = {"machine": "m", "inventory": inventory(16, 100), "history": []}
    job = {"precision": "float32", "pipeline": "native-phase-n1", "commit": "a" * 40}
    advice = resource_advice(machine, job, "cpu", 2, policy_budget=True)
    assert advice["workers"] == min(16 // 2, (100 - 40) // 20)
    assert "available RAM minus 40 GiB reserve" in advice["reason"]
    assert "run(s) are in scope" not in advice["reason"]
    bad = resource_advice(machine, job, "cpu", 2, measured_memory={"bytes": 0, "basis": "x"})
    assert bad["workers"] is None


def test_machine_inventory_has_cores_and_memory_but_no_names():
    facts = resources.machine_inventory()
    assert set(facts) == {"cpu", "memory"} and "available_bytes" in facts["memory"]
    with pytest.raises(ValueError):
        resources.default_workers(MANIFEST, runs_in_scope=1, backend="opencl", inventory=inventory(4, 16))
