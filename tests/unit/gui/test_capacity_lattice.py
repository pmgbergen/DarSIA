"""The claim-time ESTIMATE of a lattice run; metadata only, no science, NAS or worker run."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity as cap, lattice_capacity as lc
from darsia.fluidflower.storage_outputs import LA_ENCODING, REQUEST_FORMAT
from .test_capacity import ledger_fixture, owner

CANVAS = (3572, 6304)  # a production-sized native canvas
EXTENT = [0.0, 1.0, 0.0, 0.6]


def lattice_run(computing=1.0, stored=1.0, precision="float32", cases=("r1_f1", "r1_f2")):
    """A lattice task and its run configuration, as job preparation writes them."""
    storage = dict(format=REQUEST_FORMAT, gas_score_encoding=LA_ENCODING, resolution_mm=stored,
                   extent_m=list(EXTENT), computing_resolution_mm=computing)
    frames = [dict(case_id=case, mass_spec=dict(photo_top=dict(shape_hw=list(CANVAS)))) for case in cases]
    task = dict(run="r1", pipeline="lattice-phase-n1", computing_resolution_mm=computing, precision=precision,
                ordered_case_ids=list(cases), commit="a" * 40, attempt_id="1" * 32)
    return task, dict(run="r1", frames=frames, storage=storage)


def test_the_estimate_counts_lattice_cells_at_the_run_precision_and_the_native_canvas_in_flight():
    value = lc.lattice_estimate(*lattice_run())
    cells = 600 * 1000
    assert value.lattice_shape_hw == (600, 1000) and value.stored_shape_hw == (600, 1000)
    assert value.products_per_frame_estimated_bytes == cells * (108 + 5 * 4) + 5_000_000
    assert value.run_constant_estimated_bytes == cells * 44 + 1000 * 8 + 5_000_000
    assert value.native_transient_estimated_bytes == CANVAS[0] * CANVAS[1] * 13
    assert value.inflight_estimated_bytes == (value.products_per_frame_estimated_bytes
                                              + value.native_transient_estimated_bytes)
    assert value.products_estimated_bytes == 2 * value.products_per_frame_estimated_bytes
    assert value.transient_estimated_bytes == value.inflight_estimated_bytes + value.run_constant_estimated_bytes
    assert value.needed(reserve_bytes=0) == value.products_estimated_bytes + value.transient_estimated_bytes
    assert value.model.startswith("ESTIMATE, not a measurement") and value.estimate["estimate"] is True
    assert value.estimate["compression"] == "none assumed"


@pytest.mark.parametrize("computing, stored, precision, cells, stored_cells, float_bytes", [
    (0.5, 0.5, "float64", 1200 * 2000, 0, 8), (0.5, 1.0, "float32", 1200 * 2000, 600 * 1000, 4),
    (0.5, 2.0, "float64", 1200 * 2000, 300 * 500, 8), (1.0, 2.0, "float32", 600 * 1000, 300 * 500, 4)])
def test_finer_computing_and_coarser_storage_scale_the_products(computing, stored, precision, cells,
                                                                stored_cells, float_bytes):
    value = lc.lattice_estimate(*lattice_run(computing, stored, precision))
    assert value.computing_resolution_mm == computing and value.stored_resolution_mm == stored
    assert value.products_per_frame_estimated_bytes == (cells * (108 + 5 * float_bytes) + stored_cells * 256
                                                        + 5_000_000)
    assert value.estimate["lattice_cells"] == cells


@pytest.mark.parametrize("change, message", [
    (lambda task, config: task.update(pipeline="native-phase-n1"), "task pipeline"),
    (lambda task, config: task.update(computing_resolution_mm=0.5), "computing resolutions differ"),
    (lambda task, config: task.update(ordered_case_ids=task["ordered_case_ids"][::-1]), "exact case order"),
    (lambda task, config: task.update(precision="float16"), "precision"),
    (lambda task, config: config["frames"][1]["mass_spec"].pop("photo_top"), "declares no native canvas"),
    (lambda task, config: config.update(gas_probability_storage={}), "gas-probability"),
    (lambda task, config: config["storage"].pop("computing_resolution_mm"), "computing_resolution_mm"),
])
def test_a_task_or_configuration_that_does_not_match_is_refused(change, message):
    task, config = lattice_run()
    change(task, config)
    with pytest.raises(ValueError, match=message):
        lc.lattice_estimate(task, config)


def test_the_ledger_reserves_a_lattice_estimate_like_any_run(tmp_path, monkeypatch):
    ledger, _, _ = ledger_fixture(tmp_path, monkeypatch, free=10**12)
    task, config = lattice_run()
    value = lc.lattice_estimate(task, config)
    claim = SimpleNamespace(task=dict(task, job_id="one"), token=task["attempt_id"])
    with ledger.locked():
        path = ledger.bind(tmp_path / "job", claim, owner(), value)
        row = cap._read_local(path / "reserved.json")
        assert row["format"] == "darsia-local-capacity-reservation-v1"
        assert row["products_per_frame_estimated_bytes"] == value.products_per_frame_estimated_bytes
        assert row["transient_estimated_bytes"] == value.transient_estimated_bytes
        assert row["estimate"]["model"] == lc.LATTICE_ESTIMATE_MODEL
        assert ledger.remaining() == value.products_estimated_bytes + value.transient_estimated_bytes
        ledger.progress(path, owner(), dict(state="frame_completed", complete=True, run="r1", case_id="r1_f1"))
        assert ledger.remaining() == value.products_per_frame_estimated_bytes + value.transient_estimated_bytes
    report = cap.report(value, ledger.fresh_volume(), 0, 0)
    assert report["sufficient"] and report["needed_bytes"] == value.needed(reserve_bytes=0)
    assert report["model"] == lc.LATTICE_ESTIMATE_MODEL and asdict(value)["estimate"]["format"] == lc.ESTIMATE_FORMAT
