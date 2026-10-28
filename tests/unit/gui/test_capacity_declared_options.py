"""TESTONLY: the declared gas probability on top of a qualified native-phase-n1 core.

A full task on the qualified head and route may declare the gas probability.
It adds its upper-bound budget per frame and pins its writer. The storage
layer, the pixel-domain route and the decision probability storage are
refused on this route, and a task without options takes the unchanged path.
Synthetic data only.
"""
import inspect
import io

import numpy as np
import pytest

from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline import option_capacity as oc
from darsia.fluidflower.phase import probability_storage as ps
from darsia.fluidflower.mass import n1_pipeline
from darsia_cache import control
from darsia_cache.policy import CachePolicyError
from .test_capacity import bound_output_profile_TESTONLY, estimate, native_layout
from .test_bound_profile_other_runs import (bound_cache_TESTONLY, other_cache_task_TESTONLY,
                                            other_run_TESTONLY, run_layout_TESTONLY)

LAYOUT = dict(native_shape_hw=[2979, 5472], phase_shape_hw=[3572, 6304])
PROBABILITY = dict(format=ps.REQUEST_FORMAT, encoding=ps.ENCODING)
OPTIONS = dict(gas_probability_storage=PROBABILITY)


def arrays_TESTONLY(shape, seed=0):
    """Valid probability arrays with incompressible scored values."""
    generator = np.random.default_rng(seed)
    p = generator.random(shape, dtype=np.float32)
    r = generator.random(shape, dtype=np.float32)
    codes = np.full(shape, ps.CODES["centre"], np.uint8)
    return dict(zip(ps.ARRAYS, (p, r, codes, np.zeros(shape, np.uint16))))


def test_the_probability_budget_counts_each_array_at_its_exact_dtype():
    value = oc.probability_budget(OPTIONS, canvas_hw=(3572, 6304), precision="float32")
    pixels = 3572 * 6304
    payloads = dict(pixel_probability=4 * pixels, regional_probability=4 * pixels,
                    branch_support=pixels, later_rule_changes=2 * pixels)
    assert value["frame_bytes"] == oc.npz_bound(payloads) + oc.RECEIPT_BYTES and value["run_bytes"] == 0
    assert {name: row["payload_bytes"] for name, row in value["arrays"].items()} == payloads
    assert value["frame_bytes"] < 11 * pixels * 1.002 + 2 * oc.RECEIPT_BYTES


def test_the_probability_dtypes_are_the_writers_closed_schema():
    assert tuple(oc.PROBABILITY_ARRAYS) == ps.ARRAYS
    arrays = arrays_TESTONLY((2, 3))
    ps.validate_arrays(arrays)
    assert {name: value.dtype.str for name, value in arrays.items()} == oc.PROBABILITY_ARRAYS
    for name in ps.ARRAYS:
        wider = dict(arrays, **{name: arrays[name].astype(np.float64)})
        with pytest.raises(ValueError, match="Exact float32/uint8/uint16"):
            ps.validate_arrays(wider)


def test_the_archive_bound_covers_the_real_writer_on_incompressible_values(tmp_path):
    shape = (61, 97)
    arrays = arrays_TESTONLY(shape)
    written = n1_pipeline._write_npz(tmp_path / "phase_probabilities.npz", arrays)
    pixels = shape[0] * shape[1]
    bound = oc.npz_bound({name: pixels * np.dtype(dtype).itemsize
                          for name, dtype in oc.PROBABILITY_ARRAYS.items()})
    assert written["size_bytes"] <= bound


def test_the_header_margins_cover_any_two_dimensional_array_and_empty_members(tmp_path):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.empty((999999999, 0), np.float32), allow_pickle=False)
    assert len(stream.getvalue()) <= oc.NPY_HEADER_BYTES
    empty = {name: np.empty((0, 999999999), np.dtype(dtype)) for name, dtype in oc.PROBABILITY_ARRAYS.items()}
    written = n1_pipeline._write_npz(tmp_path / "empty.npz", empty)
    assert written["size_bytes"] <= oc.npz_bound({name: 0 for name in empty})


def test_the_writer_writes_one_archive_and_one_receipt():
    source = inspect.getsource(ps.write_product)
    assert source.count("_write_npz(") == 1 and source.count("_write_json(") == 1
    assert '"phase_probabilities.npz"' in source and '"phase_probabilities_receipt.json"' in source


def test_declared_options_refuse_the_other_optional_outputs_on_this_route():
    assert oc.declared_options({}) == ()
    assert oc.declared_options(dict(OPTIONS)) == oc.OPTIONS
    assert oc.declared_options(dict(gas_probability_storage=None)) == oc.OPTIONS
    assert oc.declared_options(dict(storage=None, pixel_domain=None, probability_storage=None)) == ()
    for key, reason in oc.REFUSED.items():
        with pytest.raises(ValueError, match=key + " " + reason.split(";")[0]):
            oc.declared_options({key: {"TESTONLY": "request"}})


def test_an_option_without_a_budget_is_refused(monkeypatch):
    monkeypatch.delitem(oc.BUDGETS, "gas_probability_storage")
    with pytest.raises(ValueError, match="has no budget"):
        oc.option_budget(dict(OPTIONS), layout=LAYOUT, precision="float32")


@pytest.mark.parametrize("request_value", [dict(PROBABILITY, encoding="TESTONLY"), None, "TESTONLY"])
def test_an_invalid_probability_request_is_refused(request_value):
    with pytest.raises(ValueError, match="decision probability request"):
        oc.option_budget(dict(gas_probability_storage=request_value), layout=LAYOUT, precision="float32")


@pytest.mark.parametrize("layout", [{}, dict(phase_shape_hw=[0, 6304]), dict(phase_shape_hw=[3572.0, 6304]),
                                    dict(phase_shape_hw=[3572, 6304, 3])])
def test_the_budget_needs_an_explicit_phase_canvas(layout):
    with pytest.raises(ValueError, match="Explicit positive grid shape required"):
        oc.option_budget(dict(OPTIONS), layout=layout, precision="float32")


def test_requests_are_checked_by_the_writers_own_validator_without_a_layout():
    assert oc.validate_requests({}) == ()
    assert oc.validate_requests(dict(OPTIONS)) == oc.OPTIONS
    with pytest.raises(ValueError, match="Explicit exact-float32"):
        oc.validate_requests(dict(gas_probability_storage=dict(PROBABILITY, encoding="TESTONLY")))


def test_the_option_budget_names_its_writer():
    value = oc.option_budget(dict(OPTIONS), layout=LAYOUT, precision="float32")
    assert value.options == oc.OPTIONS and value.writers == oc.WRITERS["gas_probability_storage"]
    assert value.frame_bytes == value.parts["gas_probability_storage"]["frame_bytes"]
    with pytest.raises(ValueError, match="No declared optional output"):
        oc.option_budget({}, layout=LAYOUT, precision="float32")


def test_the_probability_writer_is_pinned_from_this_repository():
    import subprocess
    from pathlib import Path
    root = Path(oc.__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("Needs a Git checkout of this repository")
    rows = cap._pinned_writer_rows(root, commit, oc.WRITERS["gas_probability_storage"])
    assert [row["relative_path"] for row in rows] == list(oc.WRITERS["gas_probability_storage"])
    assert all(len(row["sha256"]) == 64 for row in rows)


def bound_TESTONLY(monkeypatch, *, layout=None):
    """The qualified 39-frame bound profile, its writers checked by recording only."""
    task, config, wrapped = bound_output_profile_TESTONLY()
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: dict(layout or native_layout()))
    pins = []
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: ("TESTONLY " + args[-1]).encode())
    monkeypatch.setattr(cap, "_writer_identity", lambda repository, commit, row: pins.append((commit, row)))
    return task, config, wrapped, pins


def estimate_TESTONLY(task, config, wrapped):
    """The bound-profile estimate of a task, on synthetic stores."""
    return cap._bound_result_estimate(None, "TESTONLY", task, config, wrapped, {},
                                      repository="TESTONLY", metadata_rows=None)


def test_a_task_without_options_takes_the_unchanged_path(monkeypatch):
    task, config, wrapped, pins = bound_TESTONLY(monkeypatch)
    monkeypatch.setattr(oc, "option_budget", lambda *args, **kwargs: pytest.fail("No option budget without options"))
    value = estimate_TESTONLY(task, config, wrapped)
    actual = wrapped["result_profile"]
    assert type(value) is cap.SpaceEstimate and pins == []
    assert value == cap.estimate_run(task, config, actual, measurement=actual["measurement"],
                                     completion=actual["completion"], requested_native_layout=native_layout(),
                                     profile_identity={})


def test_the_gas_probability_adds_its_budget_per_frame_and_pins_its_writer(monkeypatch):
    task, config, wrapped, pins = bound_TESTONLY(monkeypatch)
    core = estimate_TESTONLY(task, config, wrapped)
    value = estimate_TESTONLY(task, dict(config, **OPTIONS), wrapped)
    budget = oc.option_budget(dict(OPTIONS), layout=native_layout(), precision="float32")
    assert type(value) is cap.OptionSpaceEstimate
    assert value.option_frame_bytes == budget.frame_bytes and value.option_run_bytes == 0
    assert value.products_per_frame_estimated_bytes == core.products_per_frame_estimated_bytes + budget.frame_bytes
    assert value.products_estimated_bytes == core.products_estimated_bytes + 39 * budget.frame_bytes
    run_bytes = oc.run_file_bytes(wrapped["result_profile"]["measurement"]["inventory"],
                                 wrapped["result_profile"]["measured_case_ids"])
    retained = 3 * (39 * (core.transient_estimated_bytes + budget.frame_bytes) + run_bytes)
    assert value.retained_serialization_estimated_bytes == retained
    assert value.transient_estimated_bytes == core.transient_estimated_bytes + budget.frame_bytes + retained
    assert value.native_expected_bytes == core.native_expected_bytes
    assert value.options["declared"] == list(oc.OPTIONS)
    assert [row["relative_path"] for row in value.options["writers"]] == list(budget.writers)
    assert [commit for commit, _ in pins] == [task["commit"]] * len(budget.writers)
    assert "declared option" in value.model


def test_another_run_with_the_gas_probability_gets_its_budget_per_frame(monkeypatch):
    task, config, wrapped = other_run_TESTONLY(count=41)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: run_layout_TESTONLY())
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: b"TESTONLY")
    monkeypatch.setattr(cap, "_writer_identity", lambda *args: None)
    core = estimate_TESTONLY(task, config, wrapped)
    value = estimate_TESTONLY(task, dict(config, **OPTIONS), wrapped)
    assert value.case_count == 41
    assert value.products_estimated_bytes == core.products_estimated_bytes + 41 * value.option_frame_bytes


def test_an_option_without_a_budget_is_refused_at_claim(monkeypatch):
    task, config, wrapped, pins = bound_TESTONLY(monkeypatch)
    monkeypatch.delitem(oc.BUDGETS, "gas_probability_storage")
    with pytest.raises(ValueError, match="qualified head and route: Optional output gas_probability_storage has no budget"):
        estimate_TESTONLY(task, dict(config, **OPTIONS), wrapped)
    assert pins == []


@pytest.mark.parametrize("key", sorted(oc.REFUSED))
def test_other_optional_outputs_are_refused_at_claim_before_any_writer_read(monkeypatch, key):
    task, config, wrapped, _ = bound_TESTONLY(monkeypatch)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: pytest.fail("No writer read"))
    with pytest.raises(ValueError, match="qualified head and route: Optional output " + key):
        estimate_TESTONLY(task, dict(config, **{key: {"TESTONLY": "request"}}), wrapped)


def test_an_invalid_request_is_refused_at_claim_before_any_profile_read(monkeypatch):
    task, config, wrapped, pins = bound_TESTONLY(monkeypatch)
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: pytest.fail("No writer read"))
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: pytest.fail("No layout read"))
    with pytest.raises(ValueError, match="qualified head and route: Explicit exact-float32"):
        estimate_TESTONLY(task, dict(config, gas_probability_storage=dict(PROBABILITY, encoding="TESTONLY")),
                          wrapped)
    assert pins == []


@pytest.mark.parametrize("key,value", [("backend", "cuda"), ("precision", "float64"),
                                       ("commit", "f" * 40), ("pipeline", "n1")])
def test_the_option_never_opens_another_head_or_route(monkeypatch, key, value):
    task, config, wrapped, _ = bound_TESTONLY(monkeypatch)
    task[key] = value
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, dict(config, **OPTIONS), wrapped)


def test_the_option_estimate_reserves_and_reports_like_any_estimate(tmp_path, monkeypatch):
    from .test_capacity import claim_fixture, ledger_fixture, owner
    core = estimate()
    value = cap.OptionSpaceEstimate(**{key: getattr(core, key) for key in (
        "run", "case_count", "measured_native_bytes_per_frame", "native_expected_bytes",
        "coarse_uncompressed_bytes", "header_metadata_copy_estimated_bytes", "profile", "measured_commit")},
        transient_estimated_bytes=core.transient_estimated_bytes + 7,
        products_per_frame_estimated_bytes=core.products_per_frame_estimated_bytes + 7,
        option_frame_bytes=7, options=dict(declared=["TESTONLY"]))
    ledger, _, _ = ledger_fixture(tmp_path, monkeypatch)
    with ledger.locked():
        ledger.bind(tmp_path / "job", claim_fixture(), owner(), value)
        assert ledger.remaining() == value.products_per_frame_estimated_bytes * 2 + value.transient_estimated_bytes
    report = cap.report(value, {"free_bytes": 10**12}, 0, 40)
    assert report["options"] == dict(declared=["TESTONLY"]) and report["option_frame_bytes"] == 7
    assert report["needed_bytes"] == value.products_estimated_bytes + value.transient_estimated_bytes + 40


def test_the_colour_coordinate_is_refused_unless_the_qualified_profile_measured_it(monkeypatch):
    task, config, wrapped, _ = bound_TESTONLY(monkeypatch)
    colour = dict(config, storage_profile=dict(store_colour_coordinate=True))
    with pytest.raises(ValueError, match="qualified head and route: the colour-path coordinate was not measured"):
        estimate_TESTONLY(task, colour, wrapped)
    measurement = wrapped["result_profile"]["measurement"]
    rows = [dict(relative_path="frames/" + case + "/colour_path_coordinate.npz", bytes=5)
            for case in task["ordered_case_ids"]]
    measurement.update(inventory=measurement["inventory"] + rows,
                       total_logical_bytes=measurement["total_logical_bytes"] + 5 * len(rows))
    value = estimate_TESTONLY(task, colour, wrapped)
    assert type(value) is cap.SpaceEstimate
    assert estimate_TESTONLY(task, dict(config, storage_profile=dict(store_colour_coordinate=False)),
                             wrapped) == value


def test_a_lean_task_with_the_gas_probability_keeps_the_lean_refusal(tmp_path, monkeypatch):
    from ..test_storage_profile_jobs import _lean_task
    task, config = _lean_task()
    monkeypatch.setattr(cap, "_lean_writer_identity", lambda *args: [])
    monkeypatch.setattr(oc, "option_budget", lambda *args, **kwargs: pytest.fail("No bound option rule for lean"))
    with pytest.raises(ValueError, match="gas-probability output has no documented size"):
        cap.load_estimate(None, "TESTONLY", task, dict(config, **OPTIONS), repository=tmp_path)


def test_cuda_and_another_machine_stay_refused_on_the_cache_side(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task = dict(other_cache_task_TESTONLY(f), backend="cuda")
    with pytest.raises(CachePolicyError, match="qualified head and route"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    f.adapter.lease.policy.volume_cap.document = lambda: dict(TESTONLY="another machine's volume")
    with pytest.raises(CachePolicyError, match="another physical cache bound"):
        control.applicable_job_cache_profile(f.policy, f.adapter, other_cache_task_TESTONLY(f), None)
