"""TESTONLY: a phase-one statistics task admitted by rule under a qualified native-phase-n1 profile.

Phase one runs the qualified native and phase stages on another route and
writes the same preparation files, without mass. A full phase-one task on the
qualified head is admitted as another run: the measured preparation bytes per
frame plus upper bounds for its own additions. The cache side admits it with
no claim bytes when its boundary modules are in the qualified closure.
Synthetic metadata only.
"""
import io
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.pipeline import capacity as cap
from darsia.fluidflower.pipeline import option_capacity as oc
from darsia.fluidflower.pipeline import phase1_capacity as p1
from darsia.fluidflower.phase import probability_storage as ps
from darsia_cache import control
from darsia_cache.policy import CachePolicyError
from .test_capacity import bound_output_profile_TESTONLY, native_layout, native_metadata
from .test_bound_profile_other_runs import (bound_cache_TESTONLY, other_cache_task_TESTONLY,
                                            run_layout_TESTONLY)
from tests.unit.test_staged_data import phase1_data as full_phase1_data, _bound as synthetic_bound
from darsia.fluidflower.pipeline.staged_data import validate_phase1_run

PROBABILITY = dict(gas_probability_storage=dict(format=ps.REQUEST_FORMAT, encoding=ps.ENCODING))


def phase1_task_TESTONLY(*, run="ac17", count=41):
    """The qualified 39-frame profile, and a phase-one task of another run."""
    qualified, _, wrapped = bound_output_profile_TESTONLY()
    task = dict(qualified, pipeline=p1.PIPELINE, run=run, attempt_id="2" * 32,
                ordered_case_ids=[run + "_f" + str(index) for index in range(1, count + 1)],
                configuration=dict(qualified["configuration"], sha256="e" * 64))
    config = deepcopy(full_phase1_data.__wrapped__())
    config["run"] = run
    config["ordered_case_ids"] = list(task["ordered_case_ids"])
    template = config["frames"][0]
    previous = config["phase"]["initial_previous"]["source"]["source_sha256"]
    config["frames"] = []
    for index, case in enumerate(task["ordered_case_ids"], 1):
        spec = deepcopy(template)
        digest = hashlib.sha256(("synthetic-phase1:" + run + ":" + case).encode()).hexdigest()
        spec.update(case_id=case, frame_index=index, elapsed_h=float(index))
        spec["source"].update(run=run, source_sha256=digest, previous_source_sha256=previous,
                              work_index=index, elapsed_h=float(index))
        spec["source"]["source_binding"]["source_sha256"] = digest
        spec["mass_spec"]["selected_target_h"] = float(index)
        config["frames"].append(spec)
        previous = digest
    # Budget-only source/profile/layout doubles remain explicit, but every
    # declared request uses the real full native-BEFORE DATA validator.
    validate_phase1_run(config)
    return task, config, wrapped


def estimate_TESTONLY(task, config, wrapped, monkeypatch, *, pins=None, layout=None):
    """The bound-profile estimate of a task, its layout and writers synthetic."""
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: None)
    monkeypatch.setattr(cap, "_native_layout_for_task", lambda *args: layout or run_layout_TESTONLY())
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: ("TESTONLY " + args[-1]).encode())
    monkeypatch.setattr(cap, "_writer_identity",
                        lambda repository, commit, row: (pins if pins is not None else []).append(row))
    return cap._bound_result_estimate(None, "TESTONLY", task, config, wrapped, {},
                                      repository="TESTONLY", metadata_rows=None)


@pytest.mark.parametrize("channels,mode", [(2, "LA"), (4, "RGBA")])
def test_the_png_bound_covers_the_real_writer_on_incompressible_pixels(channels, mode):
    shape = (61, 97, channels)
    pixels = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels, mode=mode).save(stream, format="PNG")
    assert len(stream.getvalue()) <= p1.png_bound(61, 97, channels)


def test_the_png_bound_is_the_uncompressed_rows_with_the_writers_library_margin(monkeypatch):
    raw = 3572 * (6304 * 2 + 1)
    monkeypatch.setattr(p1, "pillow_zlib_ng", lambda: False)
    classic = p1.png_bound(3572, 6304, 2)
    assert raw < classic < raw * 1.0015 + 4096
    monkeypatch.setattr(p1, "pillow_zlib_ng", lambda: True)
    # The larger stream also needs more 64 KiB IDAT chunk records.
    assert p1.png_bound(3572, 6304, 2) - classic >= oc.deflate_bound(raw, zlib_ng=True) - oc.deflate_bound(raw, zlib_ng=False)


@pytest.mark.parametrize("channels,mode,rows", [(2, "LA", 1800), (4, "RGBA", 900)])
def test_the_png_bound_covers_a_large_incompressible_write_on_this_runtime(channels, mode, rows):
    pixels = np.random.default_rng(3).integers(0, 256, (rows, 6304, channels), dtype=np.uint8)
    assert pixels.nbytes > 20_000_000
    stream = io.BytesIO()
    Image.fromarray(pixels, mode=mode).save(stream, format="PNG")
    assert len(stream.getvalue()) <= p1.png_bound(rows, 6304, channels)


def test_the_libraries_are_read_from_pillow_and_python(monkeypatch):
    from PIL import features
    import zlib
    monkeypatch.setattr(features, "version", lambda name: "1.3.1.zlib-ng" if name == "zlib" else None)
    assert p1.pillow_zlib_ng() is True
    monkeypatch.setattr(features, "version", lambda name: "1.3.1" if name == "zlib" else None)
    assert p1.pillow_zlib_ng() is False
    monkeypatch.setattr(features, "version", lambda name: None)
    assert p1.pillow_zlib_ng() is True  # An unknown compressor gets the larger bound.
    monkeypatch.setattr(zlib, "ZLIBNG_VERSION", "2.3.3", raising=False)
    assert oc.python_zlib_ng() is True and oc.deflate_bound(800) == 800 + 100 + oc.DEFLATE_STREAM_BYTES
    monkeypatch.delattr(zlib, "ZLIBNG_VERSION")
    assert oc.python_zlib_ng() is False and oc.deflate_bound(1000) == 1001 + oc.DEFLATE_STREAM_BYTES


def test_the_stage_bytes_are_the_measured_preparation_files_only():
    _, _, wrapped = bound_output_profile_TESTONLY()
    actual = wrapped["result_profile"]
    assert p1.stage_bytes(actual["measurement"]["inventory"], actual["measured_case_ids"]) == (101, 101)
    rows = [row for row in actual["measurement"]["inventory"] if "preparation" not in row["relative_path"]]
    with pytest.raises(ValueError, match="no preparation files"):
        p1.stage_bytes(rows, actual["measured_case_ids"])


def test_the_phase1_layout_is_read_like_the_series_layout(tmp_path):
    config, rows, layout = native_metadata(tmp_path)
    series = cap._native_layout_for_task(None, "TESTONLY", dict(pipeline="native-phase-n1", precision="float32",
                                                                run="ac52"), config, rows)
    phase1 = cap._native_layout_for_task(None, "TESTONLY", dict(pipeline=p1.PIPELINE, precision="float32",
                                                                run="ac52"),
                                         dict(config, format="darsia-native-phase-statistics-run-v1"), rows)
    assert phase1 == series == layout
    for pipeline, form in ((p1.PIPELINE, config["format"]),
                           ("native-phase-n1", "darsia-native-phase-statistics-run-v1")):
        with pytest.raises(ValueError, match="only for the genuine"):
            cap._native_layout_for_task(None, "TESTONLY", dict(pipeline=pipeline, precision="float32", run="ac52"),
                                        dict(config, format=form), rows)


def test_a_phase1_task_is_admitted_with_the_preparation_bytes_and_its_additions(monkeypatch):
    task, config, wrapped = phase1_task_TESTONLY()
    pins = []
    value = estimate_TESTONLY(task, config, wrapped, monkeypatch, pins=pins)
    height, width = run_layout_TESTONLY()["phase_shape_hw"]
    extra = p1.png_bound(height, width, 2) + p1.RECEIPTS_BYTES
    assert type(value) is cap.OptionSpaceEstimate and value.case_count == 41
    assert value.measured_native_bytes_per_frame == 101
    assert value.products_per_frame_estimated_bytes == 101 + extra
    assert value.products_estimated_bytes == 41 * (101 + extra)
    retained = 3 * (41 * (101 + extra) + p1.RUN_BYTES)
    assert value.retained_serialization_estimated_bytes == retained
    assert value.transient_estimated_bytes == 101 + extra + p1.RUN_BYTES + retained
    assert value.option_run_bytes == p1.RUN_BYTES and value.header_metadata_copy_estimated_bytes == 0
    assert value.options["route"] == p1.PIPELINE and value.options["declared"] == []
    assert [row["relative_path"] for row in pins] == list(p1.WRITERS)


def test_the_gas_probability_adds_its_budget_on_phase1(monkeypatch):
    task, config, wrapped = phase1_task_TESTONLY()
    base = estimate_TESTONLY(task, config, wrapped, monkeypatch)
    pins = []
    value = estimate_TESTONLY(task, dict(config, **PROBABILITY), wrapped, monkeypatch, pins=pins)
    budget = oc.probability_budget(PROBABILITY, canvas_hw=tuple(run_layout_TESTONLY()["phase_shape_hw"]),
                                   precision="float32")
    assert value.option_frame_bytes == base.option_frame_bytes + budget["frame_bytes"]
    assert value.options["declared"] == ["gas_probability_storage"]
    assert [row["relative_path"] for row in pins] == [*p1.WRITERS, *oc.WRITERS["gas_probability_storage"]]


def test_photographs_add_an_rgba_bound_for_every_frame(monkeypatch):
    task, config, wrapped = phase1_task_TESTONLY()
    base = estimate_TESTONLY(task, config, wrapped, monkeypatch)
    pins = []
    request = {key: synthetic_bound("photograph-" + key + ".json") for key in (
        "native_geometry", "geometry_certificate", "current_geometry", "saved_native_review")}
    request["geometry_producer_sha256"] = "a" * 64
    photos = dict(config, photographs={task["ordered_case_ids"][0]: request})
    value = estimate_TESTONLY(task, photos, wrapped, monkeypatch, pins=pins)
    height, width = run_layout_TESTONLY()["phase_shape_hw"]
    assert value.option_frame_bytes == base.option_frame_bytes + p1.png_bound(height, width, 4) + p1.RECEIPTS_BYTES
    assert pins[-1]["relative_path"] == p1.PHOTOGRAPH_WRITERS[0]
    with pytest.raises(ValueError, match="qualified head and route: Optional photographs must name only the selected case IDs"):
        estimate_TESTONLY(task, dict(config, photographs={"other_f1": request}), wrapped, monkeypatch)


@pytest.mark.parametrize("key", sorted(oc.REFUSED))
def test_other_optional_outputs_are_refused_on_phase1_before_any_read(monkeypatch, key):
    task, config, wrapped = phase1_task_TESTONLY()
    if key == "storage":
        # Preserve this parameter ID: P1 storage is mandatory native evidence,
        # rather than the unrelated phase-two output with the same key.
        value = estimate_TESTONLY(task, config, wrapped, monkeypatch)
        assert value.options["route"] == p1.PIPELINE and value.options["declared"] == []
        assert config["storage"]["resolution_mm"] == "native"
        return
    monkeypatch.setattr(cap, "_profile_writer_identity", lambda *args: pytest.fail("No writer read"))
    with pytest.raises(ValueError, match="qualified head and route:") as error:
        cap._bound_result_estimate(None, "TESTONLY", task, dict(config, **{key: {"TESTONLY": "request"}}),
                                   wrapped, {}, repository="TESTONLY", metadata_rows=None)
    assert key in str(error.value)


@pytest.mark.parametrize("key,value", [("backend", "cuda"), ("precision", "float64"), ("commit", "f" * 40),
                                       ("pipeline", "lattice-phase-n1")])
def test_phase1_never_opens_another_head_or_pipeline(monkeypatch, key, value):
    task, config, wrapped = phase1_task_TESTONLY()
    task[key] = value
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, config, wrapped, monkeypatch)


def test_phase1_needs_a_native_phase_n1_qualification(monkeypatch):
    task, config, wrapped = phase1_task_TESTONLY()
    wrapped["scope"] = dict(wrapped["scope"], pipeline=p1.PIPELINE)
    with pytest.raises(ValueError, match="qualified head and route"):
        estimate_TESTONLY(task, config, wrapped, monkeypatch)


def test_a_phase1_task_never_declares_a_storage_profile(tmp_path, monkeypatch):
    task, config, _ = phase1_task_TESTONLY()
    task.update(storage_profile="lean", format="darsia-alias-task-v1", format_version=2)
    with pytest.raises(ValueError, match="Phase one writes no N1 frame products"):
        cap.load_estimate(None, "TESTONLY", task, config, repository=tmp_path)


def test_a_phase1_task_without_preparation_measurements_is_refused(monkeypatch):
    task, config, wrapped = phase1_task_TESTONLY()
    measurement = wrapped["result_profile"]["measurement"]
    rows = [dict(row, relative_path=row["relative_path"].replace("preparation", "stage"))
            for row in measurement["inventory"]]
    measurement.update(inventory=rows)
    with pytest.raises(ValueError, match="qualified head and route: The qualified profile measures no preparation"):
        estimate_TESTONLY(task, config, wrapped, monkeypatch)


def test_the_series_route_is_unchanged_by_the_phase1_rule(monkeypatch):
    qualified, config, wrapped = bound_output_profile_TESTONLY()
    monkeypatch.setattr(p1, "phase1_estimate", lambda *args, **kwargs: pytest.fail("Not for the series route"))
    value = estimate_TESTONLY(qualified, config, wrapped, monkeypatch, layout=native_layout())
    assert type(value) is cap.SpaceEstimate


def phase1_cache_TESTONLY(f):
    """A phase-one task on the qualified head, its boundary modules in the qualified closure."""
    root = f.adapter.runtime.repository / "src" / "darsia" / "fluidflower"
    pins = [dict(path=str(path), sha256="1" * 64, size_bytes=1) for path in (
        root / "pipeline" / "phase_statistics.py", root / "pipeline" / "staged_data.py",
        root / "pipeline" / "phase1_protocol.py", root / "pipeline" / "run_io.py",
        root / "pipeline" / "phase1_capacity.py", root / "storage_outputs.py",
        root / "storage_original_photo.py", root / "phase" / "probability_storage.py")]
    return dict(other_cache_task_TESTONLY(f), pipeline=p1.PIPELINE), pins


def test_the_cache_side_admits_a_full_phase1_task_with_its_boundary_in_the_closure(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task, pins = phase1_cache_TESTONLY(f)
    with pytest.raises(CachePolicyError, match="startup writer/source/runtime closure"):
        control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    f.profile["producer_source_closure"] = f.profile["producer_source_closure"] + pins
    f.adapter.lease.profile["module_closure"] = f.adapter.lease.profile["module_closure"] + pins
    maximum, binding, profile, _, _ = control.applicable_job_cache_profile(f.policy, f.adapter, task, None)
    assert maximum == 0 and binding == f.service and profile is f.profile


def test_the_cache_side_refuses_a_phase1_task_with_a_storage_profile(bound_cache_TESTONLY):
    f = bound_cache_TESTONLY
    task, pins = phase1_cache_TESTONLY(f)
    f.profile["producer_source_closure"] = f.profile["producer_source_closure"] + pins
    f.adapter.lease.profile["module_closure"] = f.adapter.lease.profile["module_closure"] + pins
    with pytest.raises(ValueError, match="Phase one writes no N1 frame products"):
        control.applicable_job_cache_profile(f.policy, f.adapter, dict(task, storage_profile="full"), None)
    assert f.adapter.profiles == {}


@pytest.mark.parametrize("change", [dict(backend="cuda"), dict(commit="4" * 40),
                                    dict(pipeline="lattice-phase-n1")])
def test_the_cache_side_refuses_cuda_another_head_and_another_pipeline(bound_cache_TESTONLY, change):
    f = bound_cache_TESTONLY
    task, pins = phase1_cache_TESTONLY(f)
    f.profile["producer_source_closure"] = f.profile["producer_source_closure"] + pins
    f.adapter.lease.profile["module_closure"] = f.adapter.lease.profile["module_closure"] + pins
    with pytest.raises(CachePolicyError, match="qualified head and route"):
        control.applicable_job_cache_profile(f.policy, f.adapter, dict(task, **change), None)
    assert f.adapter.profiles == {}
