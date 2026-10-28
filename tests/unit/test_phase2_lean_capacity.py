"""A version-two sealed lean task uses its own exact resource DATA, never JPEG ratios."""
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput
from .test_lean_capacity import make_sealed_configuration


def _prepared(tmp_path):
    config = make_sealed_configuration(tmp_path)
    task = dict(format="darsia-alias-task-v1", format_version=2, storage_profile="lean", pipeline="n1",
        run=config["run"], ordered_case_ids=[frame["case_id"] for frame in config["frames"]],
        backend="cpu", precision="float32", commit="b" * 40)
    rows = []
    for key in ("physical_reference", "floor_curve", "depth_points"):
        bound = config["shared"][key]
        if isinstance(bound, dict):
            rows.append(ResolvedInput(dict(sha256=bound["sha256"], bytes=bound["size_bytes"]),
                                      Path(bound["path"]), ("mass." + key,)))
    return task, config, rows


def test_sealed_staged_lean_reserves_full_fallback_and_run_store_on_actual_input_headers(tmp_path, monkeypatch):
    task, config, rows = _prepared(tmp_path)
    before = deepcopy(config)
    monkeypatch.setattr(capacity, "_lean_writer_identity", lambda *args: [{"exact": "writer-closure"}])
    # This route may not touch the old JPEG-size estimate or a full result profile.
    from darsia.fluidflower import storage_profile
    monkeypatch.setattr(storage_profile, "run_estimate", lambda *args, **kwargs: pytest.fail("Legacy JPEG estimate reused"))
    value = capacity.load_estimate(None, "sample-series", task, config, repository=tmp_path, metadata_rows=rows)
    assert value.storage_profile == "lean" and value.colour_intermediates == "keep"
    assert value.pipeline == "n1" and config == before
    assert value.case_count == len(task["ordered_case_ids"])
    assert value.products_per_frame_estimated_bytes == value.estimate["admitted_frame_bytes"]
    assert value.run_constant_estimated_bytes == value.estimate["admitted_run_bytes"]
    assert value.needed(reserve_bytes=0) == value.estimate["admitted_job_bytes"] + value.estimate["inflight_bytes"]
    assert value.estimate["production_permission_granted"] is False
    assert "no compression gain" in value.model


@pytest.mark.parametrize("change", ["version", "backend", "precision", "order", "shape", "role", "duplicate_role"])
def test_uncovered_task_grid_or_rooted_input_identity_refuses_before_writer_release(tmp_path, monkeypatch, change):
    task, config, rows = _prepared(tmp_path)
    monkeypatch.setattr(capacity, "_lean_writer_identity", lambda *args: pytest.fail("Invalid input reached writer release"))
    if change == "version":
        task["format_version"] = 1
    elif change == "backend":
        task["backend"] = "cuda"
    elif change == "precision":
        task["precision"] = "float64"
    elif change == "order":
        task["ordered_case_ids"].reverse()
    elif change == "shape":
        config["frames"][-1]["current"]["phase_shape_hw"] = [3, 4]
    elif change == "role":
        rows[0].reference["sha256"] = "e" * 64
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        capacity.load_estimate(None, "sample-series", task, config, repository=tmp_path, metadata_rows=rows)


def test_staged_lean_writer_inventory_includes_each_existing_optional_producer_and_guard():
    paths = capacity._lean_writers("n1", dict(storage={}, pixel_domain=dict(n16={})))
    assert len(paths) == len(set(paths))
    for name in ("pipeline/lean_capacity.py", "pipeline/io_ops.py", "pipeline/job_queue.py",
                 "pipeline/staged_data.py", "storage_profile.py", "storage_n1_outputs.py",
                 "storage_extensive.py", "storage_extensive_io.py", "mass/pixel_pipeline.py",
                 "mass/pixels.py", "geometry/pixels.py"):
        assert "src/darsia/fluidflower/" + name in paths
