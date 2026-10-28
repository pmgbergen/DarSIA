"""Rebuild colour paths from genuine Setup completion without repeating geometry."""
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from darsia.fluidflower import setup_job_reprepare as subject
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from darsia.fluidflower.pipeline import job_queue, worker
from darsia.fluidflower.setup_completed_builds import retained_operations
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_job_reprepare import _retry, _bytes, _doc
from .test_setup_image_preparation_update import (
    initial_image_job, completed_geometry, save_independent_choice,
)
from .test_setup_completed_builds import _apply, _run


def complete_images(setup_job, monkeypatch, *, local=False):
    native = _doc(Path(setup_job.plan["native_parameters"]["path"]),
                  dict(format="private-native-parameter-metadata", source={}))
    setup_job.plan["native_parameters"] = native
    for build in setup_job.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    if local:
        case = initial_image_job(setup_job, monkeypatch)
        case.identity["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        _retry(case)
        _apply(case, monkeypatch)
        for build in case.plan["build_plans"]:
            Path(build["request_template"]["output"]["intended_root"]).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("DARSIA_TEST_FOLDER_COMPLETED_BUILD_RUNS", ",".join(case.plan["runs"]))
    else:
        case = completed_geometry(setup_job, monkeypatch)
        case.identity["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        save_independent_choice(case)
        _retry(case)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "complete")
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_COMPLETE_BASIS", "1")
    _run(case, monkeypatch)
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"], snapshot["errors"]
    image = next(row for row in snapshot["runs"] if row["operation"] == "image-preparation")
    case.completed_basis = image["result"]["image_preparation"]
    case.identity["commit"] = "9" * 40
    return case


def test_setup_image_preparation_spreads_its_runs_over_the_machine(setup_job, monkeypatch):
    """The worker gives Setup image preparation automatic stage workers, not its single slot."""
    case = complete_images(setup_job, monkeypatch)
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    recorded = [job_queue.read_json(path) for path in Path(config["output_root"]).glob("*/corrected-baseline.json")]
    assert recorded and {row["stage_workers"] for row in recorded} == {0}


@pytest.mark.parametrize("local", [False, True])
def test_completed_images_requeue_only_paths_and_pin_original_basis(setup_job, monkeypatch, local):
    case = complete_images(setup_job, monkeypatch, local=local)
    original_job = (case.job_root / "job.json").read_bytes()
    original_basis = Path(case.completed_basis["path"]).read_bytes()
    outputs = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
               for build in case.plan["build_plans"]}
    result = _retry(case)
    assert result["reused_image_basis"] == case.completed_basis
    history = Path(result["history"])
    assert (history / "previous/job.json").read_bytes() == original_job
    assert (history / "previous/done/setup-image-preparation.json").is_file()
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    assert config["reuse_image_basis"] == case.completed_basis
    assert config["colour_path_scope"] == "per-run"
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    receipt = job_queue.read_json(history / "receipt.json")
    assert receipt["image_preparation_refresh"]["reused_image_basis"] == case.completed_basis
    assert Path(case.completed_basis["path"]).read_bytes() == original_basis
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == outputs[build["run_id"]]
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert not snapshot["errors"]
    assert snapshot["completed_operations"] == len(case.plan["runs"]) + 1
    assert not snapshot["complete"]


def test_completed_image_requeue_rolls_back_with_original_completion(setup_job, monkeypatch):
    case = complete_images(setup_job, monkeypatch)
    before = _bytes(case.job_root)
    basis_bytes = Path(case.completed_basis["path"]).read_bytes()
    def fail_registration(*args, **kwargs):
        raise OSError("synthetic registry publication failure")
    monkeypatch.setattr(case.registry, "register_reprepared_setup", fail_registration)
    with pytest.raises(OSError, match="registry publication failure"):
        _retry(case)
    assert _bytes(case.job_root) == before
    assert Path(case.completed_basis["path"]).read_bytes() == basis_bytes
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert snapshot["complete"] and not snapshot["errors"]


def test_completed_image_capture_rejects_changed_basis(setup_job, monkeypatch):
    case = complete_images(setup_job, monkeypatch)
    Path(case.completed_basis["path"]).write_bytes(b'{"complete":true}')
    before = _bytes(case.job_root)
    with pytest.raises((ValueError, OSError)):
        _retry(case)
    assert _bytes(case.job_root) == before


def test_changed_image_executor_keeps_geometry_but_does_not_reuse_pixels(setup_job, monkeypatch):
    case = complete_images(setup_job, monkeypatch)
    from darsia.fluidflower.native import raw_rig_source
    old_identity = raw_rig_source.execution_identity()
    changed_identity = dict(old_identity, full_photometry="0" * 64)
    monkeypatch.setattr(raw_rig_source, "execution_identity", lambda: changed_identity)
    original = Path(case.completed_basis["path"]).read_bytes()
    result = _retry(case)
    assert not result.get("reused_image_basis")
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    assert "reuse_image_basis" not in config
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    assert Path(case.completed_basis["path"]).read_bytes() == original
