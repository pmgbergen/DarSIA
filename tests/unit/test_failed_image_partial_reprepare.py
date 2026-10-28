"""Real failed/retired queue evidence, preserved optical files and retry rollback."""
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from darsia.fluidflower.pipeline import job_queue, worker, setup_colour_paths
from darsia.fluidflower.setup_completed_builds import retained_operations
from darsia.fluidflower.jobs_metadata import JobSnapshotReader
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_job_reprepare import _retry, _bytes, _doc
from .test_setup_image_preparation_update import completed_geometry, save_independent_choice
from .test_setup_completed_builds import _run


def failed_partial(case, monkeypatch):
    native = _doc(Path(case.plan["native_parameters"]["path"]),
                  dict(format="private-native-parameter-metadata", source={}))
    case.plan["native_parameters"] = native
    for build in case.plan["build_plans"]:
        build["native_parameters"] = deepcopy(native)
    case = completed_geometry(case, monkeypatch)
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_independent_choice(case)
    _retry(case)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_MODE", "failed")
    monkeypatch.setenv("DARSIA_TEST_FOLDER_IMAGE_PARTIAL_BASIS", "1")
    _run(case, monkeypatch)
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    claim = next((case.job_root / "claimed").glob("*/setup-image-preparation.json"))
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    case.partial = Path(config["output_root"]) / job_queue.read_json(claim)["attempt_id"]
    case.failure = (case.job_root / "failed/setup-image-preparation.json").read_bytes()
    case.partial_bytes = _bytes(case.partial)
    assert (case.partial / "photometry/independent_photometry.json").is_file()
    monkeypatch.setattr(setup_colour_paths, "preflight_setup_colour_paths", lambda *a, **k:
        dict(frame_selection=dict(runs={run: [] for run in case.plan["runs"]}),
             optical_sources={run: {} for run in case.plan["runs"]}))
    case.identity["commit"] = "8" * 40
    return case


def test_failed_partial_requeue_preserves_samples_and_real_failure(setup_job, monkeypatch):
    case = failed_partial(setup_job, monkeypatch)
    result = _retry(case)
    pin = result["reused_image_basis"]
    partial = job_queue.read_json(pin["path"])
    assert partial["complete"] is False
    assert Path(partial["source_root"]) == case.partial
    assert _bytes(case.partial) == case.partial_bytes
    assert Path(partial["failed_attempt"]["path"]).read_bytes() == case.failure
    retired = job_queue.read_json(partial["retirement"]["retirement"]["path"])
    assert retired["all_absent"] and retired["complete"]
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
    snapshot = JobSnapshotReader()(case.registry.jobs(case.series, "synthetic")[0])
    assert not snapshot["errors"]
    assert not snapshot["complete"]
    assert snapshot["completed_operations"] == len(case.plan["runs"]) + 1


def test_failed_partial_retry_rollback_keeps_original_failure_and_sample_paths(setup_job, monkeypatch):
    case = failed_partial(setup_job, monkeypatch)
    from darsia.fluidflower.job_control import read_control, publish_control
    control = read_control(case.job_root)
    publish_control(case.job_root, "pause", machine="private-host", expected_previous=control["command"])
    stable = lambda: {name: value for name, value in _bytes(case.job_root).items()
                      if not name.endswith("admission.lock")}
    before = stable()
    def fail(*a, **k):
        raise OSError("synthetic registry publication failure")
    monkeypatch.setattr(case.registry, "register_reprepared_setup", fail)
    with pytest.raises(OSError, match="registry publication failure"):
        _retry(case)
    assert stable() == before
    assert _bytes(case.partial) == case.partial_bytes
    assert (case.job_root / "failed/setup-image-preparation.json").read_bytes() == case.failure


def test_second_failed_retry_retains_original_fits_and_both_partial_generations(setup_job, monkeypatch):
    case = failed_partial(setup_job, monkeypatch)
    case.identity["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    first = _retry(case)
    original_pin = first["reused_image_basis"]
    first_document = job_queue.read_json(original_pin["path"])
    _run(case, monkeypatch)
    claim = next((case.job_root / "claimed").glob("*/setup-image-preparation.json"))
    config = job_queue.read_json(case.job_root / "setup-controls/image-operation.json")
    newest = Path(config["output_root"]) / job_queue.read_json(claim)["attempt_id"]
    assert not (newest / "photometry").exists()
    newest_bytes = _bytes(newest)
    case.identity["commit"] = "7" * 40
    second = _retry(case)
    document = job_queue.read_json(second["reused_image_basis"]["path"])
    assert document["prior_reuse"] == original_pin
    assert document["photometry_record"] == first_document["photometry_record"]
    assert document["image_basis"] == first_document["image_basis"]
    assert Path(document["source_root"]) == newest
    assert _bytes(newest) == newest_bytes
    assert _bytes(case.partial) == case.partial_bytes
    assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
