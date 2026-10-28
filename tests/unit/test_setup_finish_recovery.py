"""Finish Setup retains its exact pending optical task through early Build retry."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_control
from darsia.fluidflower.pipeline import job_queue
from darsia.fluidflower.rig_raw_write import OwnedRawOutput
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_image_preparation_update import initial_image_job
from .test_setup_failed_build import failed_build
from .test_setup_job_reprepare import _retry, _bytes, _doc


@pytest.fixture
def stopped_finish_setup(setup_job, monkeypatch, request):
    if getattr(request, "param", False):
        monkeypatch.setenv("DARSIA_TEST_FOLDER_SAVE_BEFORE_FAILURE", "1")
    case = initial_image_job(setup_job, monkeypatch)
    case = failed_build.__wrapped__(case, monkeypatch, SimpleNamespace(param=True))
    root = case.job_root
    task = job_queue.read_json(root / "claimed/private-host-cpu/sample.json")
    token = task["attempt_id"]
    lease = root / "leases" / ("sample." + token)
    attempt = root.parent / root.name / "sample/.attempts" / token
    # An ordinary control stop publishes no failure or scientific completion;
    # its real claim, operation progress, and worker retirement remain intact.
    for path in (root / "failed/sample.json", root / "publishing/sample",
                 root / "decisions" / ("sample." + token + ".json"),
                 lease / "publication.json", lease / "failure_artifacts/failed.json",
                 attempt / "failed.json"):
        path.unlink()
    (lease / "failure_artifacts").rmdir()
    previous = job_control.read_control(root)
    job_control.publish_control(root, "pause", machine="private-host",
                                expected_previous=previous.get("command"))
    return case


@pytest.mark.parametrize("stopped_finish_setup", [False, True], indirect=True)
def test_finish_setup_code_update_retains_pending_typed_images_and_apply(stopped_finish_setup):
    case = stopped_finish_setup
    root = case.job_root
    controls_before = _bytes(root / "setup-controls")
    exports_before = _bytes(root / "exports")
    applied_before = (root / "done/setup-apply.json").read_bytes()
    image_before = job_queue.read_json(root / "queue/setup-image-preparation.json")
    request_file = root / "setup-controls/build/sample/raw-build-request.json"
    request = job_queue.read_json(request_file)
    source = Path(request["output"]["destination"])
    partial_before = _bytes(source)
    saved_before = partial_before.get("raw_save_completed.json")

    result = _retry(case)

    history = Path(result["history"])
    assert result["complete"] and result["paused"]
    assert len(case.registry.jobs(case.series, "synthetic")) == 1
    assert (history / "previous/done/setup-apply.json").read_bytes() == applied_before
    assert _bytes(root / "setup-controls") == controls_before
    assert _bytes(root / "exports") == exports_before
    assert _bytes(history / "partial-builds/sample") == partial_before
    if saved_before is not None:
        receipt = job_queue.read_json(history / "receipt.json")
        [saved] = receipt["failed_preparation"]["partial_builds"]
        assert saved["completed_save"]["sha256"] == job_queue.descriptor(
            history / "partial-builds/sample/raw_save_completed.json")["sha256"]
        assert not (history / "partial-builds/sample/baseline_native_renderer.json").exists()
    assert not source.exists()
    assert {path.name for path in (root / "queue").glob("*.json")} == {
        "setup-apply.json", "setup-image-preparation.json"}
    image_after = job_queue.read_json(root / "queue/setup-image-preparation.json")
    assert image_after["commit"] == case.identity["commit"]
    assert image_after["input_catalog"] == image_before["input_catalog"]
    assert image_after["attempt_id"] != image_before["attempt_id"]
    assert {key: value for key, value in image_after.items()
            if key not in {"commit", "environment_lock", "attempt_id"}} == {
        key: value for key, value in image_before.items()
        if key not in {"commit", "environment_lock", "attempt_id"}}
    owner = OwnedRawOutput(source, request["output"], operation_id="b" * 32,
                           request=job_queue.descriptor(request_file))
    assert owner.destination == source


@pytest.mark.parametrize("stopped_finish_setup", [True], indirect=True)
def test_finish_setup_completed_inner_save_rolls_back_on_retry_failure(stopped_finish_setup):
    case = stopped_finish_setup
    source = Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"])
    saved_before, metadata_before = _bytes(source), _bytes(case.job_root)
    def broken(*args, **kwargs):
        assert not source.exists()
        raise RuntimeError("original replacement producer failed")
    with pytest.raises(RuntimeError, match="original replacement producer failed"):
        _retry(case, producer=broken)
    assert _bytes(source) == saved_before
    # The original capacity lock's new acquisition nonce is operational only.
    stable = lambda rows: {key: value for key, value in rows.items()
                           if not key.endswith("/admission.lock")}
    assert stable(_bytes(case.job_root)) == stable(metadata_before)


@pytest.mark.parametrize("stopped_finish_setup", [True], indirect=True)
@pytest.mark.parametrize("fault", ["native-certificate", "different-request"])
def test_finish_setup_does_not_reset_qualified_or_foreign_save(stopped_finish_setup, fault):
    case = stopped_finish_setup
    source = Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"])
    if fault == "native-certificate":
        _doc(source / "baseline_native_renderer.json", dict(fixture="qualification present"))
    else:
        path = source / "raw_save_completed.json"
        saved = job_queue.read_json(path)
        saved["request"]["sha256"] = "f" * 64
        _doc(path, saved)
    before, saved_before = _bytes(case.job_root), _bytes(source)
    with pytest.raises(ValueError):
        _retry(case)
    assert _bytes(case.job_root) == before and _bytes(source) == saved_before


@pytest.mark.parametrize("fault", ["extra-work", "image-source"])
def test_finish_setup_retry_refuses_other_work_and_changed_image_task(stopped_finish_setup, fault):
    case = stopped_finish_setup
    root = case.job_root
    if fault == "extra-work":
        _doc(root / "queue/other-science.json", dict(run="other-science"))
    else:
        path = root / "queue/setup-image-preparation.json"
        task = job_queue.read_json(path)
        task["input_catalog"] = []
        _doc(path, task)
    before = _bytes(root)
    with pytest.raises(ValueError):
        _retry(case)
    assert _bytes(root) == before
