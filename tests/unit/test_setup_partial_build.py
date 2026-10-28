"""Owned partial output moves transactionally; no completed rig is admitted."""
from pathlib import Path

import pytest

from darsia.fluidflower.pipeline import job_queue
from darsia.fluidflower.rig_raw_write import OwnedRawOutput
from .test_setup_failed_build import failed_build, unstarted, setup_job, controls
from .test_setup_job_reprepare import _retry, _bytes


@pytest.mark.parametrize("failed_build", [True], indirect=True)
@pytest.mark.parametrize("outcome", ["failed", "stopped"])
def test_partial_build_archived_in_same_job_and_original_target_is_fresh(failed_build, outcome):
    case = failed_build
    request = job_queue.read_json(case.job_root / "setup-controls/build/sample/raw-build-request.json")
    source = Path(request["output"]["destination"])
    before = _bytes(source)
    applied = _bytes(case.job_root / "exports")
    if outcome == "stopped":
        task = job_queue.read_json(case.job_root / "claimed/private-host-cpu/sample.json")
        token = task["attempt_id"]
        lease = case.job_root / "leases" / ("sample." + token)
        attempt = case.job_root.parent / case.job_root.name / "sample/.attempts" / token
        for path in (case.job_root / "failed/sample.json", case.job_root / "publishing/sample",
                     case.job_root / "decisions" / ("sample." + token + ".json"),
                     lease / "publication.json", lease / "failure_artifacts/failed.json", attempt / "failed.json"):
            path.unlink()
        (lease / "failure_artifacts").rmdir()
    result = _retry(case)
    archived = Path(result["history"]) / "partial-builds/sample"
    assert _bytes(archived) == before and not source.exists()
    assert _bytes(case.job_root / "exports") == applied
    assert len(case.registry.jobs(case.series, "synthetic")) == 1
    # The original exclusive writer can start the same requested destination.
    owner = OwnedRawOutput(source, request["output"], operation_id="b" * 32,
        request=job_queue.descriptor(case.job_root / "setup-controls/build/sample/raw-build-request.json"))
    assert owner.destination == source and (source / "raw_build_owner.json").is_file()
    assert not (source / "baseline_native_renderer.json").exists()


@pytest.mark.parametrize("failed_build", [True], indirect=True)
def test_partial_archive_rolls_back_when_normal_reprepare_fails(failed_build):
    case = failed_build
    source = Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"])
    before = _bytes(source)
    old_job = (case.job_root / "job.json").read_bytes()
    def broken(*args, **kwargs):
        assert not source.exists()
        raise RuntimeError("replacement preparation failed")
    with pytest.raises(RuntimeError, match="replacement preparation failed"):
        _retry(case, producer=broken)
    assert _bytes(source) == before
    assert (case.job_root / "job.json").read_bytes() == old_job
    assert (case.job_root / "done/setup-apply.json").is_file()


@pytest.mark.parametrize("failed_build", [True], indirect=True)
@pytest.mark.parametrize("fault", ["certificate", "save-completed", "foreign-owner", "changed-image"])
def test_completed_or_foreign_partial_is_preserved(failed_build, fault, monkeypatch):
    from darsia.fluidflower import setup_partial_build
    case = failed_build
    source = Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"])
    if fault == "certificate":
        (source / "baseline_native_renderer.json").write_text('{}')
    elif fault == "save-completed":
        (source / "raw_save_completed.json").write_text('{}')
    elif fault == "foreign-owner":
        path = source / "raw_build_owner.json"
        document = job_queue.read_json(path)
        document["request"]["sha256"] = "f" * 64
        path.write_text(__import__('json').dumps(document))
    else:
        original = setup_partial_build.verify
        def changed(row, **kwargs):
            (source / "figures/corrected_baseline.png").write_bytes(b"different image")
            original(row, **kwargs)
        monkeypatch.setattr(setup_partial_build, "verify", changed)
    old_job = (case.job_root / "job.json").read_bytes()
    with pytest.raises(ValueError):
        _retry(case)
    assert source.exists() and (case.job_root / "job.json").read_bytes() == old_job
