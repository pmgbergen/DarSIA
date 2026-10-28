"""Real queue/publication/retirement, synthetic scientific-operation boundary."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import job_control, setup_failed_build
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower.pipeline import job_queue
from .test_setup_job_reprepare import (
    unstarted, setup_job, controls, _doc, _retry, _completed_apply_case, _bytes,
)


PARSER_TRACE = '''Traceback (most recent call last):
  File "C:/app/fluidflower_config.py", line 119, in __init__
    self.registry.load(path)
  File "C:/app/data_registry.py", line 197, in load
    raise ValueError()
ValueError: DataRegistry: duplicate key 'calibration1' in [[data_interval]]. Names must be globally unique.
'''


def test_build_metadata_progress_is_not_restricted_to_named_error_or_stage(tmp_path):
    token = "a" * 32
    event = dict(state="setup_progress", operation="build", run="sample", step="read_protocols")
    path = tmp_path / "progress_00000000.json"
    _doc(path, dict(run="sample", attempt_id=token, sequence=0, event=event))
    assert setup_failed_build._progress(tmp_path, token, "sample") == {path.name}
    for changed in (dict(state="frame_completed"), dict(operation="apply"), dict(run="another")):
        _doc(path, dict(run="sample", attempt_id=token, sequence=0, event=dict(event, **changed)))
        with pytest.raises(ValueError):
            setup_failed_build._progress(tmp_path, token, "sample")


@pytest.fixture
def failed_build(unstarted, monkeypatch, request):
    from darsia.fluidflower.pipeline import worker
    from .test_folder_cache_worker import _activate
    from ._folder_worker_fixture import work
    make_partial = bool(getattr(request, "param", False))
    case = unstarted
    _completed_apply_case(case, monkeypatch)
    root = case.job_root
    config = job_queue.read_json(root / "setup-controls/apply-operation.json")
    completed = job_queue.read_json(root / "setup-controls/apply-complete.json")
    children = []
    for build in case.plan["build_plans"]:
        run = build["run_id"]
        Path(build["request_template"]["output"]["intended_root"]).mkdir(parents=True, exist_ok=True)
        row = next(row for row in completed["runs"] if row["run_id"] == run)
        directory = root / "setup-controls/build" / run
        request = job_queue.descriptor(directory / "raw-build-request.json")
        operation = dict(format=config["format"], run=run, operation="build",
            **{key: deepcopy(config[key]) for key in ("project_root", "project_manifest", "adopted_revision", "series_manifest", "setup_plan")},
            **{key: deepcopy(build[key]) for key in ("native_parameters", "target_machine", "metadata_output", "output_scope")},
            configuration_files=row["configuration_files"], raw_build_request=request,
            apply_completion=row["apply_completion"],
            coordinator_completion=job_queue.descriptor(root / "setup-controls/apply-complete.json"))
        pin = _doc(directory / "operation.json", operation)
        children.append(dict(run_id=run, configuration=pin, raw_build_request=request,
                             target_machine=build["target_machine"]))
    core = dict(format="darsia-series-setup-result-v1", complete=True, pipeline="series-setup",
        run="setup-apply", operation="apply", backend="cpu", precision="float32",
        setup_plan=config["setup_plan"], worker_references_closed=True,
        coordinator_completion=job_queue.descriptor(root / "setup-controls/apply-complete.json"),
        guarded_geometry=[], children=children, Build_run=False, optical_calibration_run=False)
    source = Path(case.root) / "synthetic-apply-result.json"
    _doc(source, core)
    monkeypatch.setenv("DARSIA_TEST_FOLDER_COMPLETED_APPLY", str(source))
    monkeypatch.setenv("DARSIA_TEST_FOLDER_PARTIAL_BUILD", "1" if make_partial else "0")
    prior = job_control.read_control(root)
    job_control.publish_control(root, "run", machine="private-host", expected_previous=prior.get("command"))
    begin_launch_request(root, machine="private-host", backend="cpu", workers=1, kind="local")
    _activate(case.stores, root, monkeypatch)
    monkeypatch.setattr(worker, "_work", work)
    job = job_queue.read_json(root / "job.json")
    args = SimpleNamespace(job=root, machine="private-host", backend="cpu", workers=1,
        once=False, exit_when_idle=True, watchdog=False, watchdog_seconds=30,
        _alias_stores=case.stores, stores=case.stores.source["path"], _cache_terminal_events=[])
    with (root / "logs/private-host-cpu.log").open("a", encoding="utf-8") as log:
        worker._parent(args, "private-host-cpu", job, job["commit"], log)
    assert (root / "done/setup-apply.json").is_file()
    assert (root / "failed/sample.json").is_file()
    monkeypatch.setattr(worker, "_same_process", lambda _identity: None)
    return case


def test_same_job_code_update_preserves_completed_apply_and_retired_build(failed_build):
    case = failed_build
    controls = _bytes(case.job_root / "setup-controls")
    exports = _bytes(case.job_root / "exports")
    done = (case.job_root / "done/setup-apply.json").read_bytes()
    failed = (case.job_root / "failed/sample.json").read_bytes()
    result = _retry(case)
    history = Path(result["history"])
    receipt = job_queue.read_json(history / "receipt.json")["failed_preparation"]
    assert receipt["format"] == setup_failed_build.FORMAT
    assert len(receipt["attempts"]) == 2
    assert len(receipt["retirement"]) == 2
    assert (history / "previous/done/setup-apply.json").read_bytes() == done
    assert (history / "previous/failed/sample.json").read_bytes() == failed
    assert _bytes(case.job_root / "setup-controls") == controls
    assert _bytes(case.job_root / "exports") == exports
    assert (case.job_root / "queue/setup-apply.json").is_file()
    assert len(case.registry.jobs(case.series, "synthetic")) == 1
    for row in receipt["archived_attempts"]:
        assert Path(row["archived_output"]).is_dir()
        assert not Path(row["original_output"]).exists()


def test_cancelled_build_before_output_uses_its_real_claim_and_retirement(failed_build):
    """Model a control-stop publication: no failed/completed result is published."""
    case = failed_build
    root = case.job_root
    task = job_queue.read_json(root / "claimed/private-host-cpu/sample.json")
    token = task["attempt_id"]
    lease = root / "leases" / ("sample." + token)
    output = root.parent / root.name / "sample/.attempts" / token
    for path in (root / "failed/sample.json", root / "publishing/sample",
                 root / "decisions" / ("sample." + token + ".json"),
                 lease / "publication.json", lease / "failure_artifacts/failed.json",
                 output / "failed.json"):
        path.unlink()
    (lease / "failure_artifacts").rmdir()
    result = _retry(case)
    receipt = job_queue.read_json(Path(result["history"]) / "receipt.json")["failed_preparation"]
    assert len(receipt["retirement"]) == 2
    assert (Path(result["history"]) / "previous/done/setup-apply.json").is_file()
    assert (root / "queue/setup-apply.json").is_file()


@pytest.mark.parametrize("fault", ["output", "checkpoint", "partial", "source", "unknown-progress", "live"])
def test_early_build_recovery_refuses_science_changes_or_unproven_retirement(failed_build, monkeypatch, fault):
    from darsia.fluidflower.pipeline import worker
    case = failed_build
    before = (case.job_root / "job.json").read_bytes()
    if fault == "output":
        Path(case.plan["build_plans"][0]["request_template"]["output"]["destination"]).mkdir(parents=True)
    elif fault == "partial":
        (case.job_root / "done/.setup-apply.json.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.partial").write_text("not the completed publication")
    elif fault == "source":
        (case.job_root / "setup-controls/build/sample/raw-build-request.json").write_text("{}")
    elif fault == "unknown-progress":
        task = job_queue.read_json(case.job_root / "claimed/private-host-cpu/sample.json")
        path = case.job_root / "leases" / ("sample." + task["attempt_id"]) / "progress_00000000.json"
        value = job_queue.read_json(path)
        value["event"]["state"] = "frame_completed"
        _doc(path, value)
    elif fault == "live":
        monkeypatch.setattr(worker, "_same_process", lambda _: True)
    else:
        task = job_queue.read_json(case.job_root / "claimed/private-host-cpu/sample.json")
        output = case.job_root.parent / case.job_root.name / "sample/.attempts" / task["attempt_id"]
        (output / "numeric_checkpoint.npz").write_bytes(b"saved work")
    with pytest.raises(ValueError):
        _retry(case)
    assert (case.job_root / "job.json").read_bytes() == before
    assert (case.job_root / "done/setup-apply.json").is_file()
