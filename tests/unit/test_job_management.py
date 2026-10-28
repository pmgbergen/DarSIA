import json
from pathlib import Path

import pytest

from darsia.fluidflower import job_management as manage, native_jobs_registry as registry
from darsia.fluidflower.job_launch_state import begin_launch_request
from darsia.fluidflower import job_control
from darsia.fluidflower.job_launch_state import declare_requests_stopped, read_launch_requests
from darsia.fluidflower.pipeline import checkpoint, worker, job_queue


@pytest.fixture
def job_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "require_unc", lambda path: None)
    series = tmp_path / "series"
    root = series / "jobs" / "fixture-job"
    root.mkdir(parents=True)
    (root / "job.json").write_text(json.dumps(dict(expected_runs=["run-a"], precision="float32")), encoding="utf-8")
    (root / "result.bin").write_bytes(b"private derived result")
    raw = series / "raw"
    raw.mkdir()
    (raw / "source.ARW").write_bytes(b"source must remain unchanged")
    index = registry.JobsUIRegistry(tmp_path / "local-registry")
    index.register_job(series, "SERIES", root)
    return index, series, root, raw


def test_archive_restore_keeps_files_and_cross_job_start_reservation_visibility(job_tree):
    index, series, root, raw = job_tree
    original = (root / "job.json").read_bytes()
    manage.archive_job(index, series, "SERIES", root, actor="fixture-user")
    assert index.jobs(series, "SERIES") == []
    assert index.jobs(series, "SERIES", include_archived=True)[0]["archived"] is True
    assert index.known_jobs()[0]["root"] == str(root)
    assert (root / "job.json").read_bytes() == original
    manage.archive_job(index, series, "SERIES", root, actor="fixture-user", archived=False)
    assert len(index.jobs(series, "SERIES")) == 1
    assert index.context(series, "SERIES")["history"][-1]["actor"] == "fixture-user"


def test_delete_removes_only_the_reviewed_job_folder_and_keeps_acquisition_data(job_tree):
    index, series, root, raw = job_tree
    plan = manage.deletion_plan(index, series, "SERIES", root)
    result = manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    assert result["deleted"] and not root.exists()
    assert (raw / "source.ARW").read_bytes() == b"source must remain unchanged"
    assert index.known_jobs() == []
    assert index.context(series, "SERIES")["history"][-1]["action"] == "delete"
    assert not list((series / "jobs").glob(".deleting-*"))


def test_delete_uses_declared_results_jobs_and_keeps_other_result_products(tmp_path, monkeypatch):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "series-registry.toml"))
    monkeypatch.setattr(registry, "require_unc", lambda path: None)
    series, results = tmp_path / "DarSIA", tmp_path / "Results"
    series.mkdir()
    (series / "series.toml").write_text('[paths]\njobs="../Results/jobs"\n', encoding="utf-8")
    root = results / "jobs" / "selected-job"
    root.mkdir(parents=True)
    (root / "job.json").write_text('{"expected_runs":["sample"],"precision":"float32"}', encoding="utf-8")
    keep = results / "setup" / "keep.json"
    keep.parent.mkdir()
    keep.write_bytes(b"saved setup")
    index = registry.JobsUIRegistry(tmp_path / "jobs-registry")
    index.register_job(series, "sample", root)
    plan = manage.deletion_plan(index, series, "sample", root)
    assert manage.delete_job(index, series, "sample", plan, actor="test")["deleted"]
    assert keep.read_bytes() == b"saved setup" and not root.exists()


def test_changed_files_after_review_prevent_any_payload_deletion(job_tree):
    index, series, root, raw = job_tree
    plan = manage.deletion_plan(index, series, "SERIES", root)
    (root / "result.bin").write_bytes(b"changed by another writer")
    with pytest.raises(ValueError, match="changed after review"):
        manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    assert (root / "result.bin").exists() and len(index.known_jobs()) == 1
    assert not (root / ".job-management.lock").exists()


def test_running_or_unknown_start_blocks_archive_and_delete(job_tree, monkeypatch):
    index, series, root, raw = job_tree
    begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    with pytest.raises(ValueError, match="unresolved start"):
        manage.archive_job(index, series, "SERIES", root, actor="fixture-user")
    with pytest.raises(ValueError, match="unresolved start"):
        manage.deletion_plan(index, series, "SERIES", root)
    assert (root / "result.bin").exists()


def test_stopped_unfinished_job_offers_archive_and_preserves_every_file(job_tree):
    index, series, root, raw = job_tree
    plan = manage.deletion_plan(index, series, "SERIES", root)
    claim = root / "claimed" / "worker" / "run-a.json"
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps(dict(run="run-a", attempt_id="a" * 32)), encoding="utf-8")
    logs = root / "logs"
    logs.mkdir()
    (logs / "worker.log").write_text(
        'timestamp {"kind":"started"}\ntimestamp {"kind":"stopped"}\n', encoding="utf-8")
    original = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(manage.UnfinishedClaimError, match="Stopping a job preserves unfinished work") as error:
        manage.deletion_plan(index, series, "SERIES", root)
    assert "Archive this job to hide it while keeping all files" in str(error.value)
    with pytest.raises(manage.UnfinishedClaimError):
        manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    manage.archive_job(index, series, "SERIES", root, actor="fixture-user")
    assert index.jobs(series, "SERIES") == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == original
    with pytest.raises(manage.UnfinishedClaimError):
        manage.deletion_plan(index, series, "SERIES", root)


def test_action_lock_excludes_new_gui_start_reservations(job_tree):
    index, series, root, raw = job_tree
    with manage.JobActionLock(root):
        with pytest.raises(ValueError, match="Another job action"):
            with manage.JobActionLock(root):
                pytest.fail("A concurrent Start must not reserve the job")
    assert not (root / ".job-management.lock").exists()


def test_delete_refuses_a_registered_folder_outside_this_series_jobs(job_tree):
    index, series, root, raw = job_tree
    outside = series.parent / "outside"
    outside.mkdir()
    (outside / "job.json").write_bytes((root / "job.json").read_bytes())
    index.register_job(series, "SERIES", outside)
    with pytest.raises(ValueError, match="limited to a job folder"):
        manage.deletion_plan(index, series, "SERIES", outside)
    assert (outside / "job.json").exists()


def test_delete_keeps_metadata_used_by_another_registered_job(job_tree):
    index, series, root, raw = job_tree
    child = series / "jobs" / "dependent-job"
    child.mkdir()
    (child / "job.json").write_text(json.dumps(dict(expected_runs=["run-a"], precision="float32",
        source_job=dict(root_alias="series", relative_path="jobs/fixture-job/job.json"))), encoding="utf-8")
    index.register_job(series, "SERIES", child)
    with pytest.raises(ValueError, match="Another registered job uses"):
        manage.deletion_plan(index, series, "SERIES", root)
    assert (root / "result.bin").exists()


def control_master(root, machine="fixture"):
    job_control.write_immutable(root / "ui" / "master.json", dict(
        format="darsia-job-ui-master-v1", machine=machine,
        job=job_control.descriptor(root / "job.json")))


def dependent_job(index, series, source, name, *, machine="fixture"):
    root = series / "jobs" / name
    root.mkdir()
    (root / "job.json").write_text(json.dumps(dict(expected_runs=["run-a"], precision="float32",
        source_job=dict(root_alias="series", relative_path="jobs/" + source.name + "/job.json"))), encoding="utf-8")
    index.register_job(series, "SERIES", root)
    control_master(root, machine)
    return root


def test_stop_delete_review_lists_transitive_consumers_and_only_stops_reviewed_jobs(job_tree):
    index, series, root, raw = job_tree
    control_master(root)
    consumer = dependent_job(index, series, root, "consumer")
    indirect = dependent_job(index, series, consumer, "indirect")
    unrelated = dependent_job(index, series, raw, "unrelated")
    review = manage.stop_delete_review(index, series, "SERIES", root)
    assert [row["root"] for row in review["affected"]] == [str(root), str(consumer), str(indirect)]
    assert all(job_control.read_control(path)["sequence"] == 0 for path in (root, consumer, indirect, unrelated))
    review = manage.stop_for_deletion(index, review, machine="fixture", actor="fixture-user")
    assert all(job_control.read_control(path)["action"] == "stop" for path in (root, consumer, indirect))
    assert job_control.read_control(unrelated)["sequence"] == 0
    status = manage.stop_delete_status(index, review)
    assert status["ready"]
    with pytest.raises(ValueError, match="Another registered job uses"):
        manage.deletion_plan(index, series, "SERIES", root)
    plan = manage.deletion_plan(index, series, "SERIES", root, retirement_review=review)
    manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    assert consumer.exists() and indirect.exists() and unrelated.exists()
    assert (raw / "source.ARW").read_bytes() == b"source must remain unchanged"
    assert not root.exists()


def test_stop_delete_preflights_recorded_master_and_changed_consumer_list(job_tree):
    index, series, root, raw = job_tree
    control_master(root)
    consumer = dependent_job(index, series, root, "consumer", machine="other-master")
    review = manage.stop_delete_review(index, series, "SERIES", root)
    with pytest.raises(ValueError, match="recorded master machine"):
        manage.stop_for_deletion(index, review, machine="fixture", actor="fixture-user")
    assert job_control.read_control(root)["sequence"] == 0
    assert job_control.read_control(consumer)["sequence"] == 0
    dependent_job(index, series, root, "new-consumer")
    with pytest.raises(ValueError, match="Affected jobs changed"):
        manage.stop_for_deletion(index, review, machine="fixture", actor="fixture-user")
    assert job_control.read_control(root)["sequence"] == 0


def test_stop_delete_waits_for_original_report_not_operator_declaration(job_tree):
    index, series, root, raw = job_tree
    control_master(root)
    begin_launch_request(root, machine="fixture", backend="cpu", workers=1, kind="local")
    declare_requests_stopped(root, read_launch_requests(root))
    review = manage.stop_delete_review(index, series, "SERIES", root)
    review = manage.stop_for_deletion(index, review, machine="fixture", actor="fixture-user")
    status = manage.stop_delete_status(index, review)
    assert not status["ready"] and status["can_wait"]
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "fixture-cpu.log").write_text(job_control.utc() + ' {"kind":"stopped"}\n', encoding="utf-8")
    assert manage.stop_delete_status(index, review)["ready"]
    claim = root / "claimed" / "fixture" / "run-a.json"
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps(dict(run="run-a", attempt_id="a" * 32)), encoding="utf-8")
    status = manage.stop_delete_status(index, review)
    assert not status["ready"] and status["can_wait"]
    assert "waiting for original claim retirement" in status["waiting"][0]
    assert not (root / "decisions").exists() and claim.exists()


def retired_claim(root, monkeypatch, service="fixture-cpu", *, alias_results=None):
    """Original supervisor receipt fixture; never start or inspect real processes."""
    machine, backend = service.rsplit("-", 1)
    begin_launch_request(root, machine=machine, backend=backend, workers=1, kind="local")
    run = job_control.read_json(root / "job.json")["expected_runs"][0]
    task = dict(run=run, attempt_id="a" * 32)
    if alias_results is not None:
        task.update(format="darsia-alias-task-v1", job_id="fixture-job")
    claimed = root / "claimed" / service / (run + ".json")
    actual = dict(machine=service, pid=10001, process_create_time="10.1")
    wrapper = dict(pid=10002, process_create_time="10.2", creation_filetime_100ns="1000200")
    supervisor = dict(machine=service, pid=10003, process_create_time="10.3")
    output = (root / "results" / run / ".attempts" / task["attempt_id"] if alias_results is None
              else alias_results / task["job_id"] / run / ".attempts" / task["attempt_id"])
    output.mkdir(parents=True)
    job_control.write_immutable(claimed, task)
    owner = dict(task=task, machine=service, token=task["attempt_id"], actual_owner=actual)
    if alias_results is None:
        owner.update(claimed_path=str(claimed), output_path=str(output))
    else:
        owner.update(format="darsia-alias-lease-owner-v1", output=dict(
            machine_id=machine, root_alias="results", relative_path=output.relative_to(alias_results).as_posix()))
    job_control.write_immutable(root / "leases" / (run + "." + task["attempt_id"]) / "owner.json", owner)
    proof = checkpoint.retirement_document(task, output, [actual, wrapper], supervisor)
    proof.update(actual_launch_wrapper=wrapper, owned_launch_handle_retired=True,
                 retired_utc=job_control.utc())
    if alias_results is not None:
        proof.update(machine_id=machine, output_reference=owner["output"])
    job_control.write_immutable(output / "retirement.json", proof)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / (service + ".log")).write_text(job_control.utc() + ' {"kind":"stopped"}\n', encoding="utf-8")
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    return claimed


def test_confirmed_delete_cancels_original_retired_claim_without_retry(job_tree, monkeypatch):
    index, series, root, raw = job_tree
    control_master(root)
    claimed = retired_claim(root, monkeypatch)
    review = manage.stop_for_deletion(index, manage.stop_delete_review(index, series, "SERIES", root),
                                     machine="fixture", actor="fixture-user")
    status = manage.stop_delete_status(index, review)
    assert status["ready"] and len(status["claims"]) == 1
    with pytest.raises(manage.UnfinishedClaimError):
        manage.deletion_plan(index, series, "SERIES", root)
    plan = manage.deletion_plan(index, series, "SERIES", root, retirement_review=review)
    assert claimed.exists() and not (root / "decisions").exists()
    decisions = []
    original_cancel = manage.cancel_retired_claim
    def cancel(*args, **kwargs):
        path = original_cancel(*args, **kwargs)
        decisions.append(job_control.read_json(path))
        assert not (root / "queue").exists()
        return path
    monkeypatch.setattr(manage, "cancel_retired_claim", cancel)
    manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    assert not root.exists() and len(decisions) == 1
    assert decisions[0]["outcome"] == "rejected" and decisions[0]["cancellation"] is True
    assert (raw / "source.ARW").read_bytes() == b"source must remain unchanged"


def test_claim_cancellation_refuses_live_original_epoch_and_changed_stop(job_tree, monkeypatch):
    index, series, root, raw = job_tree
    control_master(root)
    claimed = retired_claim(root, monkeypatch)
    review = manage.stop_for_deletion(index, manage.stop_delete_review(index, series, "SERIES", root),
                                     machine="fixture", actor="fixture-user")
    evidence = manage.stop_delete_status(index, review)["claims"][0]["evidence"]
    plan = manage.deletion_plan(index, series, "SERIES", root, retirement_review=review)
    monkeypatch.setattr(worker, "_same_process", lambda identity: object())
    with pytest.raises(ValueError, match="still running"):
        job_queue.cancel_retired_claim(root, claimed, evidence=evidence, actor="fixture-user")
    with pytest.raises(ValueError, match="still running"):
        manage.delete_job(index, series, "SERIES", plan, actor="fixture-user")
    assert claimed.exists() and not (root / "decisions").exists()
    monkeypatch.setattr(worker, "_same_process", lambda identity: None)
    job_control.publish_control(root, "run", machine="fixture", expected_previous=evidence["stop"])
    with pytest.raises(ValueError, match="exact recorded Stop"):
        job_queue.cancel_retired_claim(root, claimed, evidence=evidence, actor="fixture-user")
    assert not (root / "decisions").exists()


def test_terminal_cancellation_retains_files_without_requeue_or_science_publication(job_tree, monkeypatch):
    index, series, root, raw = job_tree
    control_master(root)
    claimed = retired_claim(root, monkeypatch)
    review = manage.stop_for_deletion(index, manage.stop_delete_review(index, series, "SERIES", root),
                                     machine="fixture", actor="fixture-user")
    evidence = manage.stop_delete_status(index, review)["claims"][0]["evidence"]
    original = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    decision = job_queue.cancel_retired_claim(root, claimed, evidence=evidence, actor="fixture-user")
    assert job_control.read_json(decision)["outcome"] == "rejected"
    assert all((root / path).read_bytes() == value for path, value in original.items())
    queue = job_queue.JobQueue(root, machine_id="fixture")
    assert queue.reclaim_stale() == []
    assert not list((root / "queue").glob("*.json"))
    assert not list((root / "done").glob("*.json")) and not list((root / "failed").glob("*.json"))


def test_alias_cancellation_requires_original_machine_output_binding(tmp_path, monkeypatch):
    from darsia.fluidflower import job_filesystem
    from darsia.fluidflower.pipeline.alias_job import job_metadata_reference
    root = tmp_path / "alias-job"
    job_control.write_immutable(root / "job.json", dict(format="darsia-alias-job-v1", format_version=1,
        expected_runs=["run-a"], series_id="fixture-series", precision="float32"))
    job_control.write_immutable(root / "ui" / "master.json", dict(format="darsia-job-ui-master-v2",
        machine="fixture", job=job_metadata_reference(root, root / "job.json")))
    results = tmp_path / "local-results"
    claimed = retired_claim(root, monkeypatch, alias_results=results)
    command = job_control.publish_control(root, "stop", machine="fixture", expected_previous=None)
    output = results / "fixture-job" / "run-a" / ".attempts" / ("a" * 32)
    admission = job_filesystem.LocalFileAdmission(True, "fixture-local", "fixture only", "No actual filesystem qualification claimed", str(tmp_path))
    binding = job_queue.LocalOutputBinding("fixture", results, output, admission)
    evidence = job_queue.retired_claim_evidence(root, claimed, local_output=binding, expected_stop=command)
    foreign = job_queue.LocalOutputBinding("other-machine", results, output, admission)
    with pytest.raises(ValueError, match="another machine"):
        job_queue.cancel_retired_claim(root, claimed, evidence=evidence, actor="fixture-user", local_output=foreign)
    decision = job_queue.cancel_retired_claim(root, claimed, evidence=evidence, actor="fixture-user", local_output=binding)
    assert job_control.read_json(decision)["cancellation"] is True
    assert (output / "retirement.json").exists() and claimed.exists()
