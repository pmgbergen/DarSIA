"""Several machines share one current-Build P1 job by claiming its runs.

Spawned test runs use the small numerical fixture of the parallel tests; no
RAW decoder, campaign or scientific job is started.
"""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset, _write_json
from darsia.fluidflower.pipeline import current_build_job as subject
from darsia.fluidflower.pipeline import current_inputs
from darsia.fluidflower.shared_job_service import SharedJobService, backend_blocker

from .test_current_build_parallel import _fixture_run, _job


def _shared(tmp_path, count=3, machine="Second"):
    job = _job(tmp_path, count)
    job.update({subject.SHARED_RUNS: True, "job_id": "p1-shared", "executing_machine": machine,
                "pipeline": "native-phase-statistics"})
    return job


def test_a_claim_is_exclusive_and_names_its_machine(tmp_path):
    job = _shared(tmp_path, 2)
    assert subject.claimable_runs(tmp_path, job) == ["run-0", "run-1"]
    assert subject._claim_run(job, "run-0") is True
    claim = json.loads((tmp_path / "claims" / "run-0.json").read_bytes())
    assert claim["format"] == subject.CLAIM_FORMAT and claim["owner"]["machine"] == "Second-cpu"
    before = (tmp_path / "claims" / "run-0.json").read_bytes()
    assert subject._claim_run(dict(job, executing_machine="Main"), "run-0") is False
    assert (tmp_path / "claims" / "run-0.json").read_bytes() == before
    assert subject.claimable_runs(tmp_path, job) == ["run-1"]
    subject._write_json(tmp_path / "failed" / "run-0.json", dict(run="run-0", traceback="TESTONLY"))
    assert subject.claimable_runs(tmp_path, job) == ["run-1"]  # A failure leaves the other runs.
    assert subject._claim_run(job, "run-1") is True and subject.claimable_runs(tmp_path, job) == []


def test_a_second_machine_runs_only_the_runs_it_claims(tmp_path):
    job = _shared(tmp_path)
    assert subject._claim_run(dict(job, executing_machine="Main"), "run-1")
    log = io.StringIO()
    result = subject._dispatch_runs(job, "a" * 40, None, 2, log, runner=_fixture_run,
        recommendation=lambda rows, active: dict(workers=max(0, 2 - len(active)), reason="TESTONLY capacity"),
        capacity_factory=lambda row: nullcontext(), claim=lambda row: subject._claim_run(job, row["run"]))
    assert result == 0
    for run in ("run-0", "run-2"):
        published = DataReader().json(asset(tmp_path / "results" / run / "completed.json"))
        assert published["executed_by"] == "Second" and published["run"] == run
        claim = json.loads((tmp_path / "claims" / (run + ".json")).read_bytes())
        assert claim["owner"]["machine"] == "Second-cpu"
    assert not (tmp_path / "results" / "run-1").exists()
    assert json.loads((tmp_path / "claims" / "run-1.json").read_bytes())["owner"]["machine"] == "Main-cpu"
    events = [json.loads(line[line.index("{"):]) for line in log.getvalue().splitlines() if "{" in line]
    workers = {event["actual_worker"]["machine"] for event in events if event.get("kind") == "progress"}
    assert workers == {"Second-cpu"}


def test_default_jobs_publish_unchanged_results(tmp_path):
    job = _job(tmp_path, 1)
    assert subject._dispatch_runs(job, "a" * 40, None, 1, io.StringIO(), runner=_fixture_run,
        recommendation=lambda rows, active: dict(workers=1, reason="TESTONLY capacity"),
        capacity_factory=lambda row: nullcontext()) == 0
    published = DataReader().json(asset(tmp_path / "results" / "run-0" / "completed.json"))
    assert "executed_by" not in published and not (tmp_path / "claims").exists()


def test_the_runner_admits_other_machines_only_for_a_shared_job(tmp_path, monkeypatch):
    from darsia.fluidflower import store_map
    calls = []
    monkeypatch.setattr(subject, "_dispatch_runs", lambda *args, **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr(store_map, "read_store_map", lambda path: SimpleNamespace(stores=path))
    monkeypatch.setattr(subject, "require_shared_machine", lambda job, stores: calls.append(job["executing_machine"]))
    args = SimpleNamespace(job=tmp_path, backend="cpu", machine="Second", workers=2, stores="stores.toml")
    default = _job(tmp_path, 2)
    with pytest.raises(ValueError, match="selected Build machine"):
        subject.run_current_build_job(args, default, "a" * 40, io.StringIO())
    subject._write_json(tmp_path / "results" / "run-0" / "completed.json", dict(complete=True, run="run-0"))
    shared = _shared(tmp_path, 2)
    shared.pop("executing_machine")
    assert subject.run_current_build_job(args, shared, "a" * 40, io.StringIO()) == 0
    assert calls[0] == "Second" and callable(calls[1]["claim"])
    main = SimpleNamespace(**dict(vars(args), machine="Main"))
    with pytest.raises(FileExistsError, match="already has completed work"):
        subject.run_current_build_job(main, default, "a" * 40, io.StringIO())
    with pytest.raises(ValueError, match="requires CPU workers|CPU workers"):
        subject.run_current_build_job(SimpleNamespace(**dict(vars(args), backend="cuda")), shared,
                                      "a" * 40, io.StringIO())


def test_a_joining_machine_must_share_results_and_space_ledger(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    job = dict(series_id="test", results_root=str(results / "p1-shared"))
    def stores(results_root, cache=True):
        folders = {"results": str(results_root), **({"cache": str(tmp_path / "cache")} if cache else {})}
        return SimpleNamespace(roots=lambda series: folders, document={"stores": {"test": folders}})
    assert subject.require_shared_machine(job, stores(results)) is None
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="Choose the same Results folder"):
        subject.require_shared_machine(job, stores(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="output-space reservations"):
        subject.require_shared_machine(job, stores(results, cache=False))


@pytest.mark.skipif(os.name != "nt", reason="A directory junction is a Windows spelling of one folder")
def test_another_spelling_of_the_same_results_folder_is_the_same_folder(tmp_path):
    import _winapi
    results = tmp_path / "results"
    results.mkdir()
    spelling = tmp_path / "mapped-results"
    _winapi.CreateJunction(str(results), str(spelling))
    job = dict(series_id="test", results_root=str(results / "p1-shared"))
    folders = {"results": str(spelling), "cache": str(tmp_path / "cache")}
    stores = SimpleNamespace(roots=lambda series: folders, document={"stores": {"test": folders}})
    assert subject.same_folder(spelling, results) and not subject.same_folder(tmp_path / "missing", results)
    assert subject.require_shared_machine(job, stores) is None


def test_shared_reader_needs_positive_evidence_of_a_network_share(tmp_path, monkeypatch):
    from darsia.fluidflower import job_filesystem
    bound = _write_json(tmp_path / "input.json", dict(value=1))
    assert current_inputs.current_input_reader().json(bound) == dict(value=1)
    shared = current_inputs.current_input_reader(shared_only=True)
    reports = dict(local=None, synchronized=SimpleNamespace(allowed=False),
                   smb=SimpleNamespace(allowed=True))
    for name, report in reports.items():
        monkeypatch.setattr(current_inputs, "_SHARED_FOLDERS", {})
        if report is not None:
            monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda folder, report=report: report)
        if name == "smb":
            assert shared.json(bound) == dict(value=1)
        else:  # A local disk, a sync-client folder and an inspection failure all refuse.
            with pytest.raises(ValueError, match="not on a shared network folder"):
                shared.json(bound)
    def failed(folder):
        raise OSError("unreachable")
    monkeypatch.setattr(current_inputs, "_SHARED_FOLDERS", {})
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", failed)
    with pytest.raises(ValueError, match="not on a shared network folder"):
        shared.json(bound)


def test_the_service_starts_other_machines_only_while_runs_remain(tmp_path):
    shared = _shared(tmp_path, 2)
    shared.update(pipeline="native-phase-statistics", execution_mode=subject.MODE)
    assert backend_blocker(shared, "cpu", machine="Second", master_machine="Main") == ""
    assert backend_blocker(shared, "cuda", machine="Second", master_machine="Main")
    default = {key: value for key, value in shared.items() if key != subject.SHARED_RUNS}
    assert backend_blocker(default, "cpu", machine="Second", master_machine="Main")
    count = lambda job: SharedJobService._queued_worker_count(None, tmp_path, job, "cpu", 3)
    assert count(shared) == 3 and count(default) == 3
    from darsia.fluidflower.shared_job_service import _shared_runs_remain
    subject._write_json(tmp_path / "failed" / "run-0.json", dict(run="run-0", traceback="TESTONLY"))
    assert count(shared) == 3 and _shared_runs_remain(tmp_path, shared)  # Other runs still start.
    assert not _shared_runs_remain(tmp_path, default)
    assert subject._claim_run(shared, "run-1")
    assert count(shared) == 0 and not _shared_runs_remain(tmp_path, shared)  # No relaunch without work.
    assert count(default) == 3
