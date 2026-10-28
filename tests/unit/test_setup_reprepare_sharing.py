"""Sharing violations retry one Setup move, preserving the original transaction."""
import os
from pathlib import Path

import pytest

from darsia.fluidflower import job_control, setup_job_reprepare as recovery
from darsia.fluidflower.pipeline import job_queue, series_setup_jobs
from darsia.fluidflower.setup_completed_builds import retained_operations
from darsia_cache import transient
from .test_series_setup_jobs import setup_job, controls  # noqa: F401
from .test_setup_job_reprepare import unstarted, _retry, _bytes  # noqa: F401
from .test_setup_image_preparation_update import completed_geometry, save_independent_choice


def sharing_error():
    error = PermissionError(13, "Windows reader has not released its sharing handle")
    error.winerror = 32
    return error


def image_upgrade(case, monkeypatch):
    case = completed_geometry(case, monkeypatch)
    save_independent_choice(case)
    previous = job_control.read_control(case.job_root)
    job_control.publish_control(case.job_root, "pause", machine="private-host",
                                expected_previous=previous.get("command"))
    return case


@pytest.mark.parametrize("persistent", [False, True])
def test_job_json_sharing_retry_preserves_genuine_completions(setup_job, monkeypatch, persistent):
    case = image_upgrade(setup_job, monkeypatch)
    before, completed = recovery._snapshot(case.job_root), _bytes(case.job_root / "done")
    controls_before = _bytes(case.job_root / "setup-controls")
    outputs = {build["run_id"]: _bytes(Path(build["request_template"]["output"]["destination"]))
               for build in case.plan["build_plans"]}
    publication = case.folder.jobs()[0]
    registry = case.registry.context(case.series, "synthetic")
    original_rename = Path.rename
    attempts, waits = [], []
    def rename(source, destination):
        destination = Path(destination)
        if source == case.job_root / "job.json" and destination.parent.name == "previous":
            attempts.append((source, destination))
            if persistent or len(attempts) <= 2:
                raise sharing_error()
        return original_rename(source, destination)
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(transient.time, "sleep", waits.append)
    if persistent:
        with pytest.raises(PermissionError) as failure:
            _retry(case)
        assert failure.value.winerror == 32
        assert len(attempts) == 4 and waits == list(transient.TRANSIENT_RETRY_DELAYS)
        assert recovery._snapshot(case.job_root) == before
        assert _bytes(case.job_root / "setup-controls") == controls_before
        assert case.folder.jobs()[0] == publication
        assert case.registry.context(case.series, "synthetic") == registry
        failures = list((case.job_root / "history").glob("*/failure.json"))
        assert len(failures) == 1 and job_queue.read_json(failures[0])["restored_original"] is True
    else:
        result = _retry(case)
        assert len(attempts) == 3 and waits == [0.5, 2.0]
        assert len(set(attempts)) == 1  # Every try uses the same source/destination.
        assert _bytes(Path(result["history"]) / "previous" / "done") == completed
        assert set(retained_operations(case.job_root)) == {"setup-apply", *case.plan["runs"]}
        assert {path.stem for path in (case.job_root / "queue").glob("*.json")} == {"setup-image-preparation"}
    for build in case.plan["build_plans"]:
        assert _bytes(Path(build["request_template"]["output"]["destination"])) == outputs[build["run_id"]]


def test_sharing_retry_also_restores_the_original_during_rollback(unstarted, monkeypatch):
    case, attempts, waits = unstarted, [], []
    before = _bytes(case.job_root)
    original_rename = Path.rename
    def rename(source, destination):
        destination = Path(destination)
        if source.parent.name == "previous" and destination == case.job_root / "job.json":
            attempts.append((source, destination))
            if len(attempts) <= 2:
                raise sharing_error()
        return original_rename(source, destination)
    def failed_producer(bound, root, **options):
        series_setup_jobs.prepare_setup_job(bound, root, **options)
        raise ValueError("The original producer failed after partial replacement")
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(transient.time, "sleep", waits.append)
    with pytest.raises(ValueError, match="original producer failed"):
        _retry(case, producer=failed_producer)
    assert len(attempts) == 3 and waits == [0.5, 2.0]
    assert _bytes(case.job_root) == before
    failure = next((case.job_root / "history").glob("*/failure.json"))
    assert job_queue.read_json(failure)["restored_original"] is True


def test_retry_refuses_a_changed_source_even_with_unchanged_file_stat(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.json", tmp_path / "saved.json"
    source.write_bytes(b"original")
    original_stat = source.stat()
    attempts = []
    def rename(path, target):
        attempts.append((path, target))
        path.write_bytes(b"modified")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        raise sharing_error()
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(transient.time, "sleep", lambda _seconds: None)
    with pytest.raises(ValueError, match="transaction source changed"):
        recovery._rename(source, destination)
    assert len(attempts) == 1 and not destination.exists()


def test_interrupted_move_recognizes_only_its_exact_unchanged_destination(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.json", tmp_path / "saved.json"
    source.write_bytes(b"original")
    original_rename = Path.rename
    def moved_but_disconnected(path, target):
        original_rename(path, target)
        raise sharing_error()
    monkeypatch.setattr(Path, "rename", moved_but_disconnected)
    recovery._rename(source, destination)
    assert not source.exists() and destination.read_bytes() == b"original"


def test_transaction_move_does_not_replace_an_existing_destination(tmp_path):
    source, destination = tmp_path / "source.json", tmp_path / "saved.json"
    source.write_bytes(b"original")
    destination.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        recovery._rename(source, destination)
    assert source.read_bytes() == b"original" and destination.read_bytes() == b"foreign"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_real_windows_reader_can_close_between_transaction_move_attempts(tmp_path, monkeypatch):
    import threading
    source, destination = tmp_path / "source.json", tmp_path / "saved.json"
    source.write_bytes(b"original")
    original_rename, sharing = Path.rename, []
    def rename(path, target):
        try:
            return original_rename(path, target)
        except OSError as error:
            sharing.append(getattr(error, "winerror", None))
            raise
    monkeypatch.setattr(Path, "rename", rename)
    reader = source.open("rb")
    close = threading.Timer(0.1, reader.close)
    close.start()
    try:
        recovery._rename(source, destination)
    finally:
        close.join(timeout=2)
        reader.close()
    assert 32 in sharing
    assert not source.exists() and destination.read_bytes() == b"original"
