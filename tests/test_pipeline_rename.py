"""Tiny metadata fixtures for exclusive Windows queue renames."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time
import traceback
import uuid


def _claim_trace(stage, **details):
    """Keep bounded, best-effort breadcrumbs only for the selected fixture."""
    root = os.environ.get("DARSIA_TEST_CLAIM_DIAGNOSTICS")
    if root is None:
        return
    try:
        row = dict(stage=stage, pid=os.getpid(), epoch=time.time(),
                   monotonic=time.monotonic(), details=details)
        text = json.dumps(row, ensure_ascii=True, default=str)
        if len(text) > 1023:
            row["details"] = dict(truncated=json.dumps(details, ensure_ascii=True, default=str)[:300])
            text = json.dumps(row, ensure_ascii=True)
        with (Path(root) / f"{os.getpid()}.jsonl").open("a", encoding="ascii") as stream:
            stream.write(text + "\n")
    except Exception:
        pass


def _claim_state(process, launches):
    """Capture only this test's child state before its Process handle closes."""
    try:
        return dict(pid=process.pid, exitcode=process.exitcode,
                    launch=launches.get(process.pid))
    except Exception as error:
        return dict(snapshot_error=repr(error))


def _claim_dump(stage, states):
    """Print bounded owned logs without replacing the original test failure."""
    try:
        print("CLAIM_DIAGNOSTICS", stage, json.dumps(states, default=str))
        root = Path(os.environ["DARSIA_TEST_CLAIM_DIAGNOSTICS"])
        for pid in sorted({os.getpid(), *(row["pid"] for row in states if row.get("pid") is not None)}):
            path = root / f"{pid}.jsonl"
            try:
                with path.open("rb") as stream:
                    print(str(path), stream.read(8192).decode("ascii", errors="replace"))
            except OSError as error:
                print(str(path), repr(error))
    except Exception:
        pass


_claim_trace("test_module_entered")
try:
    _claim_trace("before_pytest_import")
    import pytest
    _claim_trace("after_pytest_import")
    _claim_trace("before_darsia_import")
    from darsia.fluidflower.pipeline import job_queue as queue
    _claim_trace("after_darsia_import")
except BaseException as error:
    _claim_trace("module_import_exception", error=repr(error)[:200],
                 traceback=traceback.format_exc(limit=5)[-500:])
    raise


def _claim_process(root, index, ready, start, answers):
    """Claim a tiny task in an independent spawned interpreter."""
    _claim_trace("target_entered", index=index)
    try:
        import psutil

        process = psutil.Process()
        owner = dict(pid=process.pid, process_create_time=str(process.create_time()))
        _claim_trace("target_owner", index=index, owner=owner)
        _claim_trace("ready_put_before", index=index)
        ready.put(index)
        _claim_trace("ready_put_after", index=index)
        _claim_trace("start_wait_before", index=index)
        released = start.wait(20)
        _claim_trace("start_wait_after", index=index, released=released)
        if not released:
            raise RuntimeError("Parent did not release metadata fixture")
        try:
            _claim_trace("claim_before", index=index)
            claim = queue.JobQueue(Path(root)).claim(f"child-{index}", "cpu", owner)
            _claim_trace("claim_after", index=index, claimed=None if claim is None else claim.token)
            _claim_trace("answer_put_before", index=index)
            answers.put(dict(index=index, claimed=None if claim is None else claim.token))
            _claim_trace("answer_put_after", index=index)
        except Exception as exc:
            answers.put(dict(index=index, error=repr(exc)))
            raise
    except BaseException as error:
        _claim_trace("target_exception", index=index, error=repr(error)[:200],
                     traceback=traceback.format_exc(limit=5)[-500:])
        raise


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive-handle semantics")
def test_thread_renames_have_one_winner(tmp_path):
    source = tmp_path / "pending.json"
    source.write_bytes(b"one immutable metadata fixture")
    barrier = threading.Barrier(16)

    def attempt(index):
        destination = tmp_path / f"claimed-{index}.json"
        barrier.wait()
        try:
            queue.rename(source, destination)
        except FileNotFoundError:
            return None
        return destination

    with ThreadPoolExecutor(16) as pool:
        winners = [path for path in pool.map(attempt, range(16)) if path is not None]
    assert len(winners) == 1
    assert winners[0].read_bytes() == b"one immutable metadata fixture"
    assert not source.exists()
    assert len(list(tmp_path.glob("claimed-*.json"))) == 1


def test_spawned_claimers_have_one_owner(tmp_path, monkeypatch):
    diagnostics = tmp_path / "claim-diagnostics"
    try:
        diagnostics.mkdir()
    except OSError:
        pass
    monkeypatch.setenv("DARSIA_TEST_CLAIM_DIAGNOSTICS", str(diagnostics))
    import psutil

    _claim_trace("parent_entered")
    shared = queue.JobQueue(tmp_path)
    shared.enqueue(dict(run="sample", commit="1" * 40,
        calibration_sha256="2" * 64,
        configuration=dict(path=r"\\server\share\fixture.json", sha256="3" * 64),
        precision="float32", backend="cpu"))
    context = multiprocessing.get_context("spawn")
    ready, answers = context.Queue(), context.Queue()
    start = context.Event()
    processes = [context.Process(target=_claim_process,
        args=(str(tmp_path), index, ready, start, answers)) for index in range(4)]
    launches, retired, failed = {}, [], False
    try:
        for index, process in enumerate(processes):
            _claim_trace("launch_before", index=index)
            process.start()
            launched = dict(epoch=time.time(), monotonic=time.monotonic())
            try:
                launched["process_create_time"] = str(psutil.Process(process.pid).create_time())
            except Exception as error:
                launched["creation_error"] = repr(error)
            launches[process.pid] = launched
            _claim_trace("launch_after", index=index, child_pid=process.pid, launch=launched)
        assert {ready.get(timeout=30) for _ in processes} == set(range(4))
        _claim_trace("all_ready")
        start.set()
        results = [answers.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert not any("error" in result for result in results)
        assert sum(result["claimed"] is not None for result in results) == 1
        assert len(list((tmp_path / "leases").glob("*/owner.json"))) == 1
        assert len(list((tmp_path / "claimed").glob("*/*.json"))) == 1
        assert not list((tmp_path / "queue").glob("*.json"))
    except BaseException as error:
        failed = True
        _claim_trace("parent_exception", error=repr(error)[:200],
                     traceback=traceback.format_exc(limit=5)[-500:])
        _claim_dump("before_cleanup", [_claim_state(process, launches) for process in processes])
        raise
    finally:
        start.set()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=30)
                state = _claim_state(process, launches)
                retired.append(state)
                _claim_trace("cleanup_child", **state)
                if not process.is_alive():
                    process.close()
        for channel in (ready, answers):
            channel.close()
            channel.join_thread()
        if failed:
            _claim_dump("after_cleanup", retired)


@pytest.mark.skipif(os.name != "nt", reason="Windows no-replace rename")
@pytest.mark.parametrize("directory", [False, True])
def test_existing_destination_is_preserved(tmp_path, directory):
    source, destination = tmp_path / "source", tmp_path / "destination"
    if directory:
        source.mkdir(); destination.mkdir()
        (source / "source.txt").write_bytes(b"source")
        (destination / "destination.txt").write_bytes(b"destination")
    else:
        source.write_bytes(b"source"); destination.write_bytes(b"destination")
    with pytest.raises(FileExistsError):
        queue.rename(source, destination)
    if directory:
        assert (source / "source.txt").read_bytes() == b"source"
        assert (destination / "destination.txt").read_bytes() == b"destination"
    else:
        assert source.read_bytes() == b"source"
        assert destination.read_bytes() == b"destination"


@pytest.mark.parametrize("directory", [False, True])
def test_file_and_directory_move_preserve_contents(tmp_path, directory):
    source, destination = tmp_path / "source", tmp_path / "renamed-ø-😀"
    if directory:
        source.mkdir()
        (source / "metadata.json").write_bytes(b"immutable fixture")
    else:
        source.write_bytes(b"immutable fixture")
    queue.rename(source, destination)
    assert not source.exists()
    assert (destination / "metadata.json" if directory else destination).read_bytes() == b"immutable fixture"


def test_missing_source_surfaces_without_creating_destination(tmp_path):
    with pytest.raises(FileNotFoundError):
        queue.rename(tmp_path / "missing", tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_reader_may_close_before_bounded_retry(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"reader fixture")
    reader = source.open("rb")
    with ThreadPoolExecutor(1) as pool:
        try:
            future = pool.submit(queue.rename, source, destination)
            time.sleep(.05)
            assert not future.done()
            assert reader.read() == b"reader fixture"
        finally:
            reader.close()
        future.result(timeout=3)
    assert destination.read_bytes() == b"reader fixture"


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_reader_remaining_open_stops_without_move(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"held reader fixture")
    with source.open("rb"):
        with pytest.raises(PermissionError) as failure:
            queue.rename(source, destination)
    assert failure.value.winerror in {5, 32}
    assert source.read_bytes() == b"held reader fixture"
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows SMB handle API")
def test_smb_validation_fixture_when_requested():
    """An explicit validation-only NAS root exercises the actual SMB API."""
    configured = os.environ.get("DARSIA_RENAME_VALIDATION_ROOT")
    if configured is None:
        pytest.skip("No explicit NAS validation fixture root requested")
    root = Path(configured)
    queue.require_unc(str(root))
    if "code/validation/" not in str(root).replace("\\", "/").casefold():
        raise ValueError("NAS fixture must stay under code/validation")
    owned = root / ("rename_smb_" + uuid.uuid4().hex)
    owned.mkdir()
    source, destination = owned / "pending.json", owned / "claimed.json"
    source.write_bytes(b"SMB immutable metadata fixture")
    queue.rename(source, destination)
    assert destination.read_bytes() == b"SMB immutable metadata fixture"
    source_directory, final_directory = owned / "lease", owned / "published"
    source_directory.mkdir()
    (source_directory / "metadata.json").write_bytes(b"SMB directory fixture")
    queue.rename(source_directory, final_directory)
    assert (final_directory / "metadata.json").read_bytes() == b"SMB directory fixture"
    print("Preserved NAS validation fixture:", owned)
