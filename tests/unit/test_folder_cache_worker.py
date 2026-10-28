"""Ordinary cache startup plus real private queue/worker/process integration.

No real user job, NAS volume, RAW decoder or scientific qualification runs.
Only the tiny science executor/native input boundary is replaced in the
spawned process; all cache bootstrap, queue and bounded output IO is real.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from darsia_cache import adapters, bootstrap, folders
from darsia_cache.policy import CachePolicyError
from darsia.fluidflower import job_control, store_map
from darsia.fluidflower.pipeline import capacity, job_queue, series_setup_jobs
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from .test_series_setup_inputs import controls, _doc  # noqa: F401
from .test_series_setup_protocol import _job


def _stores(tmp_path, *, machine="private-host", roots=None):
    roots = dict(roots or dict(series=str(tmp_path / "series"), results=str(tmp_path / "Jobs")))
    roots["cache"] = str(tmp_path / "cache")
    for path in roots.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    raw = ('[machine]\nid=' + json.dumps(machine) + '\n[stores.synthetic]\n'
           + ''.join(key + '=' + json.dumps(value) + '\n' for key, value in roots.items())).encode()
    path = tmp_path / (machine + "-stores.toml")
    path.write_bytes(raw)
    return store_map.parse_store_map(raw, source=dict(path=str(path)))


def _activate(stores, root, monkeypatch):
    monkeypatch.setattr(folders, "_CURRENT", None)
    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
    source = {key: stores.source[key] for key in ("path", "sha256", "size_bytes")}
    environment = folders.folder_environment(stores.document, source, "synthetic",
        job_root=root, parent_environment=os.environ)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    folders.activate_folder(stores.document, source, "synthetic", job_root=root)


def test_folder_bootstrap_precedes_numerical_imports_without_managed_lease(tmp_path):
    stores = _stores(tmp_path)
    root = tmp_path / "Jobs" / "private-job"
    root.mkdir()
    (root / "job.json").write_text(json.dumps(dict(series_id="synthetic")), encoding="utf-8")
    package = Path(__file__).resolve().parents[2] / "src" / "darsia" / "__init__.py"
    script = '''
import json, os, sys
from darsia_cache import bootstrap, folders
sys.orig_argv = [sys.executable, '-m', 'darsia.fluidflower.worker', '--job', sys.argv[1], '--stores', sys.argv[2]]
context = bootstrap.before_darsia_import(package_file=sys.argv[3])
assert bootstrap._LEASE is None
assert 'numba' not in sys.modules and 'matplotlib' not in sys.modules and 'darsia' not in sys.modules
assert context.pid == os.getpid()
assert sys.pycache_prefix == os.environ['PYTHONPYCACHEPREFIX']
assert os.environ['NUMBA_CACHE_DIR'].startswith(str(context.root))
assert 'DARSIA_CACHE_BOOTSTRAP_RECEIPT' not in os.environ
print(json.dumps(dict(root=str(context.root), mode='folder')))
'''
    environment = dict(os.environ)
    for key in (folders.FOLDER_ENV, bootstrap.RECEIPT_ENV, bootstrap.EXPECTED_STORES_ENV, bootstrap.POLICY_ENV):
        environment.pop(key, None)
    process = subprocess.run([sys.executable, "-B", "-c", script, str(root), stores.source["path"], str(package)],
        env=environment, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["root"] == str(tmp_path / "cache" / "private-host")


def test_unc_folder_is_explicit_and_legacy_policy_does_not_select_other_series():
    document = dict(stores={"synthetic": dict(series=r"\\server\science\Series",
        results=r"\\server\science\Jobs", cache=r"\\server\science\cache"),
        "other": dict(series=r"C:\Series", results=r"C:\Jobs")},
        cache={"deliberately_incomplete_managed_policy": True})
    assert folders.folder_roots(document, "synthetic")["cache"] == r"\\server\science\cache"
    assert folders.folder_roots(document, "other") is None
    stores = SimpleNamespace(machine_id="host", document=document,
        source=dict(path="explicit.toml", sha256="a" * 64, size_bytes=17), roots=lambda _series: {})
    with pytest.raises(CachePolicyError):
        adapters.independent_worker_environment(stores, machine="host", series_id="other",
            explicit_stores=True, parent_environment={})


@pytest.mark.skipif(os.name != "nt", reason="Mapped-drive aliases are Windows paths")
def test_cache_cannot_alias_durable_results_through_a_mapped_drive(monkeypatch):
    cache = Path(r"Z:\disposable")
    results = Path(r"\\fileserver\science\Results")
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path in (cache, results):
            return SimpleNamespace(st_dev=17, st_ino=23)
        return original_stat(path, *args, **kwargs)

    def forbidden_write(*args, **kwargs):
        pytest.fail("Reject a cache alias before creating directories")

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "mkdir", forbidden_write)
    document = dict(machine=dict(id="host"), stores={"synthetic":
        dict(results=str(results), cache=str(cache))})
    with pytest.raises(CachePolicyError, match="must not contain the results folder"):
        folders.folder_environment(document, {}, "synthetic", parent_environment={})


def test_cache_is_per_computer_and_missing_cache_has_gui_action(tmp_path):
    first, second = _stores(tmp_path), _stores(tmp_path, machine="other-host")
    one = adapters.independent_worker_environment(first, machine=first.machine_id, series_id="synthetic",
        explicit_stores=True, parent_environment={bootstrap.RECEIPT_ENV: "obsolete"})
    two = adapters.independent_worker_environment(second, machine=second.machine_id, series_id="synthetic",
        explicit_stores=True, parent_environment={})
    assert one["NUMBA_CACHE_DIR"] != two["NUMBA_CACHE_DIR"]
    assert bootstrap.RECEIPT_ENV not in one
    no_cache = deepcopy(first.document)
    no_cache["stores"]["synthetic"].pop("cache")
    missing = SimpleNamespace(machine_id=first.machine_id, document=no_cache, source=first.source,
        roots=lambda _series: first.roots("synthetic"))
    with pytest.raises(CachePolicyError, match="Start > Series folders"):
        adapters.independent_worker_environment(missing, machine=first.machine_id, series_id="synthetic",
            explicit_stores=True, parent_environment={})


def _private_build(controls):
    """A real typed Build task/catalogue over tiny synthetic input leaves."""
    case = controls
    root = case.root / "Jobs" / "fixture-job"
    root.parent.mkdir()
    Path(case.request["output"]["intended_root"]).mkdir()
    config = deepcopy(case.build)
    config["metadata_output"] = dict(maximum_output_bytes=8 * 1024**2, maximum_file_bytes=1024**2)
    config["output_scope"] = dict(metadata_roots=[str(root.parent)], scratch_roots=[])
    case.plan["build_plans"][0].update(metadata_output=config["metadata_output"], output_scope=config["output_scope"])
    config["setup_plan"] = _doc(case.plan_path, case.plan)
    case.coordinator["setup_plan"] = config["setup_plan"]
    config["coordinator_completion"] = _doc(case.coordinator_path, case.coordinator)
    bound = _doc(case.root / "control" / "fixture-build.json", config)
    identity = dict(job_id="fixture-job", commit="1" * 40, precision="float32", threads=1)
    task = series_setup_jobs._task(bound, config, identity, {"private": str(case.root)}, InputCatalog)
    job = dict(_job(task), **identity, series_id="synthetic", format_version=1, legacy_paths=False,
        execution_ready=True, runtime_support="darsia-role-io-runtime-v1", storage_profile="full",
        environment_lock=dict(path="requirements-worker.lock", sha256="2" * 64))
    queue = job_queue.JobQueue(root, machine_id="private-host")
    job_queue.write_immutable(root / "job.json", job)
    queued = queue.enqueue(task)
    return root, job, job_queue.read_json(queued)


@pytest.mark.parametrize("pause_resume", [False, True])
def test_real_spawned_worker_progresses_and_completes_without_managed_cache(controls, monkeypatch, pause_resume):
    from darsia.fluidflower.pipeline import worker
    from ._folder_worker_fixture import work
    root, job, task = _private_build(controls)
    stores = _stores(controls.root, roots=dict(series=str(controls.root / "project"),
        private=str(controls.root), results=str(root.parent)))
    _activate(stores, root, monkeypatch)
    monkeypatch.setattr(worker, "_work", work)
    args = SimpleNamespace(job=root, machine="private-host", backend="cpu", workers=1,
        once=True, exit_when_idle=True, watchdog=False, watchdog_seconds=30,
        _alias_stores=stores, stores=stores.source["path"], _cache_terminal_events=[])
    if pause_resume:
        monkeypatch.setenv("DARSIA_TEST_FOLDER_PAUSE", "1")
    with (root / "logs" / "private-host-cpu.log").open("a", encoding="utf-8") as log:
        result = worker._parent(args, "private-host-cpu", job, task["commit"], log)
    if pause_resume:
        assert result == 0
        assert not (root / "done" / "sample.json").exists()
        original = root / "leases" / ("sample." + task["attempt_id"])
        retired = root / "sample" / ".attempts" / task["attempt_id"] / "retirement.json"
        assert retired.is_file()
        assert job_queue.read_json(retired)["all_absent"] is True
        # Advance this private fixture's filesystem clock observation beyond
        # the original 600 s lease. Production timing is unchanged.
        for path in [original / "owner.json", *original.glob("progress_*.json")]:
            old = time.time() - 601
            os.utime(path, (old, old))
        queue = job_queue.JobQueue(root, machine_id="private-host")
        reclaimed = queue.reclaim_stale()
        assert len(reclaimed) == 1 and reclaimed[0]["previous_attempt"] == task["attempt_id"]
        task = job_queue.read_json(root / "queue" / "sample.json")
        prior = job_control.read_control(root)
        job_control.publish_control(root, "run", machine="private-host", expected_previous=prior["command"])
        monkeypatch.delenv("DARSIA_TEST_FOLDER_PAUSE")
        shutil.rmtree(Path(stores.roots("synthetic")["cache"]))
        # Recreate only this computer's disposable imports, then resume from
        # shared job state with a new actual child and claim generation.
        _activate(stores, root, monkeypatch)
        with (root / "logs" / "private-host-cpu.log").open("a", encoding="utf-8") as log:
            result = worker._parent(args, "private-host-cpu", job, task["commit"], log)
    log = (root / "logs" / "private-host-cpu.log").read_text(encoding="utf-8")
    assert result == 0, log
    assert '"kind": "started"' in log and "tiny_fixture_compute" in log
    assert (root / "done" / "sample.json").is_file(), log
    assert not list((root / "failed").glob("*.json")), log
    done = job_queue.read_json(root / "done" / "sample.json")
    completed = job_queue.read_json(root / done["completed"]["relative_path"])
    assert completed["actual_worker"]["pid"] != os.getpid()
    assert completed["attempt_id"] == task["attempt_id"]
    reservations = list((root / ".capacity").glob("*/r-*/reserved.json"))
    assert reservations and all((path.parent / "released.json").is_file() for path in reservations)
    cache = Path(stores.roots("synthetic")["cache"])
    assert not list(cache.rglob("owner.json")) and not list(cache.rglob("reserved.json"))
    assert not list(cache.rglob("operation_result.json"))
    shutil.rmtree(cache)
    # A different computer can read the committed result solely from Jobs.
    assert job_queue.read_json(root / done["completed"]["relative_path"]) == completed
    assert (root / "done" / "sample.json").is_file()


def test_folder_ledger_keeps_ownership_and_retained_budget_after_cache_loss(tmp_path):
    stores = _stores(tmp_path)
    root = tmp_path / "Jobs" / "fixture-job"
    root.mkdir()
    ledger = capacity.CapacityLedger(stores, "synthetic", root, job_root=root)
    from darsia.fluidflower.pipeline.folder_capacity import FolderSpaceEstimate
    estimate = FolderSpaceEstimate("sample", 1, 17, 5)
    task = dict(run="sample", attempt_id="a" * 32, ordered_case_ids=["case"])
    owner = dict(machine="private-host-cpu", pid=os.getpid(), process_create_time="1.25")
    with ledger.locked():
        reservation = ledger.bind(root, SimpleNamespace(task=task, token=task["attempt_id"]), owner, estimate)
    shutil.rmtree(tmp_path / "cache")
    other = _stores(tmp_path, machine="other-host")
    resumed = capacity.CapacityLedger(other, "synthetic", root, job_root=root)
    assert resumed.remaining() == 22
    foreign = dict(machine="other-host-cpu", pid=os.getpid(), process_create_time="1.25")
    with pytest.raises(ValueError, match="original owner"):
        resumed.release(reservation, foreign)
    retirement = dict(complete=True, all_absent=True, worker_os_references_closed=True,
        supervisor=foreign, run="sample", attempt_id=task["attempt_id"], identities=[owner])
    with resumed.locked():
        resumed.release(reservation, foreign, retirement=retirement)
    assert ledger.remaining() == 0


def test_disposable_cache_cannot_bind_job_or_input(tmp_path):
    stores = _stores(tmp_path)
    with pytest.raises(CachePolicyError, match="outside"):
        folders.require_durable_path(stores, "synthetic", tmp_path / "cache" / "checkpoint.json")
    with pytest.raises(CachePolicyError, match="outside"):
        capacity.CapacityLedger(stores, "synthetic", tmp_path / "Jobs", job_root=tmp_path / "cache" / "job")


def test_shared_job_prefix_can_move_computers_but_foreign_local_prefix_stays_guarded(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline.run_io import alias_resume_output, require_retired_source, RunIOBinding
    from darsia.fluidflower.job_filesystem import inspect_results_root
    stores = _stores(tmp_path, machine="other-host")
    root = tmp_path / "Jobs" / "fixture-job"
    root.mkdir()
    _activate(stores, root, monkeypatch)
    task = dict(job_id="fixture-job", run="sample")
    token = "a" * 32
    output = root / "sample" / ".attempts" / token
    binding = job_queue.LocalOutputBinding("other-host", root.parent, output,
        inspect_results_root(root.parent))
    owner = dict(format="darsia-alias-lease-owner-v1", machine="private-host-cpu",
        actual_owner=dict(machine="private-host-cpu"), output=dict(machine_id="private-host",
        root_alias="results", relative_path="fixture-job/sample/.attempts/" + token))
    assert alias_resume_output(owner, binding, task, token) == output
    proof = dict(complete=True, all_absent=True, worker_os_references_closed=True,
        run="sample", attempt_id=token, attempt_directory=str(output), identities=[dict(pid=123)],
        machine_id="private-host", supervisor=dict(machine="private-host-cpu"))
    require_retired_source(task, binding, output, json.dumps(proof).encode(), "private-host-cpu")
    with pytest.raises(ValueError):
        require_retired_source(task, binding, output, json.dumps(proof).encode(), "wrong-host-cpu")
    output.mkdir(parents=True)
    retired = output / "retirement.json"
    retired.write_bytes(json.dumps(proof).encode())
    source = dict(attempt_directory=str(output), retirement=job_queue.descriptor(retired))
    current = root / "sample" / ".attempts" / ("b" * 32)
    runtime = RunIOBinding.__new__(RunIOBinding)
    runtime.local_output, runtime.task, runtime.machine_id = binding, task, "other-host"
    runtime.output, runtime.output_identity, runtime.results_identity = current, current, root.parent
    runtime._prefix_validation_root = None
    runtime.resume_roots, runtime.producer_origins = set(), {}
    with runtime.prefix_validation(source):
        assert runtime._prefix_validation_root == output
    runtime.admit_resume_root(source)
    assert runtime.producer_origin_for_attempt(output)["machine_id"] == "private-host"
    monkeypatch.setattr(folders, "_CURRENT", None)
    with pytest.raises(ValueError, match="another machine"):
        alias_resume_output(owner, binding, task, token)


def test_runtime_refuses_essential_input_from_cache(controls, monkeypatch):
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog, RunUnavailable
    root, _job_value, task = _private_build(controls)
    stores = _stores(controls.root, roots=dict(series=str(controls.root / "project"),
        private=str(controls.root), results=str(root.parent)))
    row = next(row for row in task["input_catalog"]["inputs"] if "setup.native_parameters" in row["uses"])
    row["reference"]["root_alias"] = "cache"
    from darsia.fluidflower.pipeline.job_inputs import catalog_order_key
    task["input_catalog"]["inputs"].sort(key=lambda item: catalog_order_key(item["reference"]))
    with pytest.raises(RunUnavailable, match="Disposable Cache"):
        resolve_catalog(task, stores, "synthetic")


def test_setup_replays_reuse_unchanged_admitted_bytes_and_recheck_changed_inputs(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import store_runtime
    path = tmp_path / "declared.json"
    path.write_bytes(b"{}")
    import hashlib
    reference = dict(sha256=hashlib.sha256(b"{}").hexdigest(), bytes=2)
    rows = (store_runtime.ResolvedInput(reference, path, ("fixture",)),)
    task = dict(run="sample")
    first = store_runtime.verify_inputs(task, rows)
    calls = []
    original = store_runtime._verify_input_bytes
    def observe(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(store_runtime, "_verify_input_bytes", observe)
    assert store_runtime.verify_inputs(task, rows, previous=first) == first
    assert not calls
    path.write_bytes(b"[]")
    # The rewrite keeps inode and size, and can land in the first write's
    # clock tick; a later modification time makes it a changed version.
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    with pytest.raises(store_runtime.InputHashMismatch):
        store_runtime.verify_inputs(task, rows, previous=first)
    assert calls == [path]
