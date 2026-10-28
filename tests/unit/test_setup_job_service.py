"""Operational Setup startup contracts; these fixtures perform no science."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import tomllib
from pathlib import Path
import sys
import subprocess

import pytest

from darsia_cache import first_qualification as original, setup_session as session
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from darsia_cache.setup_service import stores_bytes
from .test_setup_service_qualification import protocol, pin


def prepared(mutate=None):
    old, objects, _reads = protocol()
    provider = objects[old.provider_binding["path"]]
    provider.update(format=session.PROVIDER, workers=6,
        slots={name: "session-" + name + ".json" for name in ("observation", "native_completion", "completion")})
    provider.pop("operations")
    source = pin("src/darsia_cache/setup_session.py")
    provider["producer_source_closure"].append(source)
    provider["module_closure"].append(source)
    request = objects[old.binding["path"]]
    request.update(format=session.REQUEST, budget="fresh-fixed-volume-headroom-with-original-reserves")
    for key in ("operation", "maximum_service_bytes", "predecessor"):
        request.pop(key)
    request["public_command"] = [provider["python_executable"]["path"], "-B", "-m", "darsia.fluidflower.worker",
        "--job", request["job_root"], "--backend", "cpu", "--workers", "6", "--machine", "fixture",
        "--watchdog", "--exit-when-idle", "--stores", old.stores_source["path"]]
    if mutate:
        mutate(provider, request, objects)
    objects[request["root_release"]["path"]] = dict(format=session.RELEASE, execution_authorized=True,
        provider=old.provider_binding, stores_source=old.stores_source, scope=deepcopy(provider["scope"]),
        request_payload_sha256=original._digest({key: value for key, value in request.items() if key != "root_release"}))
    read = lambda bound: deepcopy(objects[bound["path"]])
    first = session.SetupJobService(old.binding, provider_binding=old.provider_binding,
        stores_source=old.stores_source, namespace=provider["namespace_admission"], read=read, read_metadata=read)
    return first, objects


def test_real_apply_and_later_build_share_only_their_immutable_setup_job():
    first, _ = prepared()
    lease = SimpleNamespace(first_qualification=first, source=first.stores_source)
    assert first.require_task(lease, first.task) is True
    build = deepcopy(first.task)
    build.update(run="run-a", operation="build", ordered_case_ids=["build"], attempt_id="3" * 32)
    assert first.require_task(lease, build) is True
    assert first.setup_mode and first.setup_job_mode
    assert first.reference_build is False and first.bound_mode is False
    assert "import_profile" not in first.provider["slots"]
    assert "maximum_service_bytes" not in first.request


@pytest.mark.parametrize("change", [
    lambda p, r, o: p.update(workers=2),
    lambda p, r, o: p["scope"].update(pipeline="native-phase-statistics"),
    lambda p, r, o: r.update(job=pin("other-job.json")),
    lambda p, r, o: r.update(budget="borrow-old-measurement"),
    lambda p, r, o: r["public_command"].append("--once"),
    lambda p, r, o: o[r["task"]["path"]].update(commit="f" * 40),
    lambda p, r, o: o[r["configuration"]["path"]]["metadata_output"].update(maximum_output_bytes=0),
])
def test_start_refuses_other_source_job_concurrency_or_missing_output_caps(change):
    with pytest.raises(CachePolicyError):
        prepared(change)


@pytest.mark.parametrize("change", [dict(job_id="another"), dict(commit="f" * 40),
    dict(pipeline="native-phase-statistics"), dict(target_machine="other"), dict(run="absent"),
    dict(operation="build", ordered_case_ids=["build"]), dict(attempt_id="not-an-attempt")])
def test_claim_cannot_escape_the_selected_setup_plan(change):
    first, _ = prepared()
    task = dict(first.task, **change)
    with pytest.raises(CachePolicyError):
        first.require_task(SimpleNamespace(first_qualification=first, source=first.stores_source), task)


def test_two_actual_attempts_get_separate_consumption_markers_without_mutating_launch_identity():
    first, _ = prepared()
    records = {}
    class Ledger:
        def _require_lock(self): pass
        @property
        def control(self): return self
        def names(self): return list(records)
        def write_once(self, name, raw):
            assert name not in records
            records[name] = json.loads(raw)
    lease = SimpleNamespace(ledger=Ledger(), owner={"TESTONLY": "actual fixture owner"})
    key = first._once_key()
    first._remember_claim_locked(lease, first.task)
    build = dict(first.task, run="run-a", operation="build", ordered_case_ids=["build"], attempt_id="3" * 32)
    first._remember_claim_locked(lease, build)
    assert first._once_key() == key
    assert len(records) == 4
    assert {value["task"]["attempt_id"] for value in records.values() if "task" in value} == {"2" * 32, "3" * 32}


def test_service_budget_keeps_used_space_and_every_other_unwritten_reservation(monkeypatch):
    first, _ = prepared()
    first._require_launch_admission = lambda lease: None
    report = dict(hard_filesystem_quota_verified=True, reserve_bytes=PHYSICAL_RESERVE_BYTES,
        quota_headroom_bytes=200 * 1024**3, free_physical_bytes=100 * 1024**3,
        active_cache_remaining_bytes=7 * 1024**3, required_noncache_bytes=3 * 1024**3)
    lease = SimpleNamespace(ledger=SimpleNamespace(project=lambda maximum: deepcopy(report)))
    assert first.service_entitlement(lease) == 49 * 1024**3
    assert first.administrative["measured_profile"] is False
    report["free_physical_bytes"] = PHYSICAL_RESERVE_BYTES
    with pytest.raises(CachePolicyError):
        first.service_entitlement(lease)


def test_idle_partial_service_is_not_whole_setup_completion():
    first, _ = prepared()
    published = dict(job=first.request["job"], terminal_failures=[],
        original_done_and_product_index_metadata_verified=True,
        completed=[dict(run=run) for run in first.job["expected_runs"]])
    assert first.whole_job_published(published)
    published["completed"].pop()
    assert not first.whole_job_published(published)
    published["completed"] = [dict(run="apply-all")] * len(first.job["expected_runs"])
    assert not first.whole_job_published(published)


@pytest.fixture
def retired_setup_supervisor(monkeypatch):
    """Exercise original retirement, lifecycle and budget close with memory transports."""
    from darsia_cache import bootstrap, owned_process, service_context
    from darsia_cache.control import CacheBudgetLedger, _encoded
    from darsia_cache.service_lifecycle import ServiceLifecycle

    def make(*, outcome="failed", whole_job=True, unresolved_child=False,
             reject_close_recheck=False, setup_job=True, exit_code=1, claim_changes=None):
        first, objects = prepared()
        first.setup_job_mode = setup_job
        claims = [dict(first.task, **change) for change in
                  ([{}] if claim_changes is None else claim_changes)]
        slots, events, reads = {}, [], []
        token = "4" * 32
        owner = dict(pid=101, process_create_time="1.0", creation_filetime_100ns="101")
        observer = dict(pid=999, process_create_time="9.0", creation_filetime_100ns="999")
        receipt_root = Path(first.provider["output_namespace"]["path"])

        def store(path, value):
            raw = _encoded(value)
            binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            objects[binding["path"]] = deepcopy(value)
            return binding

        class Ledger(CacheBudgetLedger):
            def __init__(self):
                self.key, self.lock_active = "synthetic-pool", False
                self.bound, self.control = dict(path=str(receipt_root)), self
                self.rows = {}

            @contextmanager
            def locked(self, *, create_binding=True):
                assert not self.lock_active
                self.lock_active = True
                try:
                    yield
                finally:
                    self.lock_active = False

            def names(self):
                self._require_lock()
                return list(self.rows)

            def read_once(self, name):
                self._require_lock()
                return self.rows[name], {}

            def write_once(self, name, raw):
                self._require_lock()
                assert name not in self.rows
                self.rows[name] = raw
                store(receipt_root / name, json.loads(raw))

            def close(self, *args, **kwargs):
                events.append("reservation-close")
                kernel.closing_reservation = True
                return super().close(*args, **kwargs)

        ledger = Ledger()
        reserved = dict(format="darsia-cache-budget-reservation-v2", token=token,
                        pool=ledger.key, owner=owner, maximum_bytes=428 * 1024**3)
        raw = _encoded(reserved)
        ledger.rows["cache-r-" + token + ".json"] = raw
        ledger.rows["unrelated-retained-record.json"] = b'{"retained":true}\n'
        reservation = dict(token=token, pool=ledger.key, sha256=hashlib.sha256(raw).hexdigest())
        launch = store(receipt_root / "native-launch.json", dict(TESTONLY="synthetic native launch"))
        root = Path("C:/TESTONLY/cache/jobs") / token
        manifest = dict(owner=owner, reservation=reservation, root=str(root),
                        entry_kind="worker", role_identifier="fixture_jobs", native_launch_context=launch)
        context = SimpleNamespace(reservation=reservation, manifest=manifest, owner=owner,
            source=first.stores_source, profile_binding=first.provider_binding, root=root,
            document=dict(cache_bootstrap=dict(namespace_admission=first.provider["namespace_admission"])),
            entry_kind="worker", role_identifier="fixture_jobs", ledger=ledger,
            maximum=reserved["maximum_bytes"], entry_limit=100,
            policy=SimpleNamespace(failed_retention_hours=48))
        service_pin = store(receipt_root / "service.json", dict(manifest=manifest))
        context.receipt = service_pin
        lease = SimpleNamespace(first_qualification=first, source=first.stores_source,
            profile_binding=first.provider_binding, ledger=ledger, package_file="TESTONLY-package",
            admission=dict(receipts=first.provider["output_namespace"]),
            _record=lambda name, value: store(receipt_root / name, value))

        class Kernel:
            closing_reservation = False

            def accounting(self, job):
                return dict(active_processes=int(reject_close_recheck and self.closing_reservation),
                            total_processes=7)

            def require_job_flags(self, job):
                return {}

            def signaled(self, handle):
                return True

            def identity(self, handle):
                return {key: owner[key] for key in ("pid", "creation_filetime_100ns")}

            def exit_code(self, handle):
                return exit_code

            def close(self, handle):
                events.append("handle-close")

        kernel = Kernel()
        controlled = owned_process.OwnedWorker(lease, api=kernel)
        controlled.job, controlled.owner_handle = "synthetic-job", 3
        controlled.info = dict(process=1, thread=2)
        controlled.native_primary, controlled.observed_binding = owner, launch
        controlled.service_binding = service_pin
        controlled.primary_binding = store(receipt_root / "primary.json", dict(
            actual_service_receipt=service_pin, original_native_launch=launch, actual_primary=owner,
            held_process_handle_observed_while_live=True, same_original_job_membership_observed=True))
        controlled.intent = dict(controller=observer, job_name="synthetic-job")
        controlled.start = lambda *args: events.append("native-start")

        monkeypatch.setattr(owned_process, "current_native_membership", lambda: launch)
        lifecycle = ServiceLifecycle(context)
        lifecycle.register_owner()
        children = []
        for slot in range(6):
            details = dict(slot=slot, generation=str(slot + 1) * 32)
            child = dict(pid=200 + slot, process_create_time=str(slot + 2),
                         creation_filetime_100ns=str(200 + slot), native_launch_context=launch)
            children.append(child)
            lifecycle.record_launch("worker_spawn_intent", details)
            lifecycle.record_launch("worker_spawn_observed", dict(details, original_native_identity=child))
            if not (unresolved_child and slot == 5):
                lifecycle.record_launch("worker_children_retired", dict(details,
                    original_child_identities=[child], owned_launch_handle_retired=True))
        claim_owners = [owner]
        for child in children[:max(0, len(claims) - 1)]:
            child_context = SimpleNamespace(**vars(context))
            child_context.owner = child
            ServiceLifecycle(child_context).register_owner()
            claim_owners.append(child)
        original_traceback = "Traceback (synthetic worker):\nRuntimeError: failed before RunIO prepare"
        publication = dict(job=first.request["job"], terminal_failures=[],
            original_done_and_product_index_metadata_verified=True,
            completed=[dict(run=run) for run in first.job["expected_runs"]])
        if not whole_job:
            publication["completed"].pop()
        publication_pin = store(receipt_root / "publication.json", publication)
        seal = lifecycle.seal(logical_gate="synthetic-setup", outcome=outcome,
            utc="2026-01-01T00:00:00+00:00",
            traceback_text=original_traceback if outcome == "failed" else None,
            publication=publication_pin if outcome == "success" else None)
        before = deepcopy(ledger.rows)
        exit_inventories = {}
        for index, claim_owner in enumerate(claim_owners):
            modules = dict(request=first.binding, provider=first.provider_binding,
                actual_task=claims[index] if index < len(claims) else None,
                predeclared_module_closure_sha256=original._digest(first.provider["module_closure"]),
                all_file_backed_modules_predeclared_and_exact=True, undeclared=[])
            exit_name = (token + ".exit-intent-" + str(claim_owner["pid"]) + "-"
                         + claim_owner["creation_filetime_100ns"] + ".json")
            exit_inventories[claim_owner["pid"]] = store(receipt_root / exit_name,
                dict(owner=claim_owner, first_qualification_modules=modules))
        objects[first.provider["startup"]["path"]] = dict(source_files=[])

        def exact(binding, package):
            reads.append(binding)
            assert binding is not None, "Failed seals have no publication descriptor"
            return deepcopy(objects[binding["path"]])

        def control(actual_lease, name):
            value = objects[str(receipt_root / name)]
            return deepcopy(value), store(receipt_root / name, value)

        def write_slot(provider, role, value):
            assert role not in slots
            slots[role] = deepcopy(value)
            return store(receipt_root / provider["slots"][role], value)

        observations = SimpleNamespace(reserve_crossed=False, observe=lambda *a, **kw: None,
            document=lambda *args: dict(conservative_family_process_bound_from_kernel_total=7,
                                        sampled_active_processes_high_water=7))
        monkeypatch.setattr(bootstrap, "_owner", lambda: (observer, {}))
        monkeypatch.setattr(bootstrap, "_utc", lambda: "2026-01-01T00:00:01+00:00")
        monkeypatch.setattr(bootstrap, "_exact", exact)
        monkeypatch.setattr(owned_process, "_exact_source", lambda binding: None)
        monkeypatch.setattr(owned_process, "_control", control)
        monkeypatch.setattr(owned_process, "_public_command", lambda lease: first.request["public_command"])
        monkeypatch.setattr(owned_process, "OwnedWorker", lambda *a, **kw: controlled)
        monkeypatch.setattr(service_context, "resolve_original_service_context", lambda *a, **kw: context)
        monkeypatch.setattr("darsia_cache.adapters.verify_worker_publication",
                            lambda snapshot, binding: binding == publication_pin)
        monkeypatch.setattr(original, "validate_startup", lambda lease: None)
        monkeypatch.setattr(first, "consume", lambda lease: None)
        monkeypatch.setattr(original, "FiniteObservations", lambda first: observations)
        monkeypatch.setattr(original, "_slot_write", write_slot)
        return SimpleNamespace(lease=lease, ledger=ledger, slots=slots, before=before,
            close_name="cache-c-" + token + ".json", trace=original_traceback,
            seal=seal, publication=publication_pin, events=events, reads=reads,
            claims=claims, exit_inventories=exit_inventories)

    return make


@pytest.mark.parametrize("exit_code", [0, 1])
def test_failed_setup_seal_retains_original_failure_and_closes_only_unwritten_reservation(
        retired_setup_supervisor, exit_code):
    world = retired_setup_supervisor(exit_code=exit_code)
    with pytest.raises(CachePolicyError, match="failed before RunIO prepare") as failed:
        original.supervise_first_worker(world.lease)
    assert world.trace in str(failed.value)
    native = world.slots["native_completion"]
    assert native["complete"] is True and native["passed"] is False
    assert native["observer_native_handles_closed"] is True
    assert native["cache_reservation_closed"] is True
    assert native["service_failure"] == dict(seal=world.seal, traceback=world.trace)
    assert native["eligibility"]["retention_due"] is False
    assert "completion" not in world.slots and world.publication not in world.reads
    assert all(world.ledger.rows[name] == raw for name, raw in world.before.items())
    added = set(world.ledger.rows) - set(world.before)
    assert world.close_name in added
    assert all(name == world.close_name or "-eligibility-" in name for name in added)
    assert world.events.index("reservation-close") < world.events.index("handle-close")


@pytest.mark.parametrize("whole_job", [False, True])
def test_retired_successful_setup_requires_whole_job_publication_for_completion(
        retired_setup_supervisor, whole_job):
    world = retired_setup_supervisor(outcome="success", whole_job=whole_job, exit_code=0)
    if whole_job:
        assert original.supervise_first_worker(world.lease) == 0
        assert world.slots["completion"]["passed"] is True
    else:
        with pytest.raises(CachePolicyError, match="Setup service stopped"):
            original.supervise_first_worker(world.lease)
        assert "completion" not in world.slots
    assert world.slots["native_completion"]["passed"] is whole_job
    assert world.close_name in world.ledger.rows and world.publication in world.reads
    assert all(world.ledger.rows[name] == raw for name, raw in world.before.items())


@pytest.mark.parametrize("refusal", ["unresolved-child", "close-recheck"])
def test_failed_setup_keeps_reservation_when_original_retirement_is_refused(
        retired_setup_supervisor, refusal):
    world = retired_setup_supervisor(unresolved_child=refusal == "unresolved-child",
                                    reject_close_recheck=refusal == "close-recheck")
    with pytest.raises(CachePolicyError, match="Unresolved launch|must be retired") as failed:
        original.supervise_first_worker(world.lease)
    if refusal == "close-recheck":
        assert world.trace in "\n".join(failed.value.__notes__)
    assert world.close_name not in world.ledger.rows
    assert "native_completion" not in world.slots and "completion" not in world.slots
    assert all(world.ledger.rows[name] == raw for name, raw in world.before.items())


def test_failed_other_setup_qualification_does_not_release_its_reservation(retired_setup_supervisor):
    world = retired_setup_supervisor(setup_job=False)
    with pytest.raises(CachePolicyError, match="Setup service qualification stopped"):
        original.supervise_first_worker(world.lease)
    assert world.close_name not in world.ledger.rows
    assert "cache_reservation_closed" not in world.slots["native_completion"]
    assert world.slots["native_completion"]["passed"] is False


@pytest.mark.parametrize("outcome", ["failed", "success"])
def test_whole_setup_finalizer_checks_and_retains_every_actual_claim(
        retired_setup_supervisor, monkeypatch, outcome):
    world = retired_setup_supervisor(outcome=outcome, exit_code=0 if outcome == "success" else 1,
        claim_changes=[{}, dict(run="run-a", operation="build", ordered_case_ids=["build"],
                                attempt_id="3" * 32)])
    first = world.lease.first_qualification
    checked = []
    require_task = first.require_task

    def require_original_task(lease, task):
        checked.append(deepcopy(task))
        return require_task(lease, task)

    monkeypatch.setattr(first, "require_task", require_original_task)
    if outcome == "success":
        assert original.supervise_first_worker(world.lease) == 0
        assert world.slots["completion"]["passed"] is True
    else:
        with pytest.raises(CachePolicyError, match="failed before RunIO prepare"):
            original.supervise_first_worker(world.lease)
        assert "completion" not in world.slots
    observation = world.slots["observation"]
    actual = observation["actual_claims"]
    assert len(actual) == 2 and checked == actual
    assert {original._digest(task) for task in actual} == {original._digest(task) for task in world.claims}
    assert "actual_claim" not in observation
    inventories = observation["actual_registered_owner_exit_inventories"]
    assert [row["actual_task"] for row in inventories] == actual
    assert {row["owner"]["pid"]: row["inventory"] for row in inventories} == world.exit_inventories
    assert world.slots["native_completion"]["passed"] is (outcome == "success")
    assert world.close_name in world.ledger.rows
    assert all(world.ledger.rows[name] == raw for name, raw in world.before.items())


def test_whole_setup_finalizer_refuses_an_actual_claim_from_another_job(retired_setup_supervisor):
    world = retired_setup_supervisor(claim_changes=[{}, dict(job_id="another-job", run="run-a",
        operation="build", ordered_case_ids=["build"], attempt_id="3" * 32)])
    with pytest.raises(CachePolicyError, match="cannot admit another job"):
        original.supervise_first_worker(world.lease)
    assert world.close_name not in world.ledger.rows
    assert not world.slots
    assert all(world.ledger.rows[name] == raw for name, raw in world.before.items())


@pytest.mark.parametrize("setup_job", [False, True])
def test_finalizer_still_requires_claim_evidence_and_other_modes_still_require_exactly_one(
        retired_setup_supervisor, setup_job):
    changes = [] if setup_job else [{}, dict(run="run-a", operation="build",
        ordered_case_ids=["build"], attempt_id="3" * 32)]
    world = retired_setup_supervisor(setup_job=setup_job, claim_changes=changes)
    message = "At least one actual Setup worker claim" if setup_job else "Exactly one actual worker-slot first claim"
    with pytest.raises(CachePolicyError, match=message):
        original.supervise_first_worker(world.lease)
    assert world.close_name not in world.ledger.rows and not world.slots


def test_private_stores_serialization_preserves_aliases_unicode_and_exact_cap_integers():
    value = dict(machine=dict(id="fixture"), stores=dict(SERIES=dict(series="//server/share/series",
        results="C:\\local\\résults", source_cache="C:\\inputs")),
        cache=dict(quota_bytes=536870912000, profiles=dict(jobs=pin("provider.json"))))
    assert tomllib.loads(stores_bytes(value).decode("utf-8")) == value


def test_large_native_source_does_not_expand_the_control_metadata_reader_limit():
    from darsia_cache.policy import _descriptor, _module_source_descriptor
    value = pin("site-packages/cv2/cv2.pyd", size=75 * 1024**2)
    assert _module_source_descriptor(value) == value
    with pytest.raises(CachePolicyError):
        _descriptor(value)
    with pytest.raises(CachePolicyError):
        _module_source_descriptor(dict(value, path="C:/TESTONLY/oversized-profile.json"))
    with pytest.raises(CachePolicyError):
        _module_source_descriptor(dict(value, sha256="changed"))


def test_cold_startup_capture_declares_its_lazy_metadata_reader_before_application_import(tmp_path):
    target = tmp_path / "probe.startup.json"
    code = r'''
import json,os,sys
from pathlib import Path
from darsia_cache import bootstrap,windows
from darsia_cache.setup_service import capture_startup,CAPTURE_ENV
target=Path(sys.argv[1])
bootstrap._exact=lambda pin,package: ({"cache_bootstrap":{"namespace_admission":{"path":"C:/TESTONLY/namespace.json","sha256":"b"*64,"size_bytes":17}}}
    if pin["path"].endswith("host.json") else {"receipts":{"path":str(target.parent),"identity":{"TESTONLY":True}}})
class Namespace:
    def __init__(self,path,identity): pass
    def write_once(self,name,content,*,maximum):
        assert name==target.name and len(content)<=maximum
        target.write_bytes(content)
    def close(self): pass
windows.WindowsNamespace=Namespace
os.environ[CAPTURE_ENV]=json.dumps({"host_policy":{"path":"C:/TESTONLY/host.json","sha256":"a"*64,"size_bytes":17},"name":target.name})
capture_startup()
raise AssertionError("Capture must exit before the application")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads(target.read_text(encoding="utf-8"))
    assert captured["capture_only"] is True
    assert "darsia_cache.adapters" in captured["allowed_loaded_modules"]
    assert "multiprocessing.spawn" in captured["allowed_loaded_modules"]
    assert not any(name.split(".")[0] in {"numpy", "numba", "cv2", "matplotlib", "PySide6"}
                   for name in captured["allowed_loaded_modules"])
    validate = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
import multiprocessing.spawn,multiprocessing.queues,multiprocessing.popen_spawn_win32,multiprocessing.synchronize
import darsia_cache
from darsia_cache.setup_service import source_pin
from darsia_cache.first_qualification import validate_startup
path=Path(sys.argv[1]); startup=json.loads(path.read_text())
startup["module_closure"]=startup["source_files"]
bound=path.with_name("bound-startup.json");bound.write_text(json.dumps(startup),encoding="utf-8")
provider={"startup":source_pin(bound),"python_executable":startup["python_executable"],"module_closure":startup["module_closure"]}
validate_startup(SimpleNamespace(first_qualification=SimpleNamespace(provider=provider),package_file=darsia_cache.__file__))
'''
    result = subprocess.run([sys.executable, "-B", "-c", validate, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("machine", ["fixture", "FIXTURE", "FiXtUrE", "other-host", "stale-cap"])
def test_complete_launch_metadata_preparation_uses_original_task_and_no_scientific_process(tmp_path, monkeypatch, machine):
    from darsia_cache import setup_service as launch, bootstrap, windows
    from tests.unit.gui.test_cache_volume_cap import document, observed
    from tests.unit.gui.test_cache_first_qualification import bound_plan_TESTONLY
    from .test_series_setup_protocol import _task, _job, _operation
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return launch.source_pin(path)
    root = tmp_path / "fixture-job"
    task = dict(_task("apply", run="apply-all"), job_id=root.name, attempt_id="2" * 32)
    config_pin = write(root / "operation.json", _operation("apply", run="apply-all"))
    ref = dict(task["configuration"], relative_path="operation.json",
               sha256=config_pin["sha256"], bytes=config_pin["size_bytes"])
    old_ref, task["configuration"] = task["configuration"], ref
    task["input_catalog"]["configuration"] = deepcopy(ref)
    for row in task["input_catalog"]["inputs"]:
        if row["reference"] == old_ref:
            row["reference"] = deepcopy(ref)
    job = dict(_job(task), series_id="SERIES", expected_runs=["apply-all", "run-a", "run-b"],
        setup_operations=[dict(run="apply-all", operation="apply"), dict(run="run-a", operation="build"),
                          dict(run="run-b", operation="build")])
    write(root / "job.json", job)
    write(root / "queue" / "apply-all.json", task)
    write(root / "results" / "apply-all" / "task.json", task)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    cache = document()["cache"]
    cache.pop("profiles")
    cache.update(quota_bytes=500 * 1024**3, failed_retention_hours=48)
    cache["volume_cap"]["maximum_size_bytes"] = cache["quota_bytes"]
    cap = __import__("darsia_cache.policy", fromlist=["cache_settings"]).cache_settings(
        dict(cache=dict(cache, profiles={"jobs": config_pin}))).volume_cap
    observation = windows._match_volume_cap(cap, observed(cap))
    admission = dict(cache_identity=cache["volume_cap"]["cache_identity"], volume_cap=cap.document(), host="fixture-host",
        authorized_sids=["fixture-sid"], host_ledger=dict(path="C:/TESTONLY/ledger"),
        receipts=dict(path=str(receipts), identity=dict(TESTONLY="synthetic existing directory")))
    if machine == "stale-cap":
        admission["volume_cap"] = dict(admission["volume_cap"],
            virtual_disk_identifier="11111111-1111-4111-8111-111111111111")
    admission_pin = write(tmp_path / "admission.json", admission)
    bound = bound_plan_TESTONLY()
    bound.update(enforcement_state="PREDECLARED_Q106_VOLUME_BOUND", volume_cap_observation=observation,
        volume_capacity_bytes=observation["volume_total_bytes"], required_capacity_bound_bytes=observation["volume_total_bytes"],
        operational_admission_headroom_bytes=observation["volume_total_bytes"] - PHYSICAL_RESERVE_BYTES)
    host = dict(format=launch.HOST_FORMAT, cache=cache,
        capacity=dict(state_directory="C:/TESTONLY/state", reserve_bytes=PHYSICAL_RESERVE_BYTES),
        cache_bootstrap=dict(namespace_admission=admission_pin, accounting_entry_limit=100000),
        capacity_bound=write(tmp_path / "bound.json", bound))
    selected = dict(machine=dict(id="fixture"), stores=dict(SERIES=dict(series=str(root), configuration=str(root), results="C:/TESTONLY/results")),
        setup_cache=dict(host_policy=write(tmp_path / "host.json", host)))
    stores = tmp_path / "stores.toml"
    stores.write_bytes(launch.stores_bytes(selected))
    class Namespace:
        def __init__(self, path, identity): self.path = Path(path)
        def verify_volume_cap(self, binding): return deepcopy(observation)
        def write_once(self, name, content, *, maximum):
            assert len(content) <= maximum
            with (self.path / name).open("xb") as stream: stream.write(content)
        def close(self): pass
    monkeypatch.setattr(windows, "WindowsNamespace", Namespace)
    monkeypatch.setattr(bootstrap, "_owner", lambda: ({}, dict(host="fixture-host", service_account_sid="fixture-sid",
                                                               common_root="C:/TESTONLY/ledger")))
    monkeypatch.setattr(launch.subprocess, "check_output", lambda cmd, **kwargs: "1" * 40 if "rev-parse" in cmd else "")
    capture_calls = []
    def metadata_probe(command, *, cwd, env, check):
        capture_calls.append(command)
        request = json.loads(env[launch.CAPTURE_ENV])
        write(receipts / request["name"], dict(format="darsia-qualified-worker-startup-source-v1",
            python_executable=launch.source_pin(sys.executable), source_files=[launch.source_pin(sys.executable)]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, "run", metadata_probe)
    def closure(repository, startup):
        pins = []
        for suffix in session.REQUIRED_SOURCE_SUFFIXES | {"darsia_cache/setup_session.py"}:
            path = tmp_path / "source" / "src" / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source, never executed\n", encoding="utf-8")
            pins.append(launch.source_pin(path))
        return pins, [launch.source_pin(sys.executable)]
    monkeypatch.setattr(launch, "source_closures", closure)
    command = [sys.executable, "-B", "-m", "darsia.fluidflower.worker", "--job", str(root), "--backend", "cpu",
        "--workers", "6", "--machine", "fixture" if machine == "stale-cap" else machine,
        "--watchdog", "--exit-when-idle", "--stores", str(stores)]
    if machine in {"other-host", "stale-cap"}:
        with pytest.raises(CachePolicyError, match="machine differs" if machine == "other-host" else "exact volume cap/root"):
            launch.prepare(command, stores_path=stores, repository=tmp_path, environment={})
        assert not capture_calls and not list(receipts.iterdir())
        return
    actual, environment, request_pin = launch.prepare(command, stores_path=stores, repository=tmp_path, environment={})
    assert len(capture_calls) == 1  # The only process seam was the forced pre-import metadata probe.
    assert command[command.index("--machine") + 1] == machine
    assert actual[actual.index("--machine") + 1] == task["target_machine"]
    assert actual[actual.index("--workers") + 1] == "6"
    assert actual[actual.index("--stores") + 1] != str(stores)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    request = json.loads(Path(request_pin["path"]).read_text(encoding="utf-8"))
    provider = json.loads(Path(request["provider"]["path"]).read_text(encoding="utf-8"))
    assert provider["scope"]["target_machine"] == task["target_machine"]
    assert request["public_command"] == actual
    assert request["task"] == launch.source_pin(root / "results" / "apply-all" / "task.json")
    assert json.loads((root / "queue" / "apply-all.json").read_text()) == task
    assert set(root.iterdir()) == {root / name for name in ("job.json", "operation.json", "queue", "results")}
