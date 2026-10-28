"""Setup service wiring on original private claims and injected OS observations.

These tests exercise integration ordering and reserve STOP, not a native launch,
cache-profile qualification, RAW decode or scientific output comparison.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity
from darsia.fluidflower.pipeline import series_setup_worker_admission as admission
from darsia.fluidflower import job_control
from darsia_cache import first_qualification as first
from darsia_cache import owned_process
from darsia_cache import setup_qualification as setup
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES

from .test_series_setup_worker_admission import _runtime, _write_claim
from .gui.test_cache_owned_process import bound_supervisor_TESTONLY
from .gui.test_cache_service_context import world as service_world, path_key, PACKAGE
from .test_setup_service_qualification import protocol


def test_another_finite_qualification_refuses_before_setup_claim_or_controls(tmp_path, monkeypatch):
    runtime, task, config, owner, events, _service = _runtime(tmp_path, monkeypatch)
    pending = runtime.root / "queue" / (task["run"] + ".json")
    original = pending.read_bytes()
    runtime.cache_adapter.lease.first_qualification = SimpleNamespace(setup_mode=False,
        require_task=lambda *_args: pytest.fail("Foreign finite task admission reached"))
    with pytest.raises(ValueError, match="cannot borrow another finite"):
        admission.setup_claim_admission(runtime, task, config, owner)
    assert pending.read_bytes() == original and not events
    assert not (runtime.root / "claimed" / runtime.service_id / (task["run"] + ".json")).exists()
    assert not runtime.capacities


def test_first_setup_consumption_is_locked_and_precedes_real_output_cap_binding(tmp_path, monkeypatch):
    runtime, task, _config, owner, events, service = _runtime(tmp_path, monkeypatch)
    lease = runtime.cache_adapter.lease
    consumed = []

    def require_task(actual_lease, actual_task):
        assert actual_lease is lease and actual_task == task
        assert not service.active
        assert (runtime.root / "queue" / (task["run"] + ".json")).is_file()
        events.append("exact-first-task-before-claim")

    def remember(actual_lease, claim):
        assert actual_lease is lease and service.active
        assert claim.task == task and claim.token == task["attempt_id"]
        assert helper.preclaim_sink.claimed.is_file()
        assert not helper.admission.bound_reservations
        consumed.append(deepcopy(claim.task))
        events.append("actual-first-claim-consumed")

    lease.first_qualification = SimpleNamespace(setup_mode=True,
        require_task=require_task, remember_claim_locked=remember)
    with capacity.ClaimCapacity(runtime, owner) as candidate:
        candidate.availability(task)
        helper = candidate.setup_candidate
        original_bind = helper.admission.bind

        def bind_caps(*args, **kwargs):
            assert service.active and consumed == [task]
            events.append("original-output-cap-bind")
            return original_bind(*args, **kwargs)

        monkeypatch.setattr(helper.admission, "bind", bind_caps)
        claim = _write_claim(helper)
        candidate.bind(claim)
        assert helper.bound and helper.admission.bound_reservations
        assert runtime.setup_admissions[claim.token] is helper
    assert events.index("exact-first-task-before-claim") < events.index("service-locked")
    assert events.index("actual-first-claim-consumed") < events.index("original-output-cap-bind")
    assert events[-1] == "service-unlocked"


def test_failed_first_consumption_retains_claim_and_binds_no_output_cap(tmp_path, monkeypatch):
    runtime, task, _config, owner, _events, service = _runtime(tmp_path, monkeypatch)
    pending = runtime.root / "queue" / (task["run"] + ".json")
    original = pending.read_bytes()
    lease = runtime.cache_adapter.lease

    def refuse(actual_lease, claim):
        assert actual_lease is lease and service.active
        assert claim.task == task and helper.preclaim_sink.claimed.read_bytes() == original
        raise CachePolicyError("Exact first claim consumption could not be recorded")

    lease.first_qualification = SimpleNamespace(setup_mode=True,
        require_task=lambda actual_lease, actual_task: True, remember_claim_locked=refuse)
    with capacity.ClaimCapacity(runtime, owner) as candidate:
        candidate.availability(task)
        helper = candidate.setup_candidate
        claim = _write_claim(helper)
        with pytest.raises(CachePolicyError, match="consumption could not be recorded"):
            candidate.bind(claim)
        assert not helper.bound and not helper.admission.bound_reservations
        assert runtime.setup_admissions[claim.token] is helper
    assert helper.preclaim_sink.claimed.read_bytes() == original
    assert not helper.preclaim_sink.pending.exists() and not service.active
    assert helper.preclaim_sink.output.is_dir()


@pytest.mark.parametrize("mismatch", [None, "commit", "target_machine"])
def test_ordinary_setup_service_checks_exact_task_and_configuration_before_claim(tmp_path, monkeypatch, mismatch):
    runtime, task, config, owner, events, _service = _runtime(tmp_path, monkeypatch)
    profile = dict(format=setup.IMPORT_PROFILE, purpose="jobs", supported_workers=1,
        scope={key: task[key] for key in ("commit", "pipeline", "backend", "precision", "target_machine")})
    if mismatch:
        profile["scope"][mismatch] = "f" * 40 if mismatch == "commit" else "other-target"
    runtime.cache_adapter.lease.profile = profile
    pending = runtime.root / "queue" / (task["run"] + ".json")
    original = pending.read_bytes()
    calls = []
    require_applicable = setup.require_applicable

    def checked(actual_profile, actual_task, *, config):
        assert actual_profile is profile and actual_task == task
        calls.append(deepcopy(config))
        assert pending.read_bytes() == original and not events
        return require_applicable(actual_profile, actual_task, config=config)

    monkeypatch.setattr(setup, "require_applicable", checked)
    if mismatch:
        with pytest.raises(CachePolicyError, match="another source/target/pipeline"):
            admission.setup_claim_admission(runtime, task, config, owner)
    else:
        helper = admission.setup_claim_admission(runtime, task, config, owner)
        assert not helper.bound and not helper.admission.bound_reservations
    assert calls == [config] and pending.read_bytes() == original and not events


def test_setup_reserve_samples_remain_active_without_analysis_bound_mode_and_latch_original_stop(monkeypatch):
    now, sampled, scans, commands = [0.], [], [], []
    command = ["C:/fixture/python.exe", "-B", "-m", "darsia.fluidflower.worker",
               "--job", "C:/fixture/setup-job", "--workers", "1", "--once"]
    state = SimpleNamespace(setup_mode=True, bound_mode=False, reference_build=False,
        request=dict(cache_stat_period_seconds=30, public_command=command),
        task=dict(job_id="setup-job"), provider=dict(scope=dict(commit="c" * 40)),
        required_capacity_bound=dict(reserve_watchdog_period_seconds=1))
    lease = SimpleNamespace(policy=SimpleNamespace(root="C:/fixture/cache"),
        owner=dict(machine="fixture-cpu"), ledger=SimpleNamespace(locked=lambda **_kwargs: nullcontext()))
    lease.measure = lambda root: (scans.append(now[0]) or dict(logical_bytes=7))
    free = iter([PHYSICAL_RESERVE_BYTES + 10, PHYSICAL_RESERVE_BYTES, PHYSICAL_RESERVE_BYTES + 20])

    def volume(root):
        assert root == lease.policy.root
        sampled.append(now[0])
        return dict(free_bytes=next(free), volume_id="fixture-volume")

    previous = dict(path="C:/fixture/previous.json", sha256="a" * 64, size_bytes=1)
    monkeypatch.setattr("darsia_cache.capacity.volume_info", volume)
    monkeypatch.setattr(job_control, "read_json", lambda path: dict(commit="c" * 40))
    monkeypatch.setattr(job_control, "read_control", lambda root: dict(action="run", command=previous))

    def publish(root, action, **kwargs):
        commands.append((root, action, kwargs))
        return dict(path="C:/fixture/stop.json", sha256="b" * 64, size_bytes=1)

    monkeypatch.setattr(job_control, "publish_control", publish)
    collector = first.FiniteObservations(state, clock=lambda: now[0])
    for instant in (0., .25, 1., 2.):
        now[0] = instant
        collector.observe(SimpleNamespace(lease=lease), dict(active_processes=2))
    assert sampled == [0., 1., 2.] and scans == [0.]
    assert collector.native_poll_count == 4 and collector.reserve_crossed is True
    assert [row["at_or_below_operational_reserve"] for row in collector.reserve_samples] == [False, True, False]
    assert commands == [("C:/fixture/setup-job", "stop", dict(machine="fixture-cpu", expected_previous=previous))]
    assert collector.stop_response["state"] == "EXISTING_MASTER_STOP_COMMAND_PUBLISHED"


def test_ordinary_setup_supervisor_uses_original_native_reserve_observer_and_finalizer(monkeypatch):
    lease, command, events, stopped = bound_supervisor_TESTONLY(monkeypatch)
    lease.profile["format"] = setup.IMPORT_PROFILE
    assert owned_process.supervise_original_worker(lease) == 0
    assert events.count("one-original-start") == 1 and ("await", True) in events
    assert events.count("original-finalizer") == 1 and events[-1] == "close-original-handles"
    closures = [body for _, body, _ in lease.records
        if body.get("format") == "darsia-native-bound-operational-reserve-closure-v1"]
    assert len(closures) == 1 and closures[0]["actual_command"] == command
    assert closures[0]["operational_reserve_crossed"] is False and not stopped
    assert closures[0]["samples_prove_no_crossing_between_samples"] is False


def test_original_service_resolver_reopens_actual_positive_setup_type_without_analysis_fallback(service_world, monkeypatch):
    """Use the real Setup constructor with the original read-only context ledger."""
    from darsia_cache import service_context as resolver
    state, objects, _reads = protocol()
    world = service_world
    world.policy_pin = deepcopy(state.stores_source)
    world.policy_source = deepcopy(world.policy_pin)
    world.profile_pin = deepcopy(state.provider_binding)
    world.namespace_pin = deepcopy(state.provider["namespace_admission"])
    world.document["cache"]["profiles"] = {key: deepcopy(world.profile_pin)
        for key in world.document["cache"]["profiles"]}
    for role in world.document["cache_bootstrap"]["roles"].values():
        role["profile"] = deepcopy(world.profile_pin)
    world.document["cache_bootstrap"]["namespace_admission"] = deepcopy(world.namespace_pin)
    world.receipt.update(policy=world.policy_pin, profile=world.profile_pin, namespace=world.namespace_pin,
        first_qualification=dict(request=state.binding, provider=world.profile_pin,
            administrative_entitlement=dict(measured_profile=False, maximum_bytes=128)))
    request = objects[state.binding["path"]]
    request["maximum_service_bytes"] = 128
    objects[state.provider_binding["path"]]["output_namespace"] = deepcopy(world.admission["receipts"])
    objects[request["root_release"]["path"]]["request_payload_sha256"] = first._digest(
        {key: value for key, value in request.items() if key != "root_release"})
    bindings = [state.binding, state.provider_binding, *(value for value in request.values()
        if isinstance(value, dict) and set(value) == {"path", "sha256", "size_bytes"})]
    by_path = {bound["path"]: bound for bound in bindings}
    for path, document in objects.items():
        world.metadata[path_key(path)] = (deepcopy(by_path[path]), deepcopy(document))
    world.metadata[path_key(world.namespace_pin["path"])] = (world.namespace_pin, world.admission)
    monkeypatch.setattr(setup, "_original_metadata", lambda bound: world.read_bound(bound)[0])
    monkeypatch.setattr(first.FirstQualification, "__init__",
        lambda *args, **kwargs: pytest.fail("Setup entered the analysis qualification constructor"))
    actual = resolver.resolve_original_service_context(world.receipt_pin,
        package_file=PACKAGE, first_request=state.binding)
    assert actual.maximum == 128
    assert actual.profile["format"] == "darsia-cache-first-entry-contract-v1"
    assert actual.profile["module_closure"] == state.provider["module_closure"]
    assert actual.manifest["owner"] != actual.owner
    world.assert_no_mutation()
    world.assert_closed()


def test_original_service_resolver_refuses_direct_setup_import_profile_without_its_completed_provider(service_world):
    from darsia_cache import service_context as resolver
    world = service_world
    world.profile["format"] = setup.IMPORT_PROFILE
    with pytest.raises(CachePolicyError, match="complete original provider"):
        resolver.resolve_original_service_context(world.receipt_pin, package_file=PACKAGE)
    world.assert_no_mutation()
    world.assert_closed()


def test_ordinary_setup_context_uses_its_actual_cap_guard_after_provider_resolution(service_world, monkeypatch):
    """An isolated profile-reader seam cannot bypass the real Setup cap check."""
    from darsia_cache import service_context as resolver
    world, calls = service_world, []
    world.profile["format"] = setup.PROVIDER
    profile = dict(format=setup.IMPORT_PROFILE, purpose="jobs", entry_kind="worker",
        role_identifier="worker", provider=world.profile_pin, namespace_admission=world.namespace_pin,
        scope=dict(pipeline="series-setup"), supported_workers=1)

    def resolve_profile(binding, provider, kind, read, *, stores_source, **_kwargs):
        assert binding == world.profile_pin and provider == world.profile
        assert kind == "import" and stores_source == world.policy_pin
        calls.append("actual-setup-profile-reader")
        return deepcopy(profile)

    monkeypatch.setattr(setup, "resolve_profile", resolve_profile)
    monkeypatch.setattr(first, "checked_bound_import_context",
        lambda *args, **kwargs: pytest.fail("Setup entered the analysis import-context guard"))
    with pytest.raises(CachePolicyError, match="Setup service context differs"):
        resolver.resolve_original_service_context(world.receipt_pin, package_file=PACKAGE)
    assert calls == ["actual-setup-profile-reader"] and world.policy.volume_cap is None
    world.assert_no_mutation()
    world.assert_closed()


def test_fresh_setup_metadata_reader_keeps_numerical_and_gui_packages_unimported(tmp_path):
    """Run the real preimport reader in a fresh interpreter without site hooks."""
    path = tmp_path / "operation.json"
    raw = b'{"operation":"apply","private_contract_fixture":true}\n'
    path.write_bytes(raw)
    bound = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    script = """
import json
import sys
from darsia_cache import setup_qualification
actual = setup_qualification._original_metadata(json.loads(sys.argv[1]))
assert actual == {"operation": "apply", "private_contract_fixture": True}
forbidden = {"darsia", "numpy", "cv2", "matplotlib", "PySide6", "cupy"}
loaded = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
assert not loaded, loaded
print("preimport metadata read complete")
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run([sys.executable, "-B", "-S", "-c", script, json.dumps(bound)],
        env=environment, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "preimport metadata read complete"
