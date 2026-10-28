"""UNRUN contract tests with administrative doubles, never runtime qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import preset_qualification as preset
from darsia_cache import first_qualification as first_module
from darsia_cache.policy import CachePolicyError


def _pin(label):
    return dict(path="C:/qualification/" + label + ".json",
                sha256=hashlib.sha256(label.encode()).hexdigest(), size_bytes=1)


@pytest.fixture
def released(monkeypatch):
    """Isolate exact request/control/cap selection from existing OS cap guards."""
    time = dict(order=0, time_hours=2.0, tolerance_hours=0.5)
    inputs = dict(requested_times=[time], runs=[2, 1], rois=["box"])
    input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    task = dict(format="darsia-alias-task-v1", pipeline="preset-wasserstein", backend="cpu", precision="original",
        commit="a" * 40, run="comparison-time-00000000", ordered_case_ids=["comparison-time-00000000"],
        attempt_id="1" * 32, threads=1, storage_profile="full", legacy_paths=False, execution_ready=True,
        job_id="comparison", comparison_time=time, expected_comparisons=1,
        preset_inputs_sha256=input_hash, configuration=dict(sha256="b" * 64, bytes=1))
    job = {key: deepcopy(task[key]) for key in ("format", "pipeline", "backend", "precision", "commit", "job_id", "preset_inputs_sha256")}
    job.update(format="darsia-alias-job-v1", preset_comparison_times=[time], comparisons_per_time=1)
    op = dict(format="darsia-preset-wasserstein-operation-v1", run=task["run"],
              comparison_time=time, preset_inputs=inputs)
    scope = {key: deepcopy(task[key]) for key in ("commit", "pipeline", "backend", "precision", "run",
        "preset_inputs_sha256", "comparison_time", "expected_comparisons")}
    scope.update(target_machine="logical-worker", physical_host="physical", operation_sha256="b" * 64, requested_time_count=1)
    namespace, stores, provider_pin, request_pin = map(_pin, ("namespace", "stores", "provider", "request"))
    provider = dict(format=preset.PROVIDER, scope=scope, namespace_admission=namespace,
        startup=_pin("startup"), module_closure=[], producer_source_closure=[], runtime_source_closure=[],
        output_namespace=dict(path="C:/receipts", identity={"accepted": True}),
        python_executable=dict(path="C:/runtime/python.exe", sha256="c" * 64, size_bytes=1),
        slots={name: name + ".json" for name in ("observation", "measurement", "native_completion", "completion",
                                               "bounded_profile", "bounded_completion")})
    request = dict(format=preset.REQUEST, execution_authorized=True, root_release=_pin("release"),
        provider=provider_pin, stores_source=stores, namespace_admission=namespace,
        public_command=["C:/runtime/python.exe", "-B", "-m", "darsia.fluidflower.worker", "--job", "C:/jobs/shared-job",
            "--backend", "cpu", "--workers", "1", "--machine", "logical-worker", "--stores", stores["path"], "--once", "--watchdog"],
        task=dict(_pin("task"), path="C:/jobs/shared-job/results/" + task["run"] + "/task.json"),
        job=dict(_pin("job"), path="C:/jobs/shared-job/job.json"), job_root="C:/jobs/shared-job",
        configuration=dict(_pin("config"), sha256="b" * 64), capacity_bound_plan=_pin("cap"),
        maximum_service_bytes=200, cache_stat_period_seconds=1, output_entitlement_plan=_pin("output"))
    output = dict(format=preset.OUTPUT, task=request["task"], scope=scope,
        maximum_output_bytes=100, maximum_transient_bytes=20,
        watched_output_scope=dict(format="darsia-preset-wasserstein-output-scope-v1", role="results",
            path="C:/results/comparison", identity={"accepted": True},
            stat_identity=dict(st_dev=1, st_ino=2, creation_ns=3), volume_id="volume",
            entry_limit=100, sample_period_seconds=1))
    docs = {provider_pin["path"]: provider, request_pin["path"]: request,
        namespace["path"]: dict(host="physical"),
        request["task"]["path"]: task, request["job"]["path"]: job,
        request["configuration"]["path"]: op, request["output_entitlement_plan"]["path"]: output,
        request["capacity_bound_plan"]["path"]: dict(enforcement_state="PREDECLARED_Q106_VOLUME_BOUND",
            volume_cap_observation=dict(binding=dict(volume="actual-fixed-volume"), virtual_size_bytes=1000))}
    # The full source union and physical Windows cap have their own boundaries.
    # This fixture does not pretend that its in-memory descriptors are real proof.
    monkeypatch.setattr(preset, "checked_provider", lambda value: deepcopy(value))
    monkeypatch.setattr(preset, "checked_required_capacity_bound", lambda value: deepcopy(value))

    def build():
        docs[request["root_release"]["path"]] = dict(format=preset.RELEASE, execution_authorized=True,
            provider=provider_pin, stores_source=stores, scope=scope,
            request_payload_sha256=first_module._digest({k: v for k, v in request.items() if k != "root_release"}))
        return preset.PresetQualification(request_pin, provider_binding=provider_pin, stores_source=stores,
            namespace=namespace, read=lambda pin: deepcopy(docs[pin["path"]]),
            read_metadata=lambda pin: deepcopy(docs[pin["path"]]))
    return SimpleNamespace(build=build, request=request, provider=provider, task=task, job=job,
        op=op, output=output, scope=scope, stores=stores, docs=docs)


def test_independent_job_id_and_folder_are_preserved(released):
    value = released.build()
    assert value.task["job_id"] == "comparison"
    assert value.request["job_root"].endswith("shared-job")
    assert value.preset_mode and not value.bound_mode and not value.reference_build


def test_logical_store_machine_can_differ_from_physical_host(released, monkeypatch):
    from darsia_cache.setup_qualification import PositiveSetupServiceQualification
    value = released.build()
    lease = SimpleNamespace(context=dict(host="physical"), document=dict(machine=dict(id="logical-worker")))
    sentinel = object()
    monkeypatch.setattr(PositiveSetupServiceQualification, "consume", lambda self, actual: sentinel)
    assert value.consume(lease) is sentinel
    lease.document["machine"]["id"] = "other-logical-worker"
    with pytest.raises(CachePolicyError, match="logical store machine"):
        value.consume(lease)


def test_physical_host_must_match_bound_namespace(released):
    released.scope["physical_host"] = "other"
    with pytest.raises(CachePolicyError, match="namespace admission"):
        released.build()


@pytest.mark.parametrize("key,value", [("maximum_output_bytes", 0), ("maximum_output_bytes", None),
                                      ("maximum_transient_bytes", -1), ("maximum_transient_bytes", None)])
def test_explicit_output_and_transient_entitlements_required(released, key, value):
    released.output[key] = value
    with pytest.raises(CachePolicyError, match="entitlement"):
        released.build()


def test_explicit_service_entitlement_required(released):
    released.request["maximum_service_bytes"] = 0
    with pytest.raises(CachePolicyError, match="service entitlement"):
        released.build()


@pytest.mark.parametrize("change", ["namespace", "commit", "config", "borrowed"])
def test_other_namespace_commit_configuration_or_native_request_refused(released, change):
    if change == "namespace":
        released.request["namespace_admission"] = _pin("other-namespace")
    elif change == "commit":
        released.task["commit"] = "d" * 40
    elif change == "config":
        released.task["configuration"]["sha256"] = "e" * 64
    else:
        released.request["format"] = "darsia-cache-first-qualification-request-v1"
    with pytest.raises(CachePolicyError):
        released.build()


def test_wrong_physical_machine_stops_before_launch_or_namespace_mutation(released):
    value = released.build()
    with pytest.raises(CachePolicyError, match="another physical machine"):
        value.consume(SimpleNamespace(context=dict(host="other")))


def test_actual_one_shot_markers_retain_owner_and_exact_task(released):
    value = released.build()
    records = {}
    class Ledger:
        control = None
        def _require_lock(self):
            return None
        def names(self):
            return list(records)
        def read_once(self, name):
            return records[name], None
        def write_once(self, name, raw):
            assert name not in records
            records[name] = raw
    ledger = Ledger()
    ledger.control = ledger
    lease = SimpleNamespace(ledger=ledger, owner={"pid": 11})
    value._remember_claim_locked(lease, value.task)
    value._remember_claim_locked(lease, value.task)
    assert len(records) == 2
    documents = [json.loads(raw) for raw in records.values()]
    assert any(row["format"] == "darsia-cache-first-qualification-consumed-v1" and row["no_retry"] for row in documents)
    assert any(row["format"] == "darsia-cache-first-single-claim-v1" and row["task"] == value.task for row in documents)
    lease.owner = {"pid": 12}
    with pytest.raises(CachePolicyError, match="different first claim/owner"):
        value._remember_claim_locked(lease, value.task)


def test_bounded_public_command_preserves_remote_default_and_finite_family(released):
    profile = dict(released.provider, stores_source=released.stores, qualified_watchdog=True)
    command = list(released.request["public_command"])
    preset.require_public_command(profile, command)
    index = command.index("--stores")
    del command[index:index + 2]
    original = list(command)
    preset.require_public_command(profile, command)
    assert command == original
    command.remove("--once")
    with pytest.raises(CachePolicyError):
        preset.require_public_command(profile, command)


def test_bounded_command_cannot_change_qualified_watchdog(released):
    profile = dict(released.provider, stores_source=released.stores, qualified_watchdog=True)
    command = [arg for arg in released.request["public_command"] if arg != "--watchdog"]
    with pytest.raises(CachePolicyError, match="supervision shape"):
        preset.require_public_command(profile, command)


def test_ordinary_fresh_scope_accepts_only_exact_preset_first_writer_provenance(tmp_path):
    from darsia_cache.fresh_scope import _fresh_provenance
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS, FIRST_WRITER_SUBDIRECTORIES
    token = "1" * 32
    parent = tmp_path / "jobs"
    root = parent / token
    reservation, owner, identity = dict(token=token), {"pid": 10}, {"id": 1}
    fresh = dict(format="darsia-fresh-exclusive-empty-service-scope-v1", token=token, owner=owner,
        purpose="jobs", root_identity=identity, creation_disposition="NtCreateFile.FILE_CREATE",
        exclusive_root_handle=True, initial_names=[], initial_empty_observed_utc="2026-10-23T12:00:00Z",
        observation_precedes_writer_subdirectories_and_imports=True,
        writer_subdirectories=list(FIRST_WRITER_SUBDIRECTORIES), writer_environment_keys=list(FIRST_WRITER_ENVIRONMENT_KEYS))
    context = SimpleNamespace(profile=dict(format=preset.BOUNDED_PROFILE), purpose="jobs", root=root,
        reservation=reservation, policy=SimpleNamespace(purpose=lambda value: parent),
        admission=dict(purposes=dict(jobs=dict(path=str(parent)))),
        manifest=dict(owner=owner, root_identity=identity, root=str(root), reservation=reservation, fresh_scope_provenance=fresh))
    assert _fresh_provenance(context) == fresh
    fresh["initial_names"] = ["unowned.bin"]
    with pytest.raises(CachePolicyError, match="fresh-exclusive-empty"):
        _fresh_provenance(context)


@pytest.fixture
def closed_execution(released, monkeypatch):
    """Field-join fixtures only: no Windows ownership, hashes or science proved."""
    from darsia_cache.environment import FIRST_WRITER_ENVIRONMENT_KEYS, first_qualification_environment
    first = released.build()
    docs, slots = released.docs, {}
    def put(label, document):
        pin = _pin(label)
        docs[pin["path"]] = document
        return pin
    def read(pin):
        return deepcopy(docs[pin["path"]])
    def write_slot(provider, name, value):
        assert provider == released.provider and name not in slots
        pin = put("slot-" + name, deepcopy(value))
        slots[name] = pin
        return pin
    monkeypatch.setattr(preset, "_slot_read", lambda provider, name: (read(slots[name]), slots[name]))
    monkeypatch.setattr(preset, "_slot_write", write_slot)
    owner = dict(machine="physical", pid=1, creation=2)
    launch, reservation = _pin("launch"), dict(token="2" * 32)
    manifest = dict(purpose="jobs", native_launch_context=launch, maximum_bytes=200,
                    owner=owner, reservation=reservation, root="C:/cache/jobs/service")
    service = dict(format="darsia-cache-import-service-receipt-v1", policy=first.stores_source,
        profile=first.provider_binding, entry_kind="worker", role_identifier="worker", manifest=manifest,
        namespace=first.provider["namespace_admission"],
        first_qualification=dict(request=first.binding, provider=first.provider_binding))
    service_pin = put("service", service)
    lifecycle = dict(reservation=reservation, policy=first.stores_source, profile=first.provider_binding,
                     namespace=first.provider["namespace_admission"], manifest=manifest)
    terminal = dict(wrapper_exit_code=0, accounting=dict(active_processes=0, total_processes=3))
    primary = dict(actual_service_receipt=service_pin, original_native_launch=launch, actual_primary=owner,
        held_process_handle_observed_while_live=True, same_original_job_membership_observed=True)
    retirement = dict(format="darsia-native-owned-worker-retirement-v1", native_launch=launch,
        service_receipt=service_pin, whole_family_kernel_observation=True, terminal=terminal,
        primary=owner, actual_primary_binding=put("primary", primary))
    retirement_pin = put("retirement", retirement)
    registration = dict(format="darsia-cache-service-lifecycle-v1", kind="owner", binding=lifecycle,
        actor=owner, value=dict(owner=owner, native_job_membership=launch))
    environment = {key: "C:/cache/jobs/service/location" for key in FIRST_WRITER_ENVIRONMENT_KEYS}
    environment.update(DARSIA_FLUIDFLOWER_CACHE=manifest["root"], NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator")
    environment = first_qualification_environment(environment, scope_root=manifest["root"])
    modules = dict(request=first.binding, provider=first.provider_binding,
        predeclared_module_closure_sha256=first_module._digest(first.provider["module_closure"]),
        all_file_backed_modules_predeclared_and_exact=True, undeclared=[], actual_writer_environment=environment,
        writer_environment_is_actual_IO_or_lifetime_proof=False, actual_task=first.task)
    exited = dict(owner=owner, first_qualification_modules=modules)
    result = dict(format="darsia-preset-wasserstein-result-v1", pipeline="preset-wasserstein",
        complete=True, worker_references_closed=True, task=first.task, attempt_id=first.task["attempt_id"],
        **{key: first.task[key] for key in ("commit", "run", "job_id", "backend", "precision",
                                         "preset_inputs_sha256", "comparison_time")},
        expected_count=1, comparison_count=1, comparisons=[], files={}, output_count=0, vtk={},
        nonfinite_distances=1, nonfinite_normalized_distances=1, result_statuses={"missing_data": 1}, validation={})
    receipt = {key: deepcopy(value) for key, value in result.items() if key not in
               {"format", "task", "attempt_id", "commit"}}
    receipt["format"] = "darsia-preset-wasserstein-execution-receipt-v1"
    result["execution_receipt"] = put("execution-receipt", receipt)
    publication = dict(format="darsia-native-worker-service-publication-v1",
        job=first.request["job"], original_done_and_product_index_metadata_verified=True,
        terminal_failures=[], completed=[dict(run=first.task["run"], attempt_id=first.task["attempt_id"],
                                             completed=put("result", result))])
    seal = dict(format="darsia-cache-service-lifecycle-v1", kind="seal", binding=lifecycle, actor=owner,
                value=dict(outcome="success", publication=put("publication", publication)))
    eligibility = dict(original_retirement_verifier_accepted=True, seal=put("seal", seal))
    measured = dict(format="darsia-preset-wasserstein-closed-output-diagnostic-v1", request=first.binding,
        provider=first.provider_binding, stores_source=first.stores_source, scope=first.provider["scope"],
        output_plan=first.request["output_entitlement_plan"], output_scope=first.output_entitlement["watched_output_scope"],
        entitlement_bytes=120, actual_retirement=retirement_pin, actual_terminal=terminal,
        original_retirement_accepted=True, closed_output_stat_observation=True, complete=True,
        first_stop=None, first_observation_error=None, whole_output_entitlement_observed_exceeded=False,
        cache_operational_reserve_crossed=False, sampled_highwater_is_lifetime_peak=False,
        output_hard_quota=False, closed_output_logical_bytes=15, sampled_highwater_bytes=20)
    observation = dict(format="darsia-cache-first-qualification-observation-v1", request=first.binding,
        provider=first.provider_binding, stores_source=first.stores_source, scope=first.provider["scope"],
        actual_claim=first.task, source_runtime_before_after_exact=True, actual_public_worker_exit_code=0,
        operational_reserve_crossed=False, original_publication_and_retirement_eligibility=eligibility,
        retirement=retirement_pin, original_kernel_terminal=terminal, native_launch=launch, service_receipt=service_pin,
        actual_registered_owner_exit_inventories=[dict(owner=owner, registration=put("registration", registration),
            inventory=put("inventory", exited), all_file_backed_modules_predeclared_and_exact=True, undeclared=[])],
        conservative_family_process_bound_from_kernel_total=3, sampled_cache_logical_high_water_bytes=12,
        output_measurement=write_slot(first.provider, "measurement", measured))
    observation_pin = write_slot(first.provider, "observation", observation)
    native = dict(format="darsia-native-worker-first-qualification-completion-v1", complete=True, passed=True,
        observer_native_handles_closed=True, request=first.binding, provider=first.provider_binding,
        stores_source=first.stores_source, scope=first.provider["scope"], startup=first.provider["startup"],
        module_closure=first.provider["module_closure"], observation=observation_pin,
        retirement=retirement_pin, maximum_family_processes=3, eligibility=eligibility)
    native_pin = write_slot(first.provider, "native_completion", native)
    return SimpleNamespace(first=first, read=read, put=put, slots=slots, docs=docs, released=released,
        native_pin=native_pin, proof=lambda: preset._operation_proof(first, native_pin, read=read))


def test_original_closed_execution_preserves_nonfinite_status_without_science_pass(closed_execution):
    proof = closed_execution.proof()
    assert proof["complete"] and proof["execution_closed"]
    assert proof["numerical_acceptance_claimed"] is False
    assert proof["actual_writer_IO_proven"] is False
    assert proof["cache_reservation_closed"] is False
    assert "passed" not in proof


@pytest.mark.parametrize("document,key,value", [
    ("slot-native_completion", "complete", False), ("slot-native_completion", "startup", _pin("other")),
    ("slot-native_completion", "module_closure", [_pin("other")]),
    ("slot-native_completion", "retirement", _pin("other")),
    ("slot-observation", "retirement", _pin("other")),
    ("slot-observation", "output_measurement", _pin("other")),
    ("slot-observation", "conservative_family_process_bound_from_kernel_total", 4),
    ("retirement", "whole_family_kernel_observation", False),
    ("primary", "same_original_job_membership_observed", False),
    ("service", "policy", _pin("other")), ("registration", "binding", {}),
    ("inventory", "first_qualification_modules", {}), ("seal", "binding", {}),
    ("publication", "job", _pin("other")), ("execution-receipt", "comparison_count", 2),
    ("slot-measurement", "request", _pin("other")), ("slot-measurement", "output_plan", _pin("other")),
    ("slot-measurement", "entitlement_bytes", 121), ("slot-measurement", "actual_retirement", _pin("other")),
    ("slot-measurement", "first_observation_error", "lost sample"),
])
def test_original_proof_rejects_foreign_or_incomplete_lifecycle(closed_execution, document, key, value):
    closed_execution.docs[_pin(document)["path"]][key] = value
    with pytest.raises((CachePolicyError, KeyError)):
        closed_execution.proof()


def test_fixed_slot_bounded_replay_requires_original_dag_and_independent_release(closed_execution, monkeypatch):
    fixture = closed_execution
    first = fixture.first
    closed_pin = preset.close_operation(first, fixture.native_pin, read=fixture.read)
    release = fixture.put("ordinary-release", dict(format="darsia-preset-wasserstein-bounded-release-v1",
        execution_authorized=True, provider=first.provider_binding, stores_source=first.stores_source,
        scope=first.provider["scope"], first_closed=closed_pin,
        maximum_service_bytes=200, maximum_output_bytes=100, maximum_transient_bytes=20))
    preset.publish_bounded_profile(first.provider_binding, stores_source=first.stores_source,
        ordinary_release=release, read=fixture.read, read_metadata=fixture.read)
    profile = preset.resolve_bounded_profile(first.provider_binding, first.provider, "jobs", fixture.read,
        stores_source=first.stores_source, read_metadata=fixture.read)
    assert profile["scope"]["target_machine"] == "logical-worker"
    assert profile["scope"]["physical_host"] == "physical"
    assert profile["native_supervision"]["maximum_family_processes"] == 3
    assert profile["not_a_measured_profile"] is True
    assert profile["actual_writer_IO_proven"] is False
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "physical")
    cap = SimpleNamespace(maximum_size_bytes=1000, document=lambda: dict(volume="actual-fixed-volume"))
    policy = SimpleNamespace(profiles=dict(jobs=first.provider_binding), volume_cap=cap, quota_bytes=1000)
    context = dict(purpose="jobs", role_identifier="worker", entry_kind="worker", policy=policy,
        namespace=first.provider["namespace_admission"], machine_id="logical-worker")
    assert preset.checked_import_context(profile, **context) == 200
    with pytest.raises(CachePolicyError, match="physical scope"):
        preset.checked_import_context(profile, **dict(context, machine_id="other-logical-worker"))
    monkeypatch.setattr(socket, "gethostname", lambda: "another-physical")
    with pytest.raises(CachePolicyError, match="physical scope"):
        preset.checked_import_context(profile, **context)
    monkeypatch.setattr(socket, "gethostname", lambda: "physical")
    with pytest.raises(CachePolicyError, match="physical scope"):
        preset.checked_import_context(profile, **dict(context, namespace=_pin("other-namespace")))
    cap.maximum_size_bytes = 1001
    with pytest.raises(CachePolicyError, match="physical scope"):
        preset.checked_import_context(profile, **context)
    cap.maximum_size_bytes = 1000
    for document, key, value in [("ordinary-release", "maximum_service_bytes", 201),
                                  ("slot-completion", "cache_reservation_closed", True),
                                  ("slot-native_completion", "passed", False),
                                  ("publication", "terminal_failures", ["failed"]),
                                  ("slot-bounded_profile", "maximum_cache_bytes", 201)]:
        actual = fixture.docs[_pin(document)["path"]]
        prior = actual[key]
        actual[key] = value
        with pytest.raises(CachePolicyError):
            preset.resolve_bounded_profile(first.provider_binding, first.provider, "jobs", fixture.read,
                stores_source=first.stores_source, read_metadata=fixture.read)
        actual[key] = prior


def test_adapter_refuses_changed_actual_admitted_service(released, monkeypatch):
    """Isolate adapter joins; full closure/IO qualification is not simulated."""
    from darsia_cache.worker_adapter import WorkerCacheAdapter
    from darsia.fluidflower import preset_wasserstein_capacity as capacity
    first = released.build()
    producer, runtime_pin = _pin("producer-source"), _pin("runtime-source")
    profile = dict(scope=first.provider["scope"], stores_source=first.stores_source,
        maximum_cache_bytes=200, producer_source_closure=[producer], runtime_source_closure=[runtime_pin],
        module_closure=[producer, runtime_pin])
    lease = SimpleNamespace(profile=profile, profile_binding=first.provider_binding, source=first.stores_source,
        purpose="jobs", role_identifier="worker", entry_kind="worker", policy=object(), maximum=200,
        context=dict(host="physical"), document=dict(machine=dict(id="logical-worker"),
            cache_bootstrap=dict(namespace_admission=first.provider["namespace_admission"])))
    adapter = object.__new__(WorkerCacheAdapter)
    adapter.lease, adapter.profiles = lease, {}
    adapter.runtime = SimpleNamespace(backend="cpu", commit=first.task["commit"], repository="C:/checkout",
        stores=SimpleNamespace(machine_id="logical-worker"))
    monkeypatch.setattr(capacity, "require_bounded_estimate", lambda *args: True)
    monkeypatch.setattr(capacity, "require_service_sources", lambda *args: True)
    monkeypatch.setattr(preset, "checked_import_context", lambda *args, **kwargs: 200)
    measured = dict(complete=True, first_stop=None, output_hard_quota=False, sampled_highwater_is_lifetime_peak=False)
    complete = dict(format=preset.BOUNDED_COMPLETION, complete=True, passed=True, not_a_measured_profile=True)
    def verify():
        return adapter._verify_preset_bounded(first.task, object(), first.provider_binding,
            profile, measured, complete, remember=True)
    assert verify()
    for attribute, bad in [("maximum", 201), ("source", _pin("another-map")),
                            ("context", dict(host="other-physical")),
                            ("profile_binding", _pin("other-provider"))]:
        prior = getattr(lease, attribute)
        setattr(lease, attribute, bad)
        with pytest.raises(CachePolicyError, match="actual admitted physical service"):
            verify()
        setattr(lease, attribute, prior)
    adapter.runtime.stores.machine_id = "other-logical-worker"
    with pytest.raises(CachePolicyError, match="actual admitted physical service"):
        verify()
