"""Synthetic control contracts only; no native service or science qualification."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import PureWindowsPath
from types import SimpleNamespace

import pytest

from darsia_cache import first_qualification as original
from darsia_cache import setup_qualification as setup
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from tests.unit.gui.test_cache_first_qualification import bound_plan_TESTONLY
from .test_series_setup_protocol import _task, _job, _operation


def pin(name, sha="a", size=17):
    return dict(path="C:/TESTONLY/" + name, sha256=sha * 64, size_bytes=size)


def provider():
    sources = [pin("src/" + suffix) for suffix in sorted(setup.REQUIRED_SOURCE_SUFFIXES)]
    runtime = [pin("python.exe", "b")]
    return dict(format=setup.PROVIDER, qualification_id="1" * 32,
        scope=dict(commit="1" * 40, pipeline="series-setup", backend="cpu", precision="float32",
                   target_machine="fixture", job=pin("jobs/fixture-job/job.json")),
        namespace_admission=pin("namespace.json"),
        output_namespace=dict(path="C:/TESTONLY/receipts", identity=dict(TESTONLY="synthetic namespace")),
        slots={role: "setup-" + role + ".json" for role in ("completion", "import_profile")},
        operations={operation: {role: operation + "-" + role + ".json" for role in setup.OPERATION_SLOTS}
                    for operation in setup.OPERATIONS},
        python_executable=runtime[0], startup=pin("startup.json"),
        module_closure=sources + runtime, producer_source_closure=sources, runtime_source_closure=runtime)


def protocol(mutate=None):
    """Use existing original Setup task/config fixtures and a synthetic cache bound."""
    root, binding, stores = provider(), pin("provider.json"), pin("stores.toml")
    task = _task("apply", run="apply-all")
    task.update(job_id="fixture-job", attempt_id="2" * 32)
    config, job = _operation("apply", run="apply-all"), _job(task)
    job["expected_runs"] = ["apply-all", "run-a", "run-b"]
    job["setup_operations"] = [dict(run=run, operation="apply" if index == 0 else "build")
                               for index, run in enumerate(job["expected_runs"])]
    bound = bound_plan_TESTONLY()
    bound["enforcement_state"] = "PREDECLARED_Q106_VOLUME_BOUND"
    bound["volume_cap_observation"].update(binding=dict(TESTONLY="synthetic cap", maximum_size_bytes=500 * 1024**3),
                                           virtual_size_bytes=500 * 1024**3)
    request = dict(format=setup.REQUEST, operation="apply", execution_authorized=True,
        root_release=pin("release.json"), provider=binding, stores_source=stores,
        namespace_admission=root["namespace_admission"],
        public_command=["C:/TESTONLY/python.exe", "-B", "-m", "darsia.fluidflower.worker",
            "--job", "C:/TESTONLY/jobs/fixture-job", "--backend", "cpu", "--workers", "1",
            "--machine", "fixture", "--stores", stores["path"], "--once"],
        task=pin("jobs/fixture-job/results/apply-all/task.json"), job=root["scope"]["job"], job_root="C:/TESTONLY/jobs/fixture-job",
        configuration=pin("operation.json"), capacity_bound_plan=pin("capacity.json"),
        maximum_service_bytes=1024, cache_stat_period_seconds=30, predecessor=None)
    objects = {binding["path"]: root, request["task"]["path"]: task, request["job"]["path"]: job,
        request["configuration"]["path"]: config, request["capacity_bound_plan"]["path"]: bound}
    if mutate:
        extra = mutate(root, request, task, job, config)
        if extra:
            objects.update(extra)
    objects[request["root_release"]["path"]] = dict(format=setup.ROOT_RELEASE,
        execution_authorized=True, provider=binding, stores_source=stores, scope=deepcopy(root["scope"]),
        request_payload_sha256=original._digest({key: value for key, value in request.items() if key != "root_release"}))
    request_pin = pin("request.json")
    objects[request_pin["path"]] = request
    reads = []

    def read(bound):
        reads.append(deepcopy(bound))
        return deepcopy(objects[bound["path"]])

    first = setup.PositiveSetupServiceQualification(request_pin, provider_binding=binding,
        stores_source=stores, namespace=root["namespace_admission"], read=read, read_metadata=read)
    return first, objects, reads


def test_setup_provider_is_distinct_and_preserves_original_analysis_restrictions():
    value = provider()
    before = deepcopy(value)
    assert setup.checked_provider(value) == value == before
    with pytest.raises(CachePolicyError):
        original.checked_provider(value)
    value["scope"]["pipeline"] = "native-phase-n1"
    with pytest.raises(CachePolicyError, match="CPU Setup"):
        setup.checked_provider(value)


def test_exact_apply_request_is_read_only_and_does_not_invent_analysis_or_retained_data():
    first, objects, reads = protocol()
    assert first.setup_mode and not first.bound_mode and not first.reference_build
    assert first.task["operation"] == "apply" and first.task["ordered_case_ids"] == ["apply"]
    assert first.claim is None and not first.started
    assert "retained_cache_inventory" not in first.request
    assert "job_profile" not in first.root_provider["slots"]
    assert all(bound["path"] in objects for bound in reads)
    assert not any("observation" in bound["path"] or "completion" in bound["path"] for bound in reads)


@pytest.mark.parametrize("change", [
    lambda root, request, task, job, config: request["public_command"].__setitem__(request["public_command"].index("--workers") + 1, "2"),
    lambda root, request, task, job, config: request["public_command"].remove("--once"),
    lambda root, request, task, job, config: request["public_command"].extend(["--job", request["job_root"]]),
    lambda root, request, task, job, config: task.update(commit="f" * 40),
    lambda root, request, task, job, config: task.update(target_machine="another-target"),
    lambda root, request, task, job, config: config["metadata_output"].update(maximum_output_bytes=0),
    lambda root, request, task, job, config: config["setup_plan"].update(sha256="f" * 64),
    lambda root, request, task, job, config: request.update(maximum_service_bytes=0),
    lambda root, request, task, job, config: task.update(resume_from={"synthetic": "analysis prefix"}),
    lambda root, request, task, job, config: request.update(predecessor=pin("future-build.json")),
])
def test_released_setup_request_still_refuses_wrong_compute_source_caps_or_future_evidence(change):
    with pytest.raises(CachePolicyError):
        protocol(change)


def test_consumption_keeps_retained_used_bytes_and_other_open_reservations_charged(monkeypatch):
    first, _objects, _reads = protocol()
    expected = first.required_capacity_bound["volume_cap_observation"]
    cap = SimpleNamespace(maximum_size_bytes=500 * 1024**3, document=lambda: deepcopy(expected["binding"]))
    events, records = [], {}

    class Ledger:
        @contextmanager
        def locked(self):
            events.append("locked")
            yield
            events.append("unlocked")

        def project(self, maximum):
            assert maximum == 0
            events.append("existing-accounting")
            return dict(hard_filesystem_quota_verified=True, volume_cap_observation=deepcopy(expected),
                current_cache_logical_bytes=123456, active_cache_remaining_bytes=789000,
                reserve_bytes=PHYSICAL_RESERVE_BYTES, quota_headroom_bytes=10**12,
                free_physical_bytes=10**12, required_noncache_bytes=0)

        @property
        def control(self):
            return self

        def names(self):
            return list(records)

        def write_once(self, name, raw):
            assert name not in records
            records[name] = json.loads(raw)

    class Namespace:
        def __init__(self, path, identity):
            assert dict(path=path, identity=identity) == first.provider["output_namespace"]

        def names(self):
            return []

        def close(self):
            pass

    from darsia_cache import windows
    monkeypatch.setattr(windows, "WindowsNamespace", Namespace)
    lease = SimpleNamespace(policy=SimpleNamespace(volume_cap=cap, quota_bytes=500 * 1024**3,
        profiles={"jobs": first.provider_binding}), ledger=Ledger(),
        admission=dict(receipts=deepcopy(first.provider["output_namespace"])),
        document=dict(cache_bootstrap=dict(namespace_admission=first.provider["namespace_admission"])),
        owner=dict(TESTONLY="synthetic owner"))
    first.consume(lease)
    assert first.started and first.claim is None
    assert events == ["locked", "existing-accounting", "unlocked"]
    assert list(records) == [first._launch_marker_name()]
    assert records[first._launch_marker_name()]["no_retry"] is True


def test_a_passing_boolean_cannot_close_without_actual_native_and_publication_evidence(monkeypatch):
    first, objects, _reads = protocol()
    native_pin = pin("native.json")
    native = dict(complete=True, passed=True, observer_native_handles_closed=True)
    objects[native_pin["path"]] = native
    written = []
    monkeypatch.setattr(setup, "_slot_read", lambda view, role: (
        (native, native_pin) if role == "native_completion" else ({}, pin("observation.json"))))
    monkeypatch.setattr(setup, "_slot_write", lambda *args: written.append(args))
    with pytest.raises(CachePolicyError, match="native completion"):
        setup.close_operation(first, native_pin, read=first._read)
    assert written == []


@pytest.mark.parametrize("field,value", [
    ("held_process_handle_observed_while_live", False),
    ("same_original_job_membership_observed", False),
    ("actual_service_receipt", pin("unrelated-service.json")),
])
def test_native_pass_flags_cannot_replace_original_primary_handle_and_membership(monkeypatch, field, value):
    first, objects, _reads = protocol()
    observation_pin, native_pin = pin("observation.json"), pin("native.json")
    native_launch, service = pin("launch.json"), pin("service.json")
    retirement_pin, primary_pin = pin("retirement.json"), pin("primary.json")
    terminal = dict(wrapper_exit_code=0, accounting=dict(active_processes=0, total_processes=3))
    primary = dict(pid=10, creation_filetime_100ns="123")
    objects[retirement_pin["path"]] = dict(format="darsia-native-owned-worker-retirement-v1",
        native_launch=native_launch, service_receipt=service, primary=primary,
        whole_family_kernel_observation=True, terminal=terminal, actual_primary_binding=primary_pin)
    objects[primary_pin["path"]] = dict(actual_service_receipt=service, original_native_launch=native_launch,
        actual_primary=primary, held_process_handle_observed_while_live=True,
        same_original_job_membership_observed=True)
    objects[primary_pin["path"]][field] = value
    observation = dict(actual_claim=first.task, source_runtime_before_after_exact=True,
        actual_public_worker_exit_code=0, operational_reserve_crossed=False,
        original_kernel_terminal=terminal, native_launch=native_launch, service_receipt=service)
    native = dict(format="darsia-native-worker-first-qualification-completion-v1",
        complete=True, passed=True, observer_native_handles_closed=True,
        request=first.binding, provider=first.provider_binding, stores_source=first.stores_source,
        scope=first.provider["scope"], startup=first.provider["startup"],
        module_closure=first.provider["module_closure"], observation=observation_pin,
        retirement=retirement_pin, maximum_family_processes=3)
    objects[native_pin["path"]] = native
    monkeypatch.setattr(setup, "_slot_read", lambda view, role: (observation, observation_pin))
    with pytest.raises(CachePolicyError, match="primary handle"):
        setup._operation_proof(first, native_pin, first._read)


def test_build_controls_join_only_the_original_post_apply_child(monkeypatch):
    previous_pin, result_pin = pin("apply-completion.json"), pin("apply-result.json")
    previous = dict(format=setup.CLOSED, result=result_pin)
    # Only the predecessor's native-proof reader is a synthetic seam here.
    # Native handle/retirement rejection is exercised separately above.
    monkeypatch.setattr(setup, "_slot_read", lambda view, role: (previous, previous_pin))
    monkeypatch.setattr(setup, "checked_operation", lambda value, **kwargs: value)

    def build(root, request, task, job, config):
        task.clear()
        task.update(_task("build", run="run-a"), job_id="fixture-job", attempt_id="3" * 32)
        config.clear()
        config.update(_operation("build", run="run-a"))
        request.update(operation="build", predecessor=previous_pin)
        request["task"] = pin("jobs/fixture-job/results/run-a/task.json")
        return {result_pin["path"]: dict(coordinator_completion=config["coordinator_completion"],
            children=[dict(run_id="run-a", configuration=request["configuration"],
                           raw_build_request=config["raw_build_request"])]), request["task"]["path"]: task}

    first, _objects, _reads = protocol(build)
    assert first.operation == "build" and first.request["predecessor"] == previous_pin

    def wrong_child(*args):
        extra = build(*args)
        extra[result_pin["path"]]["children"][0]["raw_build_request"] = pin("another-request.json")
        return extra

    with pytest.raises(CachePolicyError, match="post-Apply child"):
        protocol(wrong_child)


def test_ordinary_service_applicability_permits_further_setup_tasks_without_analysis_fallback():
    first, _objects, _reads = protocol()
    profile = dict(format=setup.IMPORT_PROFILE, purpose="jobs", scope=first.root_provider["scope"], supported_workers=1)
    task, config = _task("build", run="run-b"), _operation("build", run="run-b")
    assert setup.require_applicable(profile, task, config=config)
    for field, value in (("pipeline", "n1"), ("commit", "f" * 40), ("backend", "cuda"),
                         ("precision", "float64"), ("target_machine", "another-target")):
        with pytest.raises(CachePolicyError):
            setup.require_applicable(profile, dict(task, **{field: value}), config=config)


def test_setup_provider_never_resolves_as_a_measured_analysis_job_profile():
    with pytest.raises(CachePolicyError, match="never a measured analysis job"):
        setup.resolve_profile(pin("provider.json"), provider(), "jobs", lambda bound: pytest.fail("No profile slot may be read"),
                              stores_source=pin("stores.toml"))


def test_ordinary_public_controller_keeps_one_worker_without_requiring_once():
    first, _objects, _reads = protocol()
    profile = dict(format=setup.IMPORT_PROFILE, supported_workers=1)
    command = first.request["public_command"][:-1]
    assert "--once" not in command and setup.require_public_command(profile, command)
    command[command.index("--workers") + 1] = "2"
    with pytest.raises(CachePolicyError, match="one-worker"):
        setup.require_public_command(profile, command)


@pytest.mark.parametrize("workers", ["1", "2"])
def test_original_controller_applies_setup_concurrency_guard_before_launch(monkeypatch, workers):
    from darsia_cache import bootstrap, owned_process
    command = ["TESTONLY_python", "-B", "-m", "darsia.fluidflower.worker", "--workers", workers]
    monkeypatch.setattr(owned_process, "sys", SimpleNamespace(orig_argv=command,
        executable=command[0], flags=SimpleNamespace(dont_write_bytecode=True)))
    monkeypatch.setattr(bootstrap, "_python_module_invocation",
        lambda: ("darsia.fluidflower.worker", command[4:]))
    monkeypatch.setattr(owned_process, "_exact_source", lambda binding: None)
    lease = SimpleNamespace(profile=dict(format=setup.IMPORT_PROFILE, supported_workers=1,
        python_executable=dict(path=command[0], sha256="a" * 64, size_bytes=1)))
    if workers == "1":
        assert owned_process._public_command(lease) == command
    else:
        with pytest.raises(CachePolicyError, match="one-worker"):
            owned_process._public_command(lease)


def _closed_setup_chain(monkeypatch):
    """Run real metadata close APIs with explicitly synthetic kernel/service IO.

    Original ServiceLifecycle creates registration/seal/eligibility records, and
    OwnedWorker creates and verifies its retirement shape. No native process,
    files, resource reservation, producer or qualification is performed here.
    """
    from darsia_cache import bootstrap, owned_process
    from darsia_cache.environment import derive_cache_environment, first_qualification_environment
    from darsia_cache.service_lifecycle import ServiceLifecycle

    first, objects, _reads = protocol()
    slots, owner_exits = {}, []

    def store(path, value):
        raw = json.dumps(value, sort_keys=True).encode()
        bound = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        objects[bound["path"]] = deepcopy(value)
        return bound

    def slot_path(view, role):
        return str(PureWindowsPath(view["output_namespace"]["path"]) / view["slots"][role])

    def write_slot(view, role, value):
        path = slot_path(view, role)
        assert path not in slots
        bound = store(path, value)
        slots[path] = deepcopy(value), bound
        return deepcopy(bound)

    monkeypatch.setattr(setup, "_slot_write", write_slot)
    monkeypatch.setattr(setup, "_slot_read", lambda view, role: deepcopy(slots[slot_path(view, role)]))
    monkeypatch.setattr(owned_process, "_bound_control", lambda lease, bound: first._read(bound))
    observer = dict(pid=999, process_create_time="9.0", creation_filetime_100ns="999")
    monkeypatch.setattr(bootstrap, "_owner", lambda: (observer, {}))

    build_config = _operation("build", run="run-a")
    build_config_pin = store("C:/TESTONLY/build-operation.json", build_config)

    def close_native(state, payload):
        index = setup.OPERATIONS.index(state.operation)
        token = str(index + 4) * 32
        receipt_root = PureWindowsPath(state.provider["output_namespace"]["path"])
        owner = dict(pid=100 + index, process_create_time=str(index + 1) + ".0",
                     creation_filetime_100ns=str(index + 100), machine="fixture-cpu")
        launch = store(receipt_root / (token + "-launch.json"), dict(TESTONLY="synthetic native launch"))
        root = str(PureWindowsPath("C:/TESTONLY/cache/jobs") / token)
        reservation = dict(token=token)
        manifest = dict(format="darsia-cache-external-import-scope-v1", token=token, owner=owner,
            purpose="jobs", role_identifier="fixture_jobs", entry_kind="worker", root=root,
            root_identity=dict(TESTONLY="synthetic file identity"), reservation=reservation,
            maximum_bytes=state.maximum_service_bytes, native_launch_context=launch)
        service = dict(format="darsia-cache-import-service-receipt-v1", policy=state.stores_source,
            role_identifier="fixture_jobs", entry_kind="worker", namespace=state.provider["namespace_admission"],
            profile=state.provider_binding, manifest=manifest,
            first_qualification=dict(request=state.binding, provider=state.provider_binding))
        service_pin = store(receipt_root / (token + "-service.json"), service)
        ledger_rows = {}

        class Ledger:
            bound = dict(path=str(receipt_root))

            @property
            def control(self):
                return self

            @contextmanager
            def locked(self):
                yield

            def require_reservation(self, actual, actor, maximum):
                assert actual == reservation and actor == owner and maximum == state.maximum_service_bytes

            def names(self):
                return list(ledger_rows)

            def read_once(self, name):
                return ledger_rows[name], {}

            def write_once(self, name, raw):
                assert name not in ledger_rows
                ledger_rows[name] = raw
                store(receipt_root / name, json.loads(raw))

        lease = SimpleNamespace(reservation=reservation, source=state.stores_source,
            profile_binding=state.provider_binding, document=dict(cache_bootstrap=dict(
                namespace_admission=state.provider["namespace_admission"])), manifest=manifest,
            root=root, entry_kind="worker", role_identifier="fixture_jobs", owner=owner,
            maximum=state.maximum_service_bytes, ledger=Ledger(), entry_limit=100,
            admission=dict(receipts=state.provider["output_namespace"]),
            _record=lambda name, value: store(receipt_root / name, value))
        monkeypatch.setattr(owned_process, "current_native_membership", lambda: launch)
        lifecycle = ServiceLifecycle(lease)
        registration = lifecycle.register_owner()
        env = derive_cache_environment(dict(cache=dict(root="C:/TESTONLY/cache", quota_bytes=500 * 1024**3,
            profiles={"jobs": state.provider_binding})), purpose="jobs", token=token,
            scope_root=root, parent_environment={})
        env = first_qualification_environment(env, scope_root=root)
        modules = dict(request=state.binding, provider=state.provider_binding, actual_task=state.task,
            predeclared_module_closure_sha256=original._digest(state.provider["module_closure"]),
            all_file_backed_modules_predeclared_and_exact=True, undeclared=[],
            actual_writer_environment=env, writer_environment_is_actual_IO_or_lifetime_proof=False)
        exited = store(receipt_root / (token + "-exit.json"), dict(owner=owner, first_qualification_modules=modules))
        owner_exits.append(exited)

        class Kernel:
            """Explicit native OS observation seam; no real handles are supplied."""
            def signaled(self, handle):
                return True

            def identity(self, handle):
                return {key: owner[key] for key in ("pid", "creation_filetime_100ns")}

            def accounting(self, job):
                return dict(active_processes=0, total_processes=3)

            def require_job_flags(self, job):
                return {}

            def close(self, handle):
                pass

        controlled = owned_process.OwnedWorker(lease, api=Kernel())
        controlled.job, controlled.owner_handle = "synthetic-job", 3
        controlled.info = dict(process=1, thread=2)
        controlled.native_primary, controlled.observed_binding, controlled.service_binding = owner, launch, service_pin
        controlled.primary_binding = store(receipt_root / (token + "-primary.json"),
            dict(original_native_launch=launch, actual_service_receipt=service_pin, actual_primary=owner,
                 held_process_handle_observed_while_live=True, same_original_job_membership_observed=True))
        controlled.intent = dict(controller=observer, job_name="synthetic-job")
        terminal = dict(accounting=dict(active_processes=0, total_processes=3), wrapper_exit_code=0)
        retired = controlled.record_closed_family(terminal)

        receipt = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", complete=True,
            run=state.task["run"], operation=state.operation, backend="cpu", precision=state.task["precision"],
            setup_plan=state.configuration["setup_plan"], worker_references_closed=True, **payload)
        receipt_pin = store(receipt_root / (token + "-operation_result.json"), receipt)
        result_pin = store(receipt_root / (token + "-completed.json"), dict(receipt,
            files={"operation_result.json": receipt_pin}, task=state.task,
            attempt_id=state.task["attempt_id"], commit=state.task["commit"], actual_worker=owner))
        publication = dict(format="darsia-native-worker-service-publication-v1", job=state.request["job"],
            completed=[dict(run=state.task["run"], attempt_id=state.task["attempt_id"], completed=result_pin)],
            terminal_failures=[], original_done_and_product_index_metadata_verified=True)
        publication_pin = store(receipt_root / (token + "-publication.json"), publication)
        lifecycle.seal(logical_gate="synthetic-setup", outcome="success", utc="2026-01-01T00:00:00+00:00",
                       publication=publication_pin)
        eligibility = lifecycle.retention_eligibility(now_utc="2026-01-01T00:00:01+00:00", retirement=retired,
            verify_retirement=controlled.verify_retirement,
            verify_publication=lambda snapshot, bound: bound == publication_pin)
        observation = dict(request=state.binding, provider=state.provider_binding, stores_source=state.stores_source,
            scope=state.provider["scope"], actual_claim=state.task, native_launch=launch, service_receipt=service_pin,
            retirement=retired, original_kernel_terminal=terminal, actual_public_worker_exit_code=0,
            source_runtime_before_after_exact=True, operational_reserve_crossed=False,
            original_publication_and_retirement_eligibility=eligibility,
            conservative_family_process_bound_from_kernel_total=3,
            actual_registered_owner_exit_inventories=[dict(owner=owner, registration=registration,
                inventory=exited, all_file_backed_modules_predeclared_and_exact=True, undeclared=[])])
        observed_pin = write_slot(state.provider, "observation", observation)
        controlled.close()
        native = dict(format="darsia-native-worker-first-qualification-completion-v1", request=state.binding,
            provider=state.provider_binding, stores_source=state.stores_source, scope=state.provider["scope"],
            startup=state.provider["startup"], module_closure=state.provider["module_closure"],
            observation=observed_pin, retirement=retired, eligibility=eligibility,
            maximum_family_processes=3, observer_native_handles_closed=True, complete=True, passed=True)
        native_pin = write_slot(state.provider, "native_completion", native)
        return setup.close_operation(state, native_pin, read=state._read)

    applied = close_native(first, dict(Build_run=False, coordinator_completion=build_config["coordinator_completion"],
        children=[dict(run_id="run-a", configuration=build_config_pin, raw_build_request=build_config["raw_build_request"])]))
    task = _task("build", run="run-a")
    task.update(job_id="fixture-job", attempt_id="3" * 32)
    previous = task["configuration"]
    task["configuration"] = dict(previous, sha256=build_config_pin["sha256"], bytes=build_config_pin["size_bytes"])
    task["input_catalog"]["configuration"] = deepcopy(task["configuration"])
    for row in task["input_catalog"]["inputs"]:
        if row["reference"] == previous:
            row["reference"] = deepcopy(task["configuration"])
    task_pin = store("C:/TESTONLY/jobs/fixture-job/results/run-a/task.json", task)
    request = dict(first.request, operation="build", predecessor=applied, task=task_pin,
        configuration=build_config_pin, root_release=pin("build-release.json"))
    objects[request["root_release"]["path"]] = dict(format=setup.ROOT_RELEASE, execution_authorized=True,
        provider=first.provider_binding, stores_source=first.stores_source, scope=first.root_provider["scope"],
        request_payload_sha256=original._digest({key: value for key, value in request.items() if key != "root_release"}))
    build_request = store("C:/TESTONLY/build-request.json", request)
    build = setup.PositiveSetupServiceQualification(build_request, provider_binding=first.provider_binding,
        stores_source=first.stores_source, namespace=first.provider["namespace_admission"], read=first._read,
        read_metadata=first._read_metadata)
    built = close_native(build, dict(Build_run=True))
    release = store("C:/TESTONLY/ordinary-release.json", dict(format="darsia-setup-service-ordinary-release-v1",
        execution_authorized=True, provider=first.provider_binding, stores_source=first.stores_source,
        scope=first.root_provider["scope"], operations=dict(apply=applied, build=built)))
    setup.publish_import_profile(first.provider_binding, stores_source=first.stores_source,
                                 ordinary_release=release, read=first._read, read_metadata=first._read_metadata)
    return first, objects, slots, owner_exits


def test_original_shaped_apply_build_service_chain_closes_publishes_and_replays(monkeypatch):
    first, _objects, _slots, _exits = _closed_setup_chain(monkeypatch)
    profile = setup.resolve_profile(first.provider_binding, first.root_provider, "import", first._read,
                                    stores_source=first.stores_source, read_metadata=first._read_metadata)
    assert set(profile["operations"]) == {"apply", "build"}
    assert profile["supported_workers"] == 1 and profile["maximum_cache_bytes"] == first.maximum_service_bytes
    assert profile["actual_writer_IO_proven"] is False and profile["sampled_highwater_is_lifetime_peak"] is False
    assert setup.require_applicable(profile, _task("build", run="run-b"))


def test_final_profile_replay_refuses_an_owner_writer_location_outside_its_service(monkeypatch):
    first, objects, _slots, exits = _closed_setup_chain(monkeypatch)
    actual = objects[exits[0]["path"]]["first_qualification_modules"]["actual_writer_environment"]
    actual["TEMP"] = "C:/TESTONLY/outside-service/tmp"
    with pytest.raises(CachePolicyError, match="escapes its original service"):
        setup.resolve_profile(first.provider_binding, first.root_provider, "import", first._read,
                              stores_source=first.stores_source, read_metadata=first._read_metadata)


def test_shared_setup_controls_never_use_the_local_authority_reader():
    first, objects, _reads = protocol()
    request = deepcopy(first.request)
    root = deepcopy(first.root_provider)
    shared_root = PureWindowsPath("//fixture-share/jobs/fixture-job")
    job = dict(first.request["job"], path=str(shared_root / "job.json"))
    task = dict(request["task"], path=str(shared_root / "results" / first.task["run"] / "task.json"))
    config = dict(request["configuration"], path=str(shared_root / "controls" / "apply-operation.json"))
    root["scope"]["job"] = job
    request.update(job=job, task=task, configuration=config, job_root=str(shared_root))
    request["public_command"][request["public_command"].index("--job") + 1] = str(shared_root)
    objects.update({first.provider_binding["path"]: root, first.binding["path"]: request,
        job["path"]: first.job, task["path"]: first.task, config["path"]: first.configuration})
    objects[request["root_release"]["path"]] = dict(format=setup.ROOT_RELEASE, execution_authorized=True,
        provider=first.provider_binding, stores_source=first.stores_source, scope=root["scope"],
        request_payload_sha256=original._digest({key: value for key, value in request.items() if key != "root_release"}))
    metadata_reads = []

    def authority(bound):
        assert not PureWindowsPath(bound["path"]).drive.startswith("\\\\"), "Local authority must not open shared controls"
        return deepcopy(objects[bound["path"]])

    def metadata(bound):
        metadata_reads.append(bound)
        return deepcopy(objects[bound["path"]])

    state = setup.PositiveSetupServiceQualification(first.binding, provider_binding=first.provider_binding,
        stores_source=first.stores_source, namespace=root["namespace_admission"], read=authority, read_metadata=metadata)
    assert state.task == first.task and metadata_reads == [task, job, config]


def test_default_shared_metadata_seam_is_the_original_bound_worker_reader(monkeypatch):
    from darsia_cache import adapters
    bound = dict(pin("unused"), path="//fixture-share/jobs/fixture-job/job.json")
    observed = []

    def original_reader(path, *, expected):
        observed.append((path, expected))
        return {"TESTONLY": "original bounded metadata seam"}, expected

    monkeypatch.setattr(adapters, "_worker_metadata", original_reader)
    assert setup._original_metadata(bound) == {"TESTONLY": "original bounded metadata seam"}
    assert observed == [(bound["path"], bound)]
