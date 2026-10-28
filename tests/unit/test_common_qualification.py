"""Synthetic metadata contracts only; no host admission or passing qualification."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import common_qualification as common
from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError, PHYSICAL_RESERVE_BYTES
from tests.unit.gui import test_cache_first_qualification as old


pin = old.pin


def provider():
    """Declare distinct synthetic slots and the full generic provider closure."""
    sources = [pin("src/" + suffix) for suffix in sorted(common.REQUIRED_SOURCE_SUFFIXES)]
    runtime = [pin("python.exe", "b")]
    cases = ["sample_f" + str(i) for i in range(3)]
    return dict(format=common.PROVIDER, qualification_id="1" * 32,
        scope=dict(commit="c" * 40, backend="cpu", precision="float32", run="sample", ordered_case_ids=cases),
        namespace_admission=pin("namespace.json"),
        output_namespace=dict(path="C:/TESTONLY/receipts", identity=dict(TESTONLY="not host admission")),
        slots={role: "common-" + role + ".json" for role in ("completion", "import_profile", "job_profile")},
        stages={stage: dict(pipeline=common.PIPELINES[stage], run="refit" if stage == "calibration" else "sample",
            ordered_case_ids=["fit_f1", "fit_f2"] if stage == "calibration" else deepcopy(cases),
            storage_profile=common.PROFILES[stage], colour_intermediates="keep",
            slots={role: stage + "-" + role + ".json" for role in common.STAGE_SLOTS}) for stage in common.STAGES},
        python_executable=runtime[0], startup=pin("startup.json"), device_identity=pin("device.json"),
        producer_source_closure=sources, runtime_source_closure=runtime, module_closure=sources + runtime)


def source_review(root):
    """Synthetic reviewed text content, independent from the request payload."""
    text = "Synthetic completed full source review."
    return dict(format="darsia-common-source-review-v1", verdict="PASS", base_commit=root["scope"]["commit"],
        tree="d" * 40, module_closure_sha256=first._digest(root["module_closure"]),
        producer_source_closure=deepcopy(root["producer_source_closure"]),
        answer=dict(id="review-entry", text=text, sha256=hashlib.sha256(text.encode()).hexdigest()))


def protocol(stage="phase1", mutate=None, *, allow_predecessors=False, monkeypatch=None):
    """Bind a synthetic original task, authority and explicitly released predecessor scope."""
    root, binding, stores = provider(), pin("common-provider.json"), pin("stores.toml")
    row = root["stages"][stage]
    task = dict(common.stage_provider(root, stage)["scope"], attempt_id="2" * 32,
        job_id="synthetic-" + stage, format_version=2 if stage == "phase2" else 1,
        storage_profile=row["storage_profile"], configuration=dict(root_alias="TESTONLY",
            relative_path="config.json", sha256="a" * 64, bytes=1))
    if stage == "phase1":
        task.pop("storage_profile")  # The existing phase-one schema carries no N1 storage field.
    bound = old.bound_plan_TESTONLY()
    bound["enforcement_state"] = "PREDECLARED_Q106_VOLUME_BOUND"
    bound["volume_cap_observation"].update(binding=dict(TESTONLY="synthetic cap", maximum_size_bytes=500 * 1024**3), virtual_size_bytes=500 * 1024**3)
    watched = dict(format=common.OUTPUT_SCOPE, role="results", path="C:/TESTONLY/results/" + stage,
        identity=dict(TESTONLY="not a file ID"), stat_identity=dict(st_dev=1, st_ino=2, creation_ns=3),
        volume_id="TESTONLY", entry_limit=1000, sample_period_seconds=1)
    request = dict(format=common.REQUEST, stage=stage, execution_authorized=True,
        root_release=pin(stage + "-release.json"), provider=binding,
        stores_source=stores, namespace_admission=root["namespace_admission"],
        public_command=["C:/TESTONLY/python.exe", "-B", "-m", "darsia.fluidflower.worker",
            "--job", "C:/TESTONLY/jobs/" + task["job_id"], "--stores", stores["path"]],
        task=pin(stage + "-task.json"), job=pin("jobs/" + task["job_id"] + "/job.json"),
        job_root="C:/TESTONLY/jobs/" + task["job_id"], configuration=pin("config.json"),
        capacity_bound_plan=pin("bound.json"), output_entitlement_plan=pin(stage + "-output.json"),
        retained_cache_inventory=None if stage == "phase1" else pin("retained.json"),
        cache_stat_period_seconds=30,
        predecessors={name: pin(name + "-completion.json") for name in common.STAGES[:common.STAGES.index(stage)]})
    device = dict(format="darsia-common-cpu-runtime-binding-v1", backend="cpu", mass_device_id="cpu",
        python_executable=root["python_executable"], namespace_admission=root["namespace_admission"],
        module_closure_sha256=first._digest(root["module_closure"]), driver_sources=root["runtime_source_closure"],
        device_environment={key: os.environ.get(key) for key in common.DEVICE_ENVIRONMENT},
        opencl_compute_qualified=False, opencl_drift_policy="existing unchanged drift route; no forced-off setting")
    objects = {binding["path"]: root, root["device_identity"]["path"]: device,
        request["task"]["path"]: task, request["job"]["path"]: dict(commit=task["commit"], precision="float32", expected_runs=[task["run"]]),
        request["capacity_bound_plan"]["path"]: bound,
        request["output_entitlement_plan"]["path"]: dict(format=common.OUTPUT, stage=stage,
            task=request["task"], request_scope=binding, maximum_output_bytes=1000,
            maximum_service_bytes=100, watched_output_scope=watched)}
    if mutate:
        mutate(root, request, task, device, objects)
    objects[request["root_release"]["path"]] = dict(format=common.ROOT_RELEASE,
        execution_authorized=True, provider=binding, stores_source=stores, scope=deepcopy(root["scope"]),
        source_review=pin("source-review.json"), request_payload_sha256=first._digest({key: value for key, value in request.items() if key != "root_release"}))
    objects["C:/TESTONLY/source-review.json"] = source_review(root)
    request_pin = pin(stage + "-request.json")
    objects[request_pin["path"]] = request
    reads = []
    def read(bound):
        reads.append(deepcopy(bound))
        return deepcopy(objects[bound["path"]])
    authorities = lambda bound: pytest.fail("Common code must not read a handoff log")
    if allow_predecessors:
        assert monkeypatch is not None
        # This fixture tests request/finite predecessor admission, not physical
        # qualification. Full stage evidence checks have separate cases below.
        monkeypatch.setattr(common, "_slot_read", lambda view, role:
            (dict(stage=view["scope"]["pipeline"]), request["predecessors"][next(
                key for key in common.STAGES if common.PIPELINES[key] == view["scope"]["pipeline"])]))
        monkeypatch.setattr(common, "checked_stage", lambda value, **kw: value)
    state = common.CommonQualification(request_pin, provider_binding=binding, stores_source=stores,
        namespace=root["namespace_admission"], read=read, read_authorities=authorities)
    return state, objects, reads


def test_common_root_is_acyclic_and_default_provider_does_not_accept_it():
    """Common root is acyclic and default provider does not accept it."""
    root = provider()
    assert common.checked_provider(root) == root
    assert not any(key in root for key in ("profiles", "passed", "stores_source", "request"))
    assert all(not any(key in row for key in ("task", "completion", "fit", "configuration"))
               for row in root["stages"].values())
    with pytest.raises(CachePolicyError):
        first.checked_provider(root)
    assert first.checked_provider(old.provider()) == old.provider()


@pytest.mark.parametrize("mutate", [
    lambda p: p["scope"].update(run="different"),
    lambda p: p["scope"].update(backend="cuda"),
    lambda p: p["scope"].update(ordered_case_ids=["same"] * 39),
    lambda p: p["stages"]["phase2"].update(storage_profile="full"),
    lambda p: p["stages"]["phase2"].update(colour_intermediates="drop"),
    lambda p: p["stages"]["calibration"].update(pipeline="native-phase-n1"),
    lambda p: p["slots"].update(completion="../other.json"),
    lambda p: p["slots"].update(completion=p["slots"]["job_profile"]),
    lambda p: p["producer_source_closure"].pop(),
    lambda p: p.update(passed=True),
])
def test_common_scope_never_converts_old_full_or_unknown_future_slots(mutate):
    """Common scope never converts old full or unknown future slots."""
    value = provider()
    mutate(value)
    with pytest.raises(CachePolicyError):
        common.checked_provider(value)


def test_root_released_first_stage_is_read_only_and_one_exact_original_task():
    """Root released first stage is read only and one exact original task."""
    state, objects, _ = protocol()
    before = deepcopy(objects)
    lease = SimpleNamespace(first_qualification=state, source=state.stores_source)
    assert state.require_task(lease, state.task) is True
    assert objects == before
    assert state.started is False and state.claim is None
    with pytest.raises(CachePolicyError):
        state.require_task(lease, dict(state.task, attempt_id="3" * 32))
    assert state._once_key() != old.protocol()[0]._once_key()


@pytest.mark.parametrize("mutate", [
    lambda p,r,t,d,o: t.update(storage_profile="lean"),
    lambda p,r,t,d,o: t.update(resume_from={"not": "a supported prefix"}),
    lambda p,r,t,d,o: t.update(commit="e" * 40),
    lambda p,r,t,d,o: t.update(format_version=2),
    lambda p,r,t,d,o: r.update(configuration=pin("other-config.json", "e")),
    lambda p,r,t,d,o: r["public_command"].__setitem__(-3, "C:/TESTONLY/other-job"),
    lambda p,r,t,d,o: d.update(mass_device_id="unqualified-gpu"),
    lambda p,r,t,d,o: d.update(opencl_compute_qualified=True),
    lambda p,r,t,d,o: d.update(module_closure_sha256="f" * 64),
    lambda p,r,t,d,o: d["device_environment"].update(DARSIA_CUDA_DEVICE="changed"),
    lambda p,r,t,d,o: o[r["output_entitlement_plan"]["path"]].update(maximum_output_bytes=0),
])
def test_released_common_task_and_device_cannot_drift(mutate):
    """Released common task and device cannot drift."""
    with pytest.raises(CachePolicyError):
        protocol(mutate=mutate)


def test_root_release_payload_change_is_not_an_authorization():
    """Root release payload change is not an authorization."""
    state, objects, _ = protocol()
    objects[state.binding["path"]]["cache_stat_period_seconds"] = 29
    with pytest.raises(CachePolicyError, match="independent release"):
        common.CommonQualification(state.binding, provider_binding=state.provider_binding,
            stores_source=state.stores_source, namespace=state.provider["namespace_admission"],
            read=lambda p: deepcopy(objects[p["path"]]),
            read_authorities=lambda p: pytest.fail("No handoff read"))


def test_cannot_admit_calibration_before_actual_p1_completion(monkeypatch):
    """Cannot admit calibration before actual p1 completion."""
    monkeypatch.setattr(common, "_slot_read", lambda *args: (_ for _ in ()).throw(FileNotFoundError("not completed")))
    with pytest.raises(FileNotFoundError, match="not completed"):
        protocol("calibration")


def test_each_continuation_reads_only_closed_predecessors_not_future_outputs(monkeypatch):
    """Each continuation reads only closed predecessors not future outputs."""
    state, _, _ = protocol("calibration", allow_predecessors=True, monkeypatch=monkeypatch)
    assert list(state.request["predecessors"]) == ["phase1"]
    assert state.provider["scope"]["pipeline"] == "n1-calibration-refit"
    assert state.root_provider["stages"]["phase2"]["storage_profile"] == "lean"


def test_common_whole_stage_entitlement_does_not_shrink_as_frames_finish():
    """Common whole stage entitlement does not shrink as frames finish."""
    state, _, _ = protocol()
    lease = SimpleNamespace(first_qualification=state, source=state.stores_source)
    runtime = SimpleNamespace(cache_adapter=SimpleNamespace(lease=lease))
    config = dict(run="sample", frames=[dict(case_id=case) for case in state.task["ordered_case_ids"]])
    estimate = common.common_output_entitlement(runtime, state.task, config, [])
    assert estimate.transient_estimated_bytes == 1000
    assert estimate.products_estimated_bytes == estimate.products_per_frame_estimated_bytes == 0
    assert estimate.needed(other_remaining_bytes=7, reserve_bytes=11) == 1018
    assert estimate.needed(other_remaining_bytes=7) == 1007 + PHYSICAL_RESERVE_BYTES
    config["frames"].reverse()
    with pytest.raises(CachePolicyError, match="order"):
        common.common_output_entitlement(runtime, state.task, config, [])


def test_common_output_scope_does_not_weaken_old_first_output_minimum():
    """Common output scope does not weaken old first output minimum."""
    state, _, _ = protocol()
    watch = first.BoundOutputObservations(state)
    assert watch.entitlement_bytes == 1000 and watch.common is True
    scope = state.output_entitlement["watched_output_scope"]
    with pytest.raises(CachePolicyError):
        first.checked_bound_output_scope(scope)
    assert first.checked_bound_output_scope(scope, common=True) == scope


def test_actual_claim_uses_original_atomic_consumption_and_cannot_leave_released_output(monkeypatch):
    """Actual claim uses original atomic consumption and cannot leave released output."""
    state, _, _ = protocol()
    ledger = old.Ledger_TESTONLY()
    lease = SimpleNamespace(first_qualification=state, source=state.stores_source,
        owner=dict(pid=12, process_create_time="123.0"), ledger=ledger)
    monkeypatch.setattr(state, "_require_launch_admission", lambda lease: None)
    claim = SimpleNamespace(task=state.task, token=state.task["attempt_id"],
        output=state.output_entitlement["watched_output_scope"]["path"] + "/job/attempt")
    with ledger.locked():
        state.remember_claim_locked(lease, claim)
    assert state.claim == state.task and len(ledger.documents) == 2
    with ledger.locked():
        state.remember_claim_locked(lease, claim)
    assert len(ledger.documents) == 2
    with pytest.raises(CachePolicyError, match="leaves"):
        state.remember_claim_locked(lease, SimpleNamespace(task=state.task, token=claim.token,
            output="C:/TESTONLY/results/another-stage"))


def test_incomplete_common_provider_and_direct_profile_have_no_ordinary_fallback(monkeypatch):
    """Incomplete common provider and direct profile have no ordinary fallback."""
    state, objects, _ = protocol()
    def absent(*args):
        raise CachePolicyError("common completion absent")
    monkeypatch.setattr(common, "_slot_read", absent)
    read = lambda p: deepcopy(objects[p["path"]])
    with pytest.raises(CachePolicyError, match="completion absent"):
        first.resolve_profile(state.provider_binding, "jobs", read, stores_source=state.stores_source,
            read_authorities=state._read_authorities)
    for kind in (common.IMPORT_PROFILE, common.JOB_PROFILE):
        with pytest.raises(CachePolicyError, match="complete original provider DAG"):
            first.resolve_profile(pin("direct.json"), "jobs", lambda p: dict(format=kind))


def test_generic_production_application_requires_exact_source_and_lean_keep():
    """Generic production application requires exact source and lean keep."""
    profile = dict(format=common.JOB_PROFILE, purpose="jobs", scope=provider()["scope"],
        stage_routes={stage: {key: row[key] for key in ("pipeline", "storage_profile", "colour_intermediates")}
                      for stage, row in provider()["stages"].items()},
        compute_profiles={"phase2": dict(control=dict(format="darsia-opencl-n1-v1", mode="auto", device_id=None))})
    task = dict(commit="c" * 40, backend="cpu", precision="float32", pipeline="n1",
        run="another-admitted-production-run", ordered_case_ids=["own_f1"], storage_profile="lean", format_version=2)
    config = dict(run=task["run"], frames=[dict(case_id="own_f1")],
        opencl_acceleration=deepcopy(profile["compute_profiles"]["phase2"]["control"]),
        storage_profile=dict(profile="lean", colour_intermediates="keep"))
    assert common.require_applicable(profile, task, config=config) == "phase2"
    for changed in (dict(task, commit="d" * 40), dict(task, backend="cuda"),
                    dict(task, storage_profile="full"), dict(task, format_version=1)):
        with pytest.raises(CachePolicyError):
            common.require_applicable(profile, changed, config=config)


def test_original_native_retirement_source_owner_and_once_checks_remain_required(monkeypatch):
    """Original native retirement source owner and once checks remain required."""
    state, bindings, objects, slots = old.bound_documents_TESTONLY(monkeypatch)
    state.root_provider = dict(device_identity=pin("device.json"))
    state.device = dict(device_environment={key: None for key in common.DEVICE_ENVIRONMENT})
    state.output_entitlement["maximum_output_bytes"] = state.output_entitlement["administrative_output_entitlement_bytes"]
    evidence = slots["writer_evidence"]
    for row in evidence["registered_owner_environments"]:
        modules = objects[row["inventory"]["path"]]["first_qualification_modules"]
        modules["common_device_identity"] = state.root_provider["device_identity"]
        modules["actual_device_environment"] = state.device["device_environment"]
    arguments = dict(first=state, observation=slots["observation"], evidence=evidence,
        native=slots["native_completion"], measured=slots["measurement"], once_marker=bindings["once_marker"],
        read=lambda p: deepcopy(objects[p["path"]]))
    common._original_stage_native(**arguments)
    retirement = objects[slots["observation"]["retirement"]["path"]]
    retirement["terminal"]["accounting"]["active_processes"] = 1
    with pytest.raises(CachePolicyError, match="kernel retirement"):
        common._original_stage_native(**arguments)


@pytest.mark.parametrize("field,value", [("held_process_handle_observed_while_live", False),
    ("same_original_job_membership_observed", False)])
def test_native_primary_handle_proof_cannot_be_replaced_by_a_pass_boolean(monkeypatch, field, value):
    """Native primary handle proof cannot be replaced by a pass boolean."""
    state, bindings, objects, slots = old.bound_documents_TESTONLY(monkeypatch)
    retirement = objects[slots["observation"]["retirement"]["path"]]
    objects[retirement["actual_primary_binding"]["path"]][field] = value
    with pytest.raises(CachePolicyError, match="retained native handle"):
        common._original_stage_native(state, slots["observation"], slots["writer_evidence"],
            slots["native_completion"], slots["measurement"], once_marker=bindings["once_marker"],
            read=lambda p: deepcopy(objects[p["path"]]))


def test_common_import_remains_actual_worker_role_and_physical_cap_only():
    """Common import remains actual worker role and physical cap only."""
    state, objects, _ = protocol()
    bound = state.required_capacity_bound
    profile = dict(format=common.IMPORT_PROFILE, provider=state.provider_binding, purpose="jobs",
        entry_kind="worker", role_identifier="actual_jobs_role", namespace_admission=state.provider["namespace_admission"],
        capacity_bound=bound, additional_claim_cache_bytes=0, actual_writer_IO_proven=False,
        sampled_highwater_is_lifetime_peak=False, operational_service_entitlement_bytes=100)
    policy = SimpleNamespace(quota_bytes=500 * 1024**3, profiles={"jobs": state.provider_binding},
        volume_cap=SimpleNamespace(maximum_size_bytes=500 * 1024**3,
            document=lambda: deepcopy(bound["volume_cap_observation"]["binding"])))
    arguments = dict(purpose="jobs", entry_kind="worker", role_identifier="actual_jobs_role",
        policy=policy, namespace=state.provider["namespace_admission"])
    assert common.checked_import_context(profile, **arguments) == 100
    for field, value in (("purpose", "tests"), ("entry_kind", "runtime"),
            ("role_identifier", "another_role"), ("namespace", pin("other.json"))):
        with pytest.raises(CachePolicyError):
            common.checked_import_context(profile, **dict(arguments, **{field: value}))
    changed = dict(profile, actual_writer_IO_proven=True)
    with pytest.raises(CachePolicyError):
        common.checked_import_context(changed, **arguments)


def closed_common_stage_at_quota(monkeypatch, quota):
    """Bind a synthetic closed P1 stage to one exact positive cache volume."""
    from .test_common_profile_paths import synthetic_closed_dag
    value = synthetic_closed_dag(monkeypatch)
    state = value.states["auto"]["phase1"]
    bound = deepcopy(state.required_capacity_bound)
    usable = quota - 2 * 1024**2
    observed = bound["volume_cap_observation"]
    observed["binding"]["maximum_size_bytes"] = quota
    observed.update(virtual_size_bytes=quota, volume_total_bytes=usable)
    bound.update(volume_capacity_bytes=usable, required_capacity_bound_bytes=usable,
        operational_admission_headroom_bytes=usable-PHYSICAL_RESERVE_BYTES)
    value.documents[state.request["capacity_bound_plan"]["path"]] = deepcopy(bound)
    state.required_capacity_bound = deepcopy(bound)
    key = (state.provider["output_namespace"]["path"], state.provider["slots"]["writer_evidence"])
    evidence = deepcopy(value.slots[key][0])
    evidence.update(capacity_bound=deepcopy(bound), actual_volume_observation=deepcopy(observed))
    value.write_slot(state.provider, "writer_evidence", evidence)
    return value, state, evidence


@pytest.mark.parametrize("quota_gib", [96, 150, 500])
def test_common_closed_stage_uses_its_exact_bound_volume_in_original_checker(monkeypatch, quota_gib):
    """Real bounded/stage readers accept the released quota without imposing a fixed size."""
    value, state, evidence = closed_common_stage_at_quota(monkeypatch, quota_gib * 1024**3)
    assert state.common_mode is True
    proof = value.proofs["auto"]["phase1"][0]
    assert common.checked_stage(proof, root=value.roots["auto"], provider_binding=value.bindings["auto"],
        stores_source=value.stores, read=value.read, read_authorities=None) == proof
    if quota_gib != 500:
        observation = common._slot_read(state.provider, "observation")[0]
        with pytest.raises(CachePolicyError, match="fixed500GiB"):
            first.checked_bounded_evidence(observation, evidence)


@pytest.mark.parametrize("changed", ["binding", "virtual_size"])
def test_common_closed_stage_refuses_substituted_volume_evidence(monkeypatch, changed):
    """Self-consistent evidence still cannot replace the originally released cap binding."""
    value, state, evidence = closed_common_stage_at_quota(monkeypatch, 150 * 1024**3)
    observed = evidence["capacity_bound"]["volume_cap_observation"]
    if changed == "binding":
        observed["binding"] = dict(observed["binding"], maximum_size_bytes=96 * 1024**3)
        observed["virtual_size_bytes"] = 96 * 1024**3
    else:
        observed["virtual_size_bytes"] += 512
    evidence["actual_volume_observation"] = deepcopy(observed)
    value.write_slot(state.provider, "writer_evidence", evidence)
    with pytest.raises(CachePolicyError, match="common volume observation"):
        common.checked_stage(value.proofs["auto"]["phase1"][0], root=value.roots["auto"],
            provider_binding=value.bindings["auto"], stores_source=value.stores,
            read=value.read, read_authorities=None)


def test_closed_dag_must_cross_link_all_actual_p1_c_p2_sources(monkeypatch):
    """Closed dag must cross link all actual p1 c p2 sources."""
    root, binding, stores = provider(), pin("provider.json"), pin("stores.toml")
    p1 = dict(configuration=pin("p1.json"), completion=pin("p1-completed.json"))
    proofs = {stage: dict(stage=stage, completion=pin(stage + "-result.json"),
                         configuration=pin(stage + "-config.json")) for stage in common.STAGES}
    join = dict(binding=dict(completion=proofs["calibration"]["completion"],
        plan=proofs["calibration"]["configuration"]), phase1_configuration=p1["configuration"],
        phase1_completion=p1["completion"])
    sciences = {stage: dict(phase1=deepcopy(p1), calibration_join=None if stage == "phase1" else deepcopy(join))
                for stage in common.STAGES}
    def slot(view, role):
        stage = next(key for key in common.STAGES if view["scope"]["pipeline"] == common.PIPELINES[key])
        return deepcopy(proofs[stage] if role == "completion" else sciences[stage]), pin(stage + "-" + role + ".json")
    monkeypatch.setattr(common, "_slot_read", slot)
    monkeypatch.setattr(common, "checked_stage", lambda value, **kw: value)
    stages, checked = common._closed_stages(binding, root, stores, lambda p: None, lambda p: [])
    assert list(stages) == list(common.STAGES) and checked == proofs
    sciences["phase2"]["calibration_join"]["binding"]["completion"] = pin("foreign-c.json")
    with pytest.raises(CachePolicyError, match="genuine exact"):
        common._closed_stages(binding, root, stores, lambda p: None, lambda p: [])


def test_readable_review_files_and_release_boolean_cannot_publish_a_profile(monkeypatch):
    """Readable review files and release boolean cannot publish a profile."""
    state, _, _ = protocol()
    stages = {stage: pin(stage + "-closed.json") for stage in common.STAGES}
    release_pin = pin("actual-final-release.json")
    release = dict(format="darsia-common-phase-ordinary-release-v1", provider=state.provider_binding,
        stores_source=state.stores_source, scope=state.root_provider["scope"], stages=stages,
        ordinary_use_authorized=True, source_review=pin("source-review.json"),
        importer_focus=pin("importer-focus.json"), full_suite=pin("full-suite.json"),
        accepted_comparator=pin("comparator.py"), accepted_comparison=pin("comparison.json"))
    monkeypatch.setattr(common, "_closed_stages", lambda *a: (deepcopy(stages), {}))
    writes = []
    def read(bound):
        if bound == state.provider_binding:
            return deepcopy(state.root_provider)
        if bound == release_pin:
            return deepcopy(release)
        return dict(synthetic_review_receipt=True)
    monkeypatch.setattr(common, "_slot_write", lambda *args: writes.append(args))
    with pytest.raises(CachePolicyError, match="Content-bound full source review"):
        common.publish_profiles(state.provider_binding, stores_source=state.stores_source,
            ordinary_release=release_pin, read=read, read_authorities=None)
    assert writes == []


def test_final_source_close_rehashes_every_original_consumed_payload(tmp_path):
    """Final source close rehashes every original consumed payload."""
    inputs, resolved = [], []
    for index, content in enumerate((b"raw-payload", b"model-payload")):
        path = tmp_path / ("source-" + str(index) + ".bin")
        path.write_bytes(content)
        reference = dict(root_alias="raw", relative_path=path.name, role="image",
            sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        uses = ["source." + str(index)]
        inputs.append(dict(reference=reference, uses=uses))
        stat = path.stat()
        resolved.append(dict(reference=reference, uses=uses, path=str(path),
            version=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]))
    stores = tmp_path / "stores.toml"
    stores.write_bytes(b"# original stores\n")
    stores_source = dict(path=str(stores), sha256=hashlib.sha256(stores.read_bytes()).hexdigest(),
                         size_bytes=stores.stat().st_size)
    task = dict(run="sample", input_catalog=dict(inputs=inputs))
    state = SimpleNamespace(task=task, stores_source=stores_source)
    proof = common._reverify_source_inputs(state, object(), resolved)
    assert proof["input_count"] == proof["distinct_physical_files"] == 2
    assert proof["inputs"] == inputs and proof["post_retirement"] is True
    Path(resolved[0]["path"]).write_bytes(b"RAW-payload")
    with pytest.raises(CachePolicyError, match="identity changed|SHA or bytes"):
        common._reverify_source_inputs(state, object(), resolved)


def test_lean_terminal_lengths_include_retained_partials_and_both_operation_receipts():
    """Lean terminal lengths include retained partials and both operation receipts."""
    token = "a" * 32
    watched = "C:/synthetic/results"
    attempt = watched + "/job/sample/.attempts/" + "b" * 32
    directory = attempt + "/frames/sample_1"
    commit = dict(path=directory + "/commit.json", sha256="a" * 64, size_bytes=4)
    registration = dict(path=attempt + "/ops/frame_commit_" + token + ".json", sha256="a" * 64, size_bytes=2)
    publication = dict(path=attempt + "/ops/nas_copy_" + token + ".json", sha256="a" * 64, size_bytes=2)
    files = [("frames/sample_1/result.json", 6), ("frames/sample_1/commit.json", 4),
        ("frames/sample_1/.products.npz.partial", 3),
        ("ops/frame_commit_" + token + ".json", 2), ("ops/nas_copy_" + token + ".json", 2),
        ("lean_run_static.npz", 5), (".lean_run_static.npz.retry.partial", 3)]
    prefix = "job/sample/.attempts/" + "b" * 32 + "/"
    measured = dict(final_snapshot=dict(observed_scan_anomalies=[], file_stats=[
        dict(relative_path=prefix + path, size_bytes=size) for path, size in files]))
    state = SimpleNamespace(task=dict(ordered_case_ids=["sample_1"]),
        output_entitlement=dict(watched_output_scope=dict(path=watched)))
    completed = dict(records=[dict(frame_id="sample_1", nas_directory=directory,
        producer=dict(case_id="sample_1", attempt_directory=attempt, commit=commit,
            frame_registration=registration, publication=publication))])
    rows, run = common._lean_frame_measurements(state, completed, [commit, registration, publication], measured,
        dict(admitted_frame_bytes=17, admitted_run_bytes=8, run_record_files=["lean_run_static.npz"]))
    assert rows[0]["actual_bytes"] == 17 and rows[0]["retained_partials_included"] is True
    assert run["actual_bytes"] == 8
    with pytest.raises(CachePolicyError, match="frame bytes"):
        common._lean_frame_measurements(state, completed, [commit, registration, publication], measured,
            dict(admitted_frame_bytes=16, admitted_run_bytes=8, run_record_files=["lean_run_static.npz"]))


def test_ordinary_compute_policy_uses_only_observed_devices_and_qualified_cpu_fallback():
    """Ordinary compute policy uses only observed devices and qualified cpu fallback."""
    source = provider()
    control = dict(format="darsia-opencl-n1-v1", mode="auto", device_id=None)
    profile = dict(format=common.IMPORT_PROFILE, purpose="jobs", scope=source["scope"],
        stage_routes={stage: {key: row[key] for key in ("pipeline", "storage_profile", "colour_intermediates")}
            for stage, row in source["stages"].items()},
        compute_profiles=dict(phase2=dict(control=control, qualified_devices=[], cpu_fallback_qualified=True)))
    task = dict(commit="c" * 40, backend="cpu", precision="float32", pipeline="n1", run="sample",
        ordered_case_ids=["sample_1"], storage_profile="lean", format_version=2)
    config = dict(run="sample", frames=[dict(case_id="sample_1")], opencl_acceleration=deepcopy(control),
        storage_profile=dict(profile="lean", colour_intermediates="keep"))
    assert common.runtime_compute_policy(profile, task, config) == dict(
        qualified_devices=[], cpu_fallback_qualified=True)
    config["opencl_acceleration"] = dict(control, mode="cpu")
    with pytest.raises(CachePolicyError, match="control"):
        common.runtime_compute_policy(profile, task, config)


def test_common_lean_release_must_cover_its_exact_result_bound(monkeypatch):
    """Common lean release must cover its exact result bound."""
    from darsia.fluidflower.pipeline import capacity
    state, _, _ = protocol("phase2", allow_predecessors=True, monkeypatch=monkeypatch)
    lease = SimpleNamespace(first_qualification=state, source=state.stores_source)
    runtime = SimpleNamespace(cache_adapter=SimpleNamespace(lease=lease), stores=object(),
        job=dict(series_id="sample"), repository=Path("C:/synthetic/source"))
    p1 = dict(configuration=pin("actual-p1-config.json"), completion=pin("actual-p1-completed.json"))
    joined = dict(actual_frozen_refit_verified=True, calibration=pin("actual-frozen.json"), anchor=dict(elapsed_h=1))
    config = dict(run="sample", frames=[dict(case_id=case) for case in state.task["ordered_case_ids"]],
        storage_profile=dict(profile="lean", colour_intermediates="keep"),
        phase1=dict(p1, code_commit=state.task["commit"]), calibration=joined["calibration"], anchor=joined["anchor"],
        provenance=dict(common_frozen_refit=deepcopy(joined)))
    monkeypatch.setattr(common, "_slot_read", lambda view, role:
        (dict(phase1=p1, calibration_join=joined), pin("predecessor.json")))
    monkeypatch.setattr(capacity, "_phase2_lean_estimate", lambda *args, **kwargs:
        SimpleNamespace(needed=lambda **kwargs: 1001))
    with pytest.raises(CachePolicyError, match="exact lean bounds"):
        common.common_output_entitlement(runtime, state.task, config, [])
    monkeypatch.setattr(capacity, "_phase2_lean_estimate", lambda *args, **kwargs:
        SimpleNamespace(needed=lambda **kwargs: 1000))
    assert common.common_output_entitlement(runtime, state.task, config, []).maximum_output_bytes == 1000


def test_common_p2_refuses_a_different_closed_c_join_before_any_bound_computation(monkeypatch):
    """Common p2 refuses a different closed c join before any bound computation."""
    p1 = dict(configuration=pin("actual-p1.json"), completion=pin("actual-p1-completed.json"))
    joined = dict(actual_frozen_refit_verified=True, calibration=pin("actual-frozen.json"), anchor=dict(elapsed_h=1))
    state = SimpleNamespace(stage="phase2", root_provider=provider(), task=dict(commit="c" * 40))
    config = dict(phase1=dict(p1, code_commit="c" * 40), calibration=joined["calibration"], anchor=joined["anchor"],
        provenance=dict(common_frozen_refit=deepcopy(joined)))
    monkeypatch.setattr(common, "_slot_read", lambda *a: (dict(phase1=p1, calibration_join=joined), {}))
    assert common.require_phase2_predecessor(state, config) is None
    config["provenance"]["common_frozen_refit"]["calibration"] = pin("foreign-frozen.json")
    with pytest.raises(CachePolicyError, match="admission differs"):
        common.require_phase2_predecessor(state, config)


@pytest.mark.parametrize("config", ({}, {"opencl_acceleration": None}))
def test_common_absent_or_null_control_never_claims_a_declared_compute_path(config):
    """Legacy omission/null cannot become an explicit common CPU or auto proof."""
    assert common._compute_control(config, "phase2") is None
    assert common._compute_control(dict(statistics=config), "phase1") is None


@pytest.mark.parametrize("mutation", ("text", "closure", "verdict"))
def test_source_review_receipt_requires_exact_answer_content_and_source_closure(mutation):
    """A readable review receipt cannot conceal changed answer or source bytes."""
    root = provider()
    receipt = source_review(root)
    assert common._checked_source_review(receipt, root) == receipt
    if mutation == "text":
        receipt["answer"]["text"] += " Changed after review."
    elif mutation == "closure":
        receipt["producer_source_closure"] = receipt["producer_source_closure"][:-1]
    else:
        receipt["verdict"] = "NOT CLEAR"
    with pytest.raises(CachePolicyError, match="full review"):
        common._checked_source_review(receipt, root)
