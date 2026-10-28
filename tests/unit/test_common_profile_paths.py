"""Positive synthetic cache gates and genuine tiny-fit catalogue joins.

These fixtures supply protocol data, never host admission or a passing profile.
"""

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from darsia_cache import common_qualification as common
from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError
from tests.unit import test_common_qualification as fixture
from tests.unit.gui import test_cache_first_qualification as old


def admitted_lease(state):
    """Keep real admission/marker methods with a synthetic locked host ledger."""
    lease = old.lease_for(state)
    observed = deepcopy(state.required_capacity_bound["volume_cap_observation"])
    maximum = observed["virtual_size_bytes"]
    lease.policy.volume_cap = SimpleNamespace(maximum_size_bytes=maximum,
        document=lambda: deepcopy(observed["binding"]))
    lease.policy.quota_bytes = maximum
    report = old.report(active_cache_remaining_bytes=0)
    report.update(hard_filesystem_quota_verified=True, current_cache_logical_bytes=0,
                  volume_cap_observation=observed)
    lease.ledger.project = lambda requested: deepcopy(report)
    return lease, report


def empty_receipt_namespace(monkeypatch, state):
    """Model an empty admitted directory at the native namespace I/O boundary."""
    def opened(path, identity):
        """Keep the requested namespace identity observable to the unit test."""
        assert dict(path=path, identity=identity) == state.provider["output_namespace"]
        return SimpleNamespace(names=lambda: [], close=lambda: None)
    monkeypatch.setattr("darsia_cache.windows.WindowsNamespace", opened)


def test_consume_and_service_entitlement_use_original_launch_marker_and_reserve(monkeypatch):
    """A released stage consumes one marker and uses its actual native context."""
    state, _, _ = fixture.protocol()
    lease, report = admitted_lease(state)
    empty_receipt_namespace(monkeypatch, state)
    state.consume(lease)
    assert state.started and len(lease.ledger.documents) == 1
    marker = json.loads(next(iter(lease.ledger.documents.values())))
    assert marker["request"] == state.binding and marker["provider"] == state.provider_binding
    assert marker["scope"] == state.provider["scope"] and marker["no_retry"] is True
    with pytest.raises(CachePolicyError, match="controlled native Job"):
        state.service_entitlement(lease)
    lease.native_launch_context = old.pin("actual-synthetic-native-launch.json")
    assert state.service_entitlement(lease) == state.output_entitlement["maximum_service_bytes"]
    assert state.administrative["admission"] == report
    assert state.administrative["measured_profile"] is False
    assert len(lease.ledger.documents) == 1
    with pytest.raises(CachePolicyError):
        state.consume(lease)


def test_consume_refuses_changed_actual_volume_before_writing_a_marker(monkeypatch):
    """A fresh request cannot consume a different actual physical-volume proof."""
    state, _, _ = fixture.protocol()
    lease, report = admitted_lease(state)
    empty_receipt_namespace(monkeypatch, state)
    report["volume_cap_observation"]["binding"] = dict(synthetic="another volume")
    with pytest.raises(CachePolicyError, match="resources differ"):
        state.consume(lease)
    assert not lease.ledger.documents and state.started is False


def profile_metadata(monkeypatch):
    """Supply closed-stage-shaped metadata to the real profile composition gate."""
    root, provider, stores = fixture.provider(), old.pin("provider.json"), old.pin("stores.toml")
    stages = {stage: old.pin(stage + "-closed.json") for stage in common.STAGES}
    proofs, slots, documents = {}, {}, {}
    for index, stage in enumerate(common.STAGES, 1):
        service = old.pin(stage + "-service.json")
        documents[service["path"]] = dict(role_identifier="synthetic_worker_role")
        bound = old.bound_plan_TESTONLY()
        bound["volume_cap_observation"]["binding"] = dict(synthetic="same physical bound")
        control = None if stage == "calibration" else dict(format="darsia-opencl-n1-v1", mode="cpu", device_id=None)
        slots[stage] = dict(writer_evidence=dict(capacity_bound=bound,
            administrative_service_entitlement=dict(maximum_bytes=100 * index)),
            native_completion=dict(maximum_family_processes=index + 1),
            observation=dict(service_receipt=service, sampled_cache_logical_high_water_bytes=7 * index),
            scientific_evidence=dict(compute_evidence=dict(control=control,
                qualified_devices=[], cpu_fallback_qualified=True),
                result_space=dict(configuration=old.pin(stage + "-config.json"), case_count=index,
                    maximum_output_bytes=1000 * index)))
        proofs[stage] = dict(measurement=old.pin(stage + "-measurement.json"),
                             native_completion=old.pin(stage + "-native.json"))
    def read_slot(view, role):
        """Read one synthetic immutable slot by its declared stage route."""
        stage = next(stage for stage in common.STAGES if view["scope"]["pipeline"] == common.PIPELINES[stage])
        return deepcopy(slots[stage][role]), old.pin(stage + "-" + role + ".json")
    monkeypatch.setattr(common, "_slot_read", read_slot)
    return SimpleNamespace(root=root, provider=provider, stores=stores, stages=stages,
        proofs=proofs, slots=slots, read=lambda pin: deepcopy(documents[pin["path"]]))


def test_profile_documents_compose_actual_stage_maxima_without_claiming_lifetime_proof(monkeypatch):
    """Both role profiles retain all stage scopes and count one service maximum."""
    value = profile_metadata(monkeypatch)
    profiles = common._profile_documents(value.provider, value.root, value.stores,
        value.stages, value.proofs, value.read)
    assert set(profiles) == {"import_profile", "job_profile"}
    for role, profile in profiles.items():
        assert profile["format"] == (common.IMPORT_PROFILE if role == "import_profile" else common.JOB_PROFILE)
        assert profile["maximum_cache_bytes"] == profile["operational_service_entitlement_bytes"] == 300
        assert profile["sampled_cache_highwater_bytes"] == 21
        assert profile["native_supervision"]["maximum_family_processes"] == 4
        assert profile["additional_claim_cache_bytes"] == 0
        assert profile["actual_writer_IO_proven"] is False and profile["sampled_highwater_is_lifetime_peak"] is False
        assert profile["stages"] == value.stages
        assert profile["completion"] == value.stages["phase2"]
        assert profile["measurement"] == value.proofs["phase2"]["measurement"]
        assert set(profile["compute_profiles"]) == set(common.STAGES)
        assert set(profile["result_spaces"]) == set(common.STAGES)
    profiles["job_profile"]["compute_profiles"]["phase1"]["qualified_devices"].append(dict(synthetic=True))
    assert profiles["import_profile"]["compute_profiles"]["phase1"]["qualified_devices"] == []
    assert value.slots["phase1"]["scientific_evidence"]["compute_evidence"]["qualified_devices"] == []


def test_profile_documents_refuse_a_different_actual_volume_binding(monkeypatch):
    """Independent stages cannot silently compose distinct physical-cap authorities."""
    value = profile_metadata(monkeypatch)
    value.slots["calibration"]["writer_evidence"]["capacity_bound"]["volume_cap_observation"]["binding"] = {
        "synthetic": "another physical bound"}
    with pytest.raises(CachePolicyError, match="different actual physical-volume"):
        common._profile_documents(value.provider, value.root, value.stores,
            value.stages, value.proofs, value.read)


def synthetic_closed_dag(monkeypatch, *, scope=None, stage_data=None, accelerated=False):
    """Build one auto receipt DAG while keeping every content gate live.

    Only the host slot reader is replaced. Native retirement, original bounded
    evidence, worker/science links, retained files, predecessor recursion and
    final release are checked by their actual functions. These authored unit
    records do not constitute executed science, review or host qualification.
    """
    from darsia_cache.environment import (FIRST_WRITER_ENVIRONMENT_KEYS,
        first_qualification_environment)
    from darsia_cache.policy import PHYSICAL_RESERVE_BYTES

    template, _, _ = fixture.protocol()
    stage_data = {} if stage_data is None else stage_data
    documents, slot_values, roots, bindings, proofs, states = {}, {}, {}, {}, {}, {}
    stores = template.stores_source

    def read(pin):
        """Read only the exact synthetic descriptor selected by a real gate."""
        return deepcopy(documents[pin["path"]])

    def slot(view, role):
        """Model immutable host slots without replacing any validation step."""
        key = (view["output_namespace"]["path"], view["slots"][role])
        return deepcopy(slot_values[key])

    def write_slot(view, role, value):
        """Retain one synthetic immutable slot and its explicit descriptor."""
        pin = old.pin(view["output_namespace"]["path"].removeprefix("C:/TESTONLY/") + "/" + view["slots"][role])
        documents[pin["path"]] = deepcopy(value)
        slot_values[(view["output_namespace"]["path"], view["slots"][role])] = deepcopy(value), pin
        return pin

    monkeypatch.setattr(common, "_slot_read", slot)
    for arm_index, arm in ((4, "auto"),):
        root = deepcopy(template.root_provider)
        if scope is not None:
            root["scope"] = deepcopy(scope)
            for name in ("phase1", "phase2"):
                root["stages"][name].update(run=scope["run"], ordered_case_ids=deepcopy(scope["ordered_case_ids"]))
        for name, data in stage_data.items():
            if "task" in data:
                root["stages"][name]["ordered_case_ids"] = deepcopy(data["task"]["ordered_case_ids"])
        root["qualification_id"] = str(arm_index) * 32
        root["output_namespace"]["path"] = "C:/TESTONLY/" + arm + "/receipts"
        provider_pin = old.pin(arm + "/provider.json")
        roots[arm], bindings[arm], proofs[arm], states[arm] = root, provider_pin, {}, {}
        documents[provider_pin["path"]] = root
        documents[root["device_identity"]["path"]] = deepcopy(template.device)
        documents[root["namespace_admission"]["path"]] = dict(cache_identity=dict(synthetic="cache file identity"))
        documents[root["startup"]["path"]] = dict(synthetic="startup")
        p1, join = None, None
        for stage in common.STAGES:
            prefix = arm + "/" + stage + "/"
            projected = common.stage_provider(root, stage)
            request = deepcopy(template.request)
            task = dict(projected["scope"], attempt_id="2" * 32,
                job_id=arm + "-" + stage, format_version=2 if stage == "phase2" else 1,
                configuration=dict(root_alias="series", relative_path=prefix + "config.json", sha256="a" * 64, bytes=1),
                input_catalog=dict(inputs=[dict(reference=dict(root_alias="series", relative_path=prefix + "input.bin",
                    sha256="a" * 64, bytes=1, role="image"), uses=["synthetic.exact_input"])]))
            if stage != "phase1":
                task["storage_profile"] = common.PROFILES[stage]
            data = stage_data.get(stage, {})
            if "task" in data:
                task = deepcopy(data["task"])
            job_root = "C:/TESTONLY/jobs/" + task["job_id"]
            request.update(stage=stage, provider=provider_pin, task=old.pin(prefix + "task.json"),
                root_release=old.pin(prefix + "release.json"), configuration=old.pin(prefix + "config.json"),
                job=old.pin("jobs/" + task["job_id"] + "/job.json"), job_root=job_root,
                capacity_bound_plan=old.pin(prefix + "bound.json"),
                output_entitlement_plan=old.pin(prefix + "output.json"),
                retained_cache_inventory=None if stage == "phase1" else old.pin(prefix + "retained.json"),
                predecessors={name: proofs[arm][name][1] for name in common.STAGES[:common.STAGES.index(stage)]})
            if "configuration" in data:
                request["configuration"] = deepcopy(data["configuration"])
                if "task" not in data:
                    task["configuration"].update(sha256=data["configuration"]["sha256"],
                        bytes=data["configuration"]["size_bytes"])
            request["public_command"] = [root["python_executable"]["path"], "-B", "-m",
                "darsia.fluidflower.worker", "--job", job_root, "--stores", stores["path"]]
            request_pin = old.pin(prefix + "request.json")
            output = deepcopy(template.output_entitlement)
            output.update(stage=stage, task=request["task"], request_scope=provider_pin)
            output["watched_output_scope"]["path"] = "C:/TESTONLY/" + prefix + "results"
            review_pin = old.pin(prefix + "source-review.json")
            documents.update({request_pin["path"]: request, request["task"]["path"]: task,
                request["job"]["path"]: dict(commit=task["commit"], precision=task["precision"], expected_runs=[task["run"]]),
                request["capacity_bound_plan"]["path"]: deepcopy(template.required_capacity_bound),
                request["output_entitlement_plan"]["path"]: output,
                review_pin["path"]: fixture.source_review(root),
                request["root_release"]["path"]: dict(format=common.ROOT_RELEASE,
                    execution_authorized=True, provider=provider_pin, stores_source=stores, scope=deepcopy(root["scope"]),
                    source_review=review_pin, request_payload_sha256=first._digest(
                        {key: value for key, value in request.items() if key != "root_release"}))})
            state = common.CommonQualification(request_pin, provider_binding=provider_pin, stores_source=stores,
                namespace=root["namespace_admission"], read=read)
            states[arm][stage] = state
            # Read the original rich synthetic receipt shapes, then remove its
            # legacy comparison mocks before invoking any common content gate.
            with monkeypatch.context() as original_patch:
                previous, original_bindings, original_documents, original_slots = old.bound_documents_TESTONLY(original_patch)
            service_root = "C:/TESTONLY/" + prefix + "cache/jobs/" + "1" * 32
            original_environment = original_slots["writer_evidence"]["registered_owner_environments"][0]["actual_writer_environment"]
            environment = {key: service_root + "/" + key.lower() for key in FIRST_WRITER_ENVIRONMENT_KEYS}
            environment.update(DARSIA_FLUIDFLOWER_CACHE=service_root, NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator")
            environment = first_qualification_environment(environment, scope_root=service_root)
            replacements = [(previous.task, task), (previous.provider["scope"], projected["scope"]),
                (previous.binding, request_pin), (previous.provider_binding, provider_pin),
                (previous.stores_source, stores), (previous.provider["module_closure"], root["module_closure"]),
                (previous.provider["startup"], root["startup"]),
                (previous.provider["namespace_admission"], root["namespace_admission"]),
                (original_environment, environment),
                (original_slots["writer_evidence"]["service_namespace"], service_root)]

            def retarget(value):
                """Rebind complete authored receipt values, never invent a PASS."""
                for before, after in replacements:
                    if value == before:
                        return deepcopy(after)
                if isinstance(value, dict):
                    return {key: retarget(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [retarget(item) for item in value]
                if isinstance(value, str) and value.replace("\\", "/").startswith("C:/TESTONLY/"):
                    return "C:/TESTONLY/" + prefix + value.replace("\\", "/").removeprefix("C:/TESTONLY/")
                return deepcopy(value)

            for path, value in original_documents.items():
                documents.setdefault(retarget(path), retarget(value))
            slots = retarget(original_slots)
            observation, evidence, measured, native = (slots[role] for role in
                ("observation", "writer_evidence", "measurement", "native_completion"))
            maximum = max(output["maximum_output_bytes"], measured["closed_output_logical_bytes"] + 1)
            output["maximum_output_bytes"] = maximum
            state.output_entitlement = deepcopy(output)
            evidence.update(namespace_admission=root["namespace_admission"], capacity_bound_plan=request["capacity_bound_plan"],
                capacity_bound=deepcopy(state.required_capacity_bound), actual_volume_observation=deepcopy(
                    state.required_capacity_bound["volume_cap_observation"]))
            measured.update(format="darsia-common-phase-closed-output-diagnostic-v1", closed_output_stat_observation=True,
                original_retirement_accepted=True, output_plan=request["output_entitlement_plan"],
                output_scope=output["watched_output_scope"], entitlement_bytes=maximum,
                retained_cache_files_deleted_by_this_code=False, retained_cache_scope_is_exhaustive_OS_write_proof=False)
            measured["admission"]["volume"]["free_bytes"] = PHYSICAL_RESERVE_BYTES + maximum
            snapshot = dict(root=str(PureWindowsPath(service_root).parent.parent),
                root_identity=documents[root["namespace_admission"]["path"]]["cache_identity"], fresh=True,
                no_reparse_traversal=True, payload_hashes_included=True, observed_scan_anomalies=[],
                files=[], inventory_elapsed_seconds=.01, logical_bytes=0)
            measured.update(retained_cache_before=deepcopy(snapshot), retained_cache_after=deepcopy(snapshot))
            completion = deepcopy(data.get("completion", old.pin(prefix + "actual-completed.json")))
            documents[completion["path"]] = deepcopy(data.get("completed", dict(synthetic="closed original completion")))
            if stage == "phase1":
                p1 = dict(configuration=request["configuration"], completion=completion)
            if stage == "calibration":
                join = deepcopy(data.get("calibration_join", dict(format="darsia-common-phase-calibration-join-v1", actual_frozen_refit_verified=True,
                    phase1_configuration=p1["configuration"], phase1_completion=p1["completion"],
                    binding=dict(completion=completion, plan=request["configuration"]))))
            files = [old.pin(prefix + case + "/result.json") for case in task["ordered_case_ids"]]
            compute = dict(format="darsia-common-observed-compute-v1", cpu_fallback_qualified=True,
                declared_control=stage != "calibration", qualified_devices=[], observations=[], control=None)
            if stage != "calibration":
                compute["control"] = dict(format="darsia-opencl-n1-v1", mode=arm, device_id=None)
                device = dict(device_id=0, platform="synthetic", platform_version="1",
                    name="synthetic", driver="1", global_memory_bytes=1024**2)
                if accelerated:
                    compute["qualified_devices"] = [deepcopy(device)]
                compute["observations"] = [dict(case_id=case, source=pin,
                    facts=dict(control=deepcopy(compute["control"]), used=accelerated,
                        candidates=[dict(device=deepcopy(device), identical=True)] if accelerated else [],
                        selected_device=deepcopy(device) if accelerated else None,
                        verification_required=True, accelerated_calls=2 if accelerated else 0,
                        cpu_comparison_calls=2 if accelerated else 0, comparison_failures=0,
                        calls=3, cpu_reference_ns=12, accelerated_ns=5 if accelerated else 0,
                        cpu_comparison_ns=24 if accelerated else 0, verification_ns=3 if accelerated else 0,
                        byte_verified_accelerated_calls=2 if accelerated else 0,
                        guarded_accelerated_calls=0, nonfinite_cpu_calls=0,
                        application_device_buffer_bytes=4096 if accelerated else 0,
                        peak_application_device_buffer_bytes=4096 if accelerated else 0))
                    for case, pin in zip(task["ordered_case_ids"], files)]
            resource = dict(configuration=request["configuration"], case_count=len(task["ordered_case_ids"]),
                maximum_output_bytes=maximum)
            worker = dict(format="darsia-common-worker-scientific-closure-v1", request=request_pin, provider=provider_pin,
                stores_source=stores, stage=stage, task=task, owner=evidence["registered_owner_environments"][0]["owner"],
                native_launch=observation["native_launch"], service_receipt=observation["service_receipt"],
                configuration=request["configuration"], completion=completion,
                publication=dict(job=request["job"], done=old.pin(prefix + "done.json"), products_index=old.pin(prefix + "index.json")),
                phase1=deepcopy(p1), calibration_join=deepcopy(join), output_files=files,
                frame_records=[dict(frame_id=case) for case in task["ordered_case_ids"]],
                resolved_inputs=[dict(reference=deepcopy(row["reference"]), uses=deepcopy(row["uses"]),
                    path="C:/TESTONLY/resolved/" + row["reference"]["relative_path"],
                    version=[1, 2, row["reference"]["bytes"], 4]) for row in task["input_catalog"]["inputs"]],
                compute_evidence=compute, result_space=resource,
                lean_bound=dict(admitted_frame_bytes=100, admitted_run_bytes=100) if stage == "phase2" else None,
                complete=True, process_retirement_claimed=False)
            worker_pin = write_slot(projected, "worker_scientific_evidence", worker)
            observation.update(kernel_lifetime_memory=dict(synthetic="observed kernel peaks"), native_launch_to_retirement_seconds=2.)
            for row in observation["actual_registered_owner_exit_inventories"]:
                row.update(common_device_identity=root["device_identity"], actual_device_environment=template.device["device_environment"],
                    common_worker_scientific_evidence=worker_pin)
                modules = documents[row["inventory"]["path"]]["first_qualification_modules"]
                modules.update(common_device_identity=root["device_identity"], actual_device_environment=template.device["device_environment"],
                    common_worker_scientific_evidence=worker_pin, actual_task=task,
                    predeclared_module_closure_sha256=first._digest(root["module_closure"]))
            evidence["registered_owner_environments"] = deepcopy(observation["actual_registered_owner_exit_inventories"])
            for role in ("observation", "measurement", "native_completion"):
                slots[role].update(request=request_pin, provider=provider_pin, stores_source=stores, scope=projected["scope"])
            evidence.update(request=request_pin, provider=provider_pin, stores_source=stores, scope=projected["scope"])
            observation_pin = write_slot(projected, "observation", observation)
            measurement_pin = write_slot(projected, "measurement", measured)
            native.update(observation=observation_pin, startup=root["startup"], module_closure=root["module_closure"])
            native_pin = write_slot(projected, "native_completion", native)
            observation["output_measurement"] = measurement_pin
            write_slot(projected, "observation", observation)
            evidence["observation_sha256"] = first._digest(observation)
            evidence_pin = write_slot(projected, "writer_evidence", evidence)
            science = dict(format="darsia-common-phase-scientific-closure-v1", stage=stage, task=task,
                completion=completion, configuration=request["configuration"], pipeline=task["pipeline"], run=task["run"],
                ordered_case_ids=task["ordered_case_ids"], storage_profile=common.PROFILES[stage], colour_intermediates="keep",
                phase1=deepcopy(p1), calibration_join=deepcopy(join), output_files=files,
                resume_policy="original-no-prefix-requested; phase-one-prefix-refused", source_bytes_reverified=True,
                source_verification=dict(format="darsia-common-retired-input-bytes-v1",
                    catalogue_sha256=first._digest(task["input_catalog"]), inputs=task["input_catalog"]["inputs"],
                    input_count=1, distinct_physical_files=1, post_retirement=True, all_sha_and_sizes_verified=True),
                compute_evidence=dict(deepcopy(compute), kernel_lifetime_memory=observation["kernel_lifetime_memory"],
                    native_launch_to_retirement_seconds=2.), result_space=deepcopy(resource),
                lean_frame_bytes=[dict(case_id=case, actual_bytes=1, admitted_bytes=100)
                    for case in task["ordered_case_ids"]] if stage == "phase2" else [],
                worker_scientific_evidence=worker_pin, complete=True)
            science_pin = write_slot(projected, "scientific_evidence", science)
            marker_pin = old.pin(prefix + "cache-first-" + state._once_key() + ".json")
            documents[marker_pin["path"]] = dict(format="darsia-cache-first-qualification-consumed-v1",
                request=request_pin, provider=provider_pin, stores_source=stores, scope=projected["scope"], no_retry=True)
            proof = dict(format=common.STAGE_PROOF, provider=provider_pin, stores_source=stores, stage=stage,
                request=request_pin, task=request["task"], job=request["job"], completion=completion,
                configuration=request["configuration"], observation=observation_pin, native_completion=native_pin,
                measurement=measurement_pin, writer_evidence=evidence_pin, scientific_evidence=science_pin,
                device_identity=root["device_identity"], once_marker=marker_pin, complete=True, passed=True)
            proofs[arm][stage] = proof, write_slot(projected, "completion", proof)
    return SimpleNamespace(documents=documents, slots=slot_values, roots=roots, bindings=bindings,
        stores=stores, proofs=proofs, states=states, read=read, write_slot=write_slot)


def test_checked_stage_validates_original_receipts_and_all_three_live_scientific_joins(monkeypatch):
    """Authored synthetic receipts must pass actual recursive stage validators."""
    value = synthetic_closed_dag(monkeypatch)
    assert set(value.roots) == {"auto"}
    for stage in common.STAGES:
        proof, _ = value.proofs["auto"][stage]
        assert common.checked_stage(proof, root=value.roots["auto"], provider_binding=value.bindings["auto"],
            stores_source=value.stores, read=value.read, read_authorities=None) == proof
    stage_pins, _ = common._closed_stages(value.bindings["auto"], value.roots["auto"],
        value.stores, value.read, None)
    assert stage_pins == {stage: pair[1] for stage, pair in value.proofs["auto"].items()}


def test_checked_stage_rejects_changed_original_retained_cache_inventory(monkeypatch):
    """A valid predecessor cannot cover an inconsistent terminal file total."""
    value = synthetic_closed_dag(monkeypatch)
    root = value.roots["auto"]
    view = common.stage_provider(root, "phase1")
    key = (view["output_namespace"]["path"], view["slots"]["measurement"])
    value.slots[key][0]["retained_cache_after"]["logical_bytes"] = 1
    proof = value.proofs["auto"]["phase1"][0]
    with pytest.raises(CachePolicyError, match="cache total omits"):
        common.checked_stage(proof, root=root, provider_binding=value.bindings["auto"],
            stores_source=value.stores, read=value.read, read_authorities=None)


def synthetic_ordinary_release(value, tmp_path):
    """Bind synthetic PASS receipt contents and a real comparator source file."""
    root, binding = value.roots["auto"], value.bindings["auto"]
    stages = {stage: pair[1] for stage, pair in value.proofs["auto"].items()}
    source = tmp_path / "accepted_comparator.py"
    raw = b"# Synthetic accepted comparator identity; never executed.\n"
    source.write_bytes(raw)
    comparator = dict(path=str(source), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    stage_results, selected_fields = {}, {}
    measurement = dict(format="darsia-common-inrun-measurement-v1", provider=binding, stages=stages,
        scope=deepcopy(root["scope"]), end_to_end_scope="original native launch through accepted retirement",
        stage_measurements={})
    fields = {field: "PASS" for field in ("aqueous", "gas", "concentration", "mass_total", "mass_aqueous", "mass_gas")}
    for stage in common.STAGES:
        view = common.stage_provider(root, stage)
        science, pin = value.slots[(view["output_namespace"]["path"], view["slots"]["scientific_evidence"])]
        compute = science["compute_evidence"]
        stage_results[stage] = dict(scientific_evidence=pin, outputs=deepcopy(science["output_files"]), verdict="PASS")
        measurement["stage_measurements"][stage] = dict(stage=stages[stage],
            end_to_end_wall_seconds=compute["native_launch_to_retirement_seconds"],
            kernel_lifetime_memory=deepcopy(compute["kernel_lifetime_memory"]), host_binding=root["namespace_admission"],
            application_device_buffer_bytes=max((row["facts"]["peak_application_device_buffer_bytes"]
                for row in compute["observations"]), default=0), compute_observations=deepcopy(compute["observations"]))
        if stage != "calibration":
            selected_fields[stage] = [dict(case_id=row["case_id"], source=deepcopy(row["source"]),
                facts_sha256=first._digest(row["facts"]), fields=deepcopy(fields)) for row in compute["observations"]]
    measurement_pin, comparison_pin = old.pin("inrun-measurement.json"), old.pin("accepted-comparison.json")
    value.documents[measurement_pin["path"]] = measurement
    comparison = dict(format="darsia-common-inrun-comparison-v1", verdict="PASS", comparator=comparator,
        scope=deepcopy(root["scope"]), stores_source=value.stores, provider=binding, stages=stages,
        stage_results=stage_results, selected_fields=selected_fields, measurement=measurement_pin)
    value.documents[comparison_pin["path"]] = comparison
    release = dict(format="darsia-common-phase-ordinary-release-v1", provider=binding, stores_source=value.stores,
        scope=deepcopy(root["scope"]), stages=stages, ordinary_use_authorized=True,
        source_review=old.pin("final-source-review.json"), importer_focus=old.pin("importer-focus.json"),
        full_suite=old.pin("full-suite.json"), accepted_comparator=comparator, accepted_comparison=comparison_pin)
    value.documents[release["source_review"]["path"]] = fixture.source_review(root)
    for role in ("importer_focus", "full_suite"):
        value.documents[release[role]["path"]] = dict(format="darsia-common-validation-receipt-v1", kind=role,
            verdict="PASS", scope=deepcopy(root["scope"]), module_closure_sha256=first._digest(root["module_closure"]),
            tests_passed=1, tests_failed=0)
    release_pin = old.pin("ordinary-release.json")
    value.documents[release_pin["path"]] = release
    return release, release_pin


@pytest.mark.parametrize("accelerated", [False, True])
def test_resolve_profile_reads_the_complete_live_dag_and_typed_release_contents(monkeypatch, tmp_path, accelerated):
    """Both profile roles require every real gate and one in-run verified release."""
    value = synthetic_closed_dag(monkeypatch, accelerated=accelerated)
    release, release_pin = synthetic_ordinary_release(value, tmp_path)
    root, binding = value.roots["auto"], value.bindings["auto"]
    stages, proofs = common._closed_stages(binding, root, value.stores, value.read, None)
    assert common._ordinary_release(release, binding=binding, root=root, stores_source=value.stores,
        stages=stages, read=value.read) == release
    profiles = common._profile_documents(binding, root, value.stores, stages, proofs, value.read)
    profile_pins = {role: value.write_slot(root, role, profile) for role, profile in profiles.items()}
    value.write_slot(root, "completion", dict(format=common.BUNDLE, provider=binding, stores_source=value.stores,
        scope=deepcopy(root["scope"]), stages=stages, profiles=profile_pins,
        ordinary_use_release=release_pin, complete=True, passed=True))
    for kind, role in (("import", "import_profile"), ("jobs", "job_profile")):
        assert common.resolve_profile(binding, root, kind, value.read, stores_source=value.stores) == profiles[role]
    value.documents[release["full_suite"]["path"]]["tests_failed"] = 1
    with pytest.raises(CachePolicyError, match="passing test result"):
        common.resolve_profile(binding, root, "jobs", value.read, stores_source=value.stores)


@pytest.mark.parametrize("field,bad", [
    ("cpu_comparison_calls", 1), ("verification_required", False), ("comparison_failures", 1)])
def test_inrun_comparison_keeps_original_every_call_cpu_verification(monkeypatch, tmp_path, field, bad):
    """The accepted report cannot waive the original accelerated-call predicates."""
    value = synthetic_closed_dag(monkeypatch, accelerated=True)
    root, binding = value.roots["auto"], value.bindings["auto"]
    view = common.stage_provider(root, "phase2")
    science = value.slots[(view["output_namespace"]["path"], view["slots"]["scientific_evidence"])][0]
    science["compute_evidence"]["observations"][0]["facts"][field] = bad
    release, _ = synthetic_ordinary_release(value, tmp_path)
    with pytest.raises(CachePolicyError, match="Every accelerated proof call"):
        common._ordinary_release(release, binding=binding, root=root, stores_source=value.stores,
            stages=release["stages"], read=value.read)


@pytest.mark.parametrize("changed", ["source", "facts_sha256", "outputs", "measurement", "timing", "memory"])
def test_inrun_comparison_requires_original_outputs_facts_and_intact_report(monkeypatch, tmp_path, changed):
    """A report must retain the single chain's exact science and measurement links."""
    value = synthetic_closed_dag(monkeypatch, accelerated=True)
    release, _ = synthetic_ordinary_release(value, tmp_path)
    comparison = value.documents[release["accepted_comparison"]["path"]]
    report = value.documents[comparison["measurement"]["path"]]
    if changed == "source":
        comparison["selected_fields"]["phase1"][0]["source"] = old.pin("different-runtime.json")
    elif changed == "facts_sha256":
        comparison["selected_fields"]["phase1"][0]["facts_sha256"] = "0" * 64
    elif changed == "outputs":
        comparison["stage_results"]["calibration"]["outputs"] = []
    elif changed == "measurement":
        del report["stage_measurements"]["phase1"]["compute_observations"]
    elif changed == "timing":
        report["stage_measurements"]["phase2"]["end_to_end_wall_seconds"] = 0
    else:
        report["stage_measurements"]["phase2"]["application_device_buffer_bytes"] = 0
    with pytest.raises(CachePolicyError):
        common._ordinary_release(release, binding=value.bindings["auto"], root=value.roots["auto"],
            stores_source=value.stores, stages=release["stages"], read=value.read)


def test_inrun_comparison_requires_readable_bound_measurement(monkeypatch, tmp_path):
    """A bound reader's missing-file refusal cannot be replaced by report defaults."""
    value = synthetic_closed_dag(monkeypatch)
    release, _ = synthetic_ordinary_release(value, tmp_path)
    comparison = value.documents[release["accepted_comparison"]["path"]]
    missing = comparison["measurement"]

    def read(pin):
        """Model the original bound reader's missing-file refusal only."""
        if pin == missing:
            raise FileNotFoundError(pin["path"])
        return value.read(pin)

    with pytest.raises(FileNotFoundError):
        common._ordinary_release(release, binding=value.bindings["auto"], root=value.roots["auto"],
            stores_source=value.stores, stages=release["stages"], read=read)


def test_inrun_fallback_reports_cpu_facts_without_qualifying_an_unseen_device(monkeypatch, tmp_path):
    """Original CPU fallback needs no accelerated-call minimum or claimed gain."""
    value = synthetic_closed_dag(monkeypatch)
    root, binding = value.roots["auto"], value.bindings["auto"]
    for stage in ("phase1", "phase2"):
        view = common.stage_provider(root, stage)
        science = value.slots[(view["output_namespace"]["path"], view["slots"]["scientific_evidence"])][0]
        for row in science["compute_evidence"]["observations"]:
            row["facts"].update(nonfinite_cpu_calls=1, cpu_reference_ns=None)
    release, _ = synthetic_ordinary_release(value, tmp_path)
    assert common._ordinary_release(release, binding=binding, root=root, stores_source=value.stores,
        stages=release["stages"], read=value.read) == release
    comparison = value.documents[release["accepted_comparison"]["path"]]
    report = value.documents[comparison["measurement"]["path"]]
    assert "cpu" not in value.roots
    for stage in ("phase1", "phase2"):
        rows = report["stage_measurements"][stage]["compute_observations"]
        assert all(row["facts"]["nonfinite_cpu_calls"] == 1 for row in rows)
        assert all(row["facts"]["selected_device"] is None for row in rows)
        assert all(row["facts"]["cpu_comparison_calls"] == 0 for row in rows)


def _write(path, value):
    """Write one synthetic immutable-byte test document and return its pin."""
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    Path(path).write_bytes(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def real_catalogue_join(tmp_path, monkeypatch):
    """Build original tiny numerical fit outputs and a verified own-run P1 closure."""
    from tests.fluidflower.test_n1_refit import make_plan, make_rows, make_row
    from tests.unit import test_staged_data as staged_fixture
    from darsia.fluidflower.mass import n1_refit
    from darsia.fluidflower.mass.n1_pipeline import DataReader
    from darsia.fluidflower.pipeline import common_calibration as calibration
    from darsia.fluidflower.pipeline import refit_runner, staged_data

    commit = "1" * 40
    fit_runs = ["fit-a", "fit-b", "fit-c"]
    plan = make_plan(fit_runs)
    rows = make_rows(fit_runs)
    plan.update(format=refit_runner.PLAN_FORMAT, run="refit", fold=None, statistics={})
    for row in rows:
        case = row["run"] + "_f" + str(row["frame_index"])
        pin = _write(tmp_path / (case + ".json"), row)
        plan["statistics"].setdefault(row["run"], []).append(dict(case_id=case,
            frame_index=row["frame_index"], selected_target_h=row["selected_target_h"], statistics=pin))
    plan["baseline"] = dict(calibration=_write(tmp_path / "baseline.json",
        dict(parameters=plan["rules"]["baseline"])))
    plan_pin = _write(tmp_path / "plan.json", plan)
    rules = refit_runner.validate_plan(plan)
    fit = n1_refit.fit_candidate(rows, rules)
    assert fit["stop_required"] is False
    fit_pin = _write(tmp_path / "fit.json", fit)
    cases = [row["case_id"] for run in sorted(plan["statistics"]) for row in plan["statistics"][run]]
    task = dict(pipeline=refit_runner.PIPELINE, run="refit", commit=commit,
        attempt_id="a" * 32, backend="cpu", precision="float32", configuration=plan_pin,
        ordered_case_ids=cases)
    frozen_pin = _write(tmp_path / "frozen_calibration.json", refit_runner.frozen_calibration(
        plan, fit, task=task, files={"fit.json": fit_pin}))
    completed_pin = _write(tmp_path / "completed.json", dict(format=refit_runner.RESULT_FORMAT,
        pipeline=refit_runner.PIPELINE, run="refit", fold=None, complete=True,
        stop_required=False, worker_references_closed=True, commit=commit, task=task,
        attempt_id=task["attempt_id"], backend="cpu", precision="float32", frames=len(cases),
        parameters=fit["parameters"], files={"fit.json": fit_pin, "frozen_calibration.json": frozen_pin},
        frozen_calibration=frozen_pin))

    data = staged_fixture.phase1_data.__wrapped__()
    data["run"] = "held"
    spec = data["frames"][0]
    spec["source"]["run"] = "held"
    spec["elapsed_h"] = spec["source"]["elapsed_h"] = spec["mass_spec"]["selected_target_h"] = 2.25
    materialized = {}
    def local_inputs(value):
        """Bind finite opaque ABI inputs locally without decoding image arrays."""
        if isinstance(value, dict):
            if {"path", "sha256"}.issubset(value):
                key = value["path"], value["sha256"]
                if key not in materialized:
                    path = tmp_path / ("abi-input-" + str(len(materialized)) + ".json")
                    materialized[key] = _write(path, dict(synthetic="opaque input; never decoded"))
                return dict(value, **materialized[key])
            return {key: local_inputs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [local_inputs(item) for item in value]
        return value
    data["statistics"] = local_inputs(data["statistics"])
    data["phase"]["reference"]["full"] = local_inputs(data["phase"]["reference"]["full"])
    spec["mass_spec"] = local_inputs(spec["mass_spec"])
    own = make_row("held", spec["frame_index"], 2.25)
    own["source_binding"] = dict(source_sha256=spec["source"]["source_sha256"])
    original_write = staged_fixture._write
    def own_statistics(path, value=None, raw=None):
        """Replace only synthetic statistics payload bytes with an actual estimate row."""
        if Path(path).name == "native_statistics.json.gz":
            raw = gzip.compress(json.dumps(own).encode())
        return original_write(path, value=value, raw=raw)
    monkeypatch.setattr(staged_fixture, "_write", own_statistics)
    closure = staged_fixture._closure(tmp_path / "own-phase1", data)
    reader = DataReader()
    verified = staged_data.verify_phase1_completion(closure["completion"], closure["configuration"],
        expected_commit=commit, reader=reader)
    anchor = calibration.heldout_application_anchor(verified, plan, reader=reader)["anchor"]
    joined = calibration.verify_frozen_refit(dict(completion=completed_pin, plan=plan_pin),
        verified_phase1=verified, fitted_calibration=frozen_pin, fitted_anchor=anchor,
        expected_commit=commit, reader=reader)
    phase2 = staged_data.seal_phase2_configuration(tmp_path / "sealed-phase2.json", closure["completion"], closure["configuration"],
        expected_phase1_commit=commit, fitted_calibration=frozen_pin, fitted_anchor=anchor,
        reader=reader, frozen_refit=dict(completion=completed_pin, plan=plan_pin),
        storage_profile=dict(profile="lean", colour_intermediates="keep"))
    config = json.loads(Path(phase2["configuration"]["path"]).read_bytes())
    return SimpleNamespace(root=tmp_path, config=config, join=joined, plan=plan,
        heldout_statistics=verified["records"][0]["record"]["statistics"], fit_pin=fit_pin,
        refit_task=task, plan_pin=plan_pin, completed_pin=completed_pin, frozen_pin=frozen_pin,
        phase1=closure, phase2_pin=phase2["configuration"])


def test_catalogue_refit_uses_real_catalogue_and_original_heldout_anchor(real_catalogue_join):
    """Real named-input admission keeps held-out application rows outside the fit."""
    from darsia.fluidflower.pipeline.common_calibration import catalogue_refit
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    value = real_catalogue_join
    before = deepcopy(value.config), deepcopy(value.plan), Path(value.fit_pin["path"]).read_bytes()
    catalogue = InputCatalog(dict(series=str(value.root)))
    assert catalogue_refit(catalogue, value.config) == value.join
    uses = {use for row in catalogue.entries.values() for use in row["uses"]}
    assert "mass.common_refit.application." + value.config["frames"][0]["case_id"] in uses
    assert all(".fit.held." not in use for use in uses)
    assert len([use for use in uses if use.startswith("mass.common_refit.fit.")]) == sum(
        len(rows) for rows in value.plan["statistics"].values())
    assert (value.config, value.plan, Path(value.fit_pin["path"]).read_bytes()) == before
    assert "held" not in value.plan["statistics"]


def test_real_catalogue_rechecks_heldout_source_bytes(real_catalogue_join):
    """A changed own-run statistics payload cannot retain a valid application join."""
    from darsia.fluidflower.pipeline.common_calibration import catalogue_refit
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    value = real_catalogue_join
    Path(value.heldout_statistics["path"]).write_bytes(b"changed original own statistics")
    with pytest.raises(ValueError):
        catalogue_refit(InputCatalog(dict(series=str(value.root))), value.config)


def real_common_claim(value):
    """Collect exact sealed P2 inputs for an original real RunIO replay."""
    from darsia.fluidflower.job_filesystem import LocalFileAdmission
    from darsia.fluidflower.pipeline.job_inputs import InputCatalog
    from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
    from darsia.fluidflower.pipeline.store_runtime import resolve_catalog
    from darsia.fluidflower.store_map import parse_store_map
    config = deepcopy(value.config)
    config["opencl_acceleration"] = dict(format="darsia-opencl-n1-v1", mode="auto", device_id=None)
    configuration = _write(value.root / "admitted-proof-phase2.json", config)
    manifest = _write(value.root / "series-manifest.json", dict(synthetic="explicit series metadata"))
    execution = value.root / "execution.toml"
    text = '[[runs]]\nrun=' + json.dumps(config["run"]) + '\n[runs.configuration]\n'
    text += 'path=' + json.dumps(configuration["path"]) + '\nsha256="' + configuration["sha256"] + '"\n'
    text += 'size_bytes=' + str(configuration["size_bytes"]) + '\n'
    execution.write_bytes(text.encode())
    execution_pin = dict(path=str(execution), sha256=hashlib.sha256(execution.read_bytes()).hexdigest(),
        size_bytes=execution.stat().st_size)
    catalogue = InputCatalog(dict(series=str(value.root)), config.get("data_aliases", ()))
    series = catalogue.add(execution_pin, "configuration", "run.execution_series")
    series_manifest = catalogue.add(manifest, "configuration", "run.series_manifest")
    inputs = catalogue.collect(configuration, config)
    inputs["runtime_support"] = "darsia-role-io-runtime-v1"
    task = dict(format="darsia-alias-task-v1", format_version=2, storage_profile="lean", pipeline="n1",
        run=config["run"], ordered_case_ids=[frame["case_id"] for frame in config["frames"]],
        backend="cpu", precision="float32", commit=value.refit_task["commit"],
        job_id="synthetic-proof", attempt_id="1" * 32, legacy_paths=False, execution_ready=True,
        configuration=inputs["configuration"], series=series, series_manifest=series_manifest,
        input_catalog=inputs, calibration_sha256=config["calibration"]["sha256"], threads=1)
    results = value.root / "owned-results"
    results.mkdir()
    stores = parse_store_map(('[machine]\nid="synthetic"\n[stores.synthetic]\nseries=' + json.dumps(str(value.root)) +
        '\nresults=' + json.dumps(str(results)) + '\n').encode())
    rows = resolve_catalog(task, stores, "synthetic")
    admission = LocalFileAdmission(True, "synthetic", "synthetic", "not a filesystem qualification", str(results))
    output = LocalOutputBinding("synthetic", results,
        results / task["job_id"] / task["run"] / ".attempts" / task["attempt_id"], admission)
    claim = SimpleNamespace(task=task, token=task["attempt_id"], input_rows=rows, local_output=output)
    return claim, config, configuration


def test_worker_prepare_releases_only_the_actual_joined_common_binding(real_catalogue_join, monkeypatch):
    """Real SHA admission, catalogue replay and genuine C join precede proof flags."""
    from darsia.fluidflower.pipeline import worker_io
    value = real_catalogue_join
    claim, config, configuration = real_common_claim(value)
    scope = {key: deepcopy(claim.task[key]) for key in common.COMMON_SCOPE_FIELDS}
    dag = synthetic_closed_dag(monkeypatch, scope=scope, stage_data={
        "phase1": dict(configuration=value.phase1["configuration"], completion=value.phase1["completion"],
            completed=value.phase1["document"]),
        "calibration": dict(configuration=value.plan_pin, completion=value.completed_pin,
            calibration_join=value.join),
        "phase2": dict(task=claim.task, configuration=configuration)})
    state = dag.states["auto"]["phase2"]
    lease = SimpleNamespace(first_qualification=state, source=state.stores_source, profile={})
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.commit, runtime.bindings = claim.task["commit"], {}
    runtime.cache_adapter = SimpleNamespace(lease=lease)
    before = {row.path: row.path.read_bytes() for row in claim.input_rows}
    binding = runtime.prepare(claim)
    assert binding is runtime.bindings[claim.token]
    assert binding.configuration == config and binding.task == claim.task
    assert binding.opencl_proof_released and binding.common_closure_released and binding.opencl_compare_cpu
    assert binding.common_job_binding == state.request["job"]
    assert not hasattr(binding, "opencl_policy")
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert "held" not in value.plan["statistics"]
