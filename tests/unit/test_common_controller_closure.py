"""The original post-retirement controller consumes bytes, never scientific imports."""
import builtins
from contextlib import nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia_cache import common_qualification as common
from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError


def metadata_closure(tmp_path, monkeypatch):
    def payload(name, content=b"{}"):
        path = tmp_path / name
        path.write_bytes(content)
        return dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    stores, configuration, output = payload("stores.toml"), payload("config.json"), payload("result.json")
    task = dict(run="sample", ordered_case_ids=["sample_1"],
        input_catalog=dict(inputs=[dict(reference=dict(root_alias="input", relative_path="config.json",
            role="configuration", sha256=configuration["sha256"], bytes=configuration["size_bytes"]),
            uses=["run.configuration"])]))
    provider = dict(module_closure=[payload("source.py")])
    owner = dict(pid=123, creation_filetime_100ns="100")
    native, service = payload("native.json"), payload("service.json")
    request_pin, provider_pin = payload("request.json"), payload("provider.json")
    job, task_pin = payload("job.json"), payload("task.json")
    state = SimpleNamespace(stage="phase1", common_mode=True, task=task, stores_source=stores,
        binding=request_pin, provider_binding=provider_pin, provider=provider,
        request=dict(configuration=configuration, job=job, task=task_pin),
        root_provider=dict(device_identity=payload("device.json")), _once_key=lambda: "synthetic",
        _read_authorities=None)
    control = dict(format="darsia-opencl-n1-v1", mode="cpu", device_id=None)
    compute = dict(format="darsia-common-observed-compute-v1", control=control, declared_control=True,
        cpu_fallback_qualified=True, qualified_devices=[], observations=[dict(case_id="sample_1",
            source=output, facts=dict(control=control, used=False))])
    stat = Path(configuration["path"]).stat()
    worker = dict(format="darsia-common-worker-scientific-closure-v1", request=request_pin,
        provider=provider_pin, stores_source=stores, stage="phase1", task=task, owner=owner,
        native_launch=native, service_receipt=service, configuration=configuration,
        completion=payload("completed.json"), publication=dict(job=job,
            done=payload("done.json"), products_index=payload("index.json")),
        phase1=dict(configuration=configuration, completion=payload("phase1_completed.json")),
        calibration_join=None, output_files=[output], frame_records=[],
        resolved_inputs=[dict(task["input_catalog"]["inputs"][0], path=configuration["path"],
            version=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])],
        compute_evidence=compute, result_space=dict(configuration=configuration, case_count=1,
            maximum_output_bytes=100, measured_native_layout=None), lean_bound=None,
        complete=True, process_retirement_claimed=False)
    worker_pin, exit_pin = payload("worker.json"), payload("exit.json")
    exited = dict(owner=owner, first_qualification_modules=dict(actual_task=task,
        common_worker_scientific_evidence=worker_pin,
        predeclared_module_closure_sha256=first._digest(provider["module_closure"]),
        all_file_backed_modules_predeclared_and_exact=True))
    observation = dict(native_launch=native, service_receipt=service, kernel_lifetime_memory=dict(peak=12),
        actual_registered_owner_exit_inventories=[dict(owner=owner, inventory=exit_pin,
            common_worker_scientific_evidence=worker_pin)])
    state._read = lambda pin: deepcopy(exited) if pin == exit_pin else pytest.fail("Unexpected metadata read")
    slots = dict(worker_scientific_evidence=(worker, worker_pin), observation=(observation, payload("observation.json")))
    for role in ("native_completion", "measurement", "writer_evidence"):
        slots[role] = ({}, payload(role + ".json"))
    monkeypatch.setattr(common, "_slot_read", lambda provider, role: deepcopy(slots[role]))
    writes = []
    monkeypatch.setattr(common, "_slot_write", lambda provider, role, document:
        writes.append((role, deepcopy(document))) or dict(path=str(tmp_path / (role + "-written.json")),
            sha256="a" * 64, size_bytes=1))
    monkeypatch.setattr(common, "checked_stage", lambda value, **kwargs: value)
    ledger = SimpleNamespace(locked=lambda **kwargs: nullcontext(), bound=dict(path=str(tmp_path)),
        control=SimpleNamespace(read_once=lambda name: (b"original once marker", object())))
    return state, SimpleNamespace(ledger=ledger), slots, writes


def test_original_controller_finalization_imports_no_scientific_package_after_retirement(tmp_path, monkeypatch):
    state, lease, slots, writes = metadata_closure(tmp_path, monkeypatch)
    original_import = builtins.__import__
    attempted = []
    def guarded_import(name, *args, **kwargs):
        if name == "darsia" or name.startswith("darsia.") or name.split(".")[0] in {"numpy", "scipy", "pyopencl"}:
            attempted.append(name)
            raise AssertionError("Scientific import outside the original native worker: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    first._close_common_stage_after_retirement(state, lease,
        dict(passed=True, observer_native_handles_closed=True))
    assert attempted == []
    assert [role for role, value in writes] == ["scientific_evidence", "completion"]
    science = writes[0][1]
    assert science["worker_scientific_evidence"] == slots["worker_scientific_evidence"][1]
    assert science["source_verification"]["post_retirement"] is True
    assert science["compute_evidence"]["kernel_lifetime_memory"] == dict(peak=12)


@pytest.mark.parametrize("mutation", ("worker_owner", "exit_pin", "changed_input", "unclosed_native"))
def test_controller_refuses_unbound_worker_or_changed_input_before_science_promotion(tmp_path, monkeypatch, mutation):
    state, lease, slots, writes = metadata_closure(tmp_path, monkeypatch)
    worker = slots["worker_scientific_evidence"][0]
    terminal = dict(passed=True, observer_native_handles_closed=True)
    if mutation == "worker_owner":
        worker["owner"] = dict(pid=999, creation_filetime_100ns="200")
    elif mutation == "exit_pin":
        slots["observation"][0]["actual_registered_owner_exit_inventories"][0]["common_worker_scientific_evidence"] = {}
    elif mutation == "changed_input":
        Path(worker["configuration"]["path"]).write_bytes(b"changed")
    else:
        terminal["observer_native_handles_closed"] = False
    with pytest.raises(CachePolicyError):
        first._close_common_stage_after_retirement(state, lease, terminal)
    assert writes == []


def test_close_stage_composes_a_complete_original_native_dag_with_live_content_gates(monkeypatch):
    """Use in-memory host transports while leaving every common content gate live.

    Stable on-disk hash/version verification has separate real-file tests. This
    positive composition case replaces only those transport operations and the
    immutable Windows slot transport, never a source/native/science validator.
    """
    from pathlib import PureWindowsPath
    from .test_common_profile_paths import synthetic_closed_dag
    value = synthetic_closed_dag(monkeypatch)
    state = value.states["auto"]["phase1"]
    view = common.stage_provider(value.roots["auto"], "phase1")
    worker_key = (view["output_namespace"]["path"], view["slots"]["worker_scientific_evidence"])
    worker, worker_pin = deepcopy(value.slots[worker_key])
    reference = deepcopy(state.task["input_catalog"]["inputs"][0])
    original_path = "C:/TESTONLY/original-input.bin"
    worker["resolved_inputs"] = [dict(reference, path=original_path, version=[1, 2, 1, 4])]
    value.write_slot(view, "worker_scientific_evidence", worker)
    allowed = [worker["completion"], *worker["publication"].values(), *worker["output_files"], state.stores_source,
        dict(path=original_path, sha256=reference["reference"]["sha256"], size_bytes=reference["reference"]["bytes"])]
    hashes = []
    def bound_transport(pin, *, expected_version=None):
        assert pin in allowed
        hashes.append(deepcopy(pin))
        return tuple(expected_version) if expected_version is not None else (1, 2, 1, 4)
    monkeypatch.setattr(common, "_hash_bound_file", bound_transport)
    monkeypatch.setattr(common, "_slot_write", value.write_slot)
    def read(pin):
        normalized = dict(pin, path=pin["path"].replace("\\", "/"))
        return value.read(normalized)
    state._read = read
    old_proof = value.proofs["auto"]["phase1"][0]
    marker_parent = str(PureWindowsPath(old_proof["once_marker"]["path"]).parent)
    ledger = SimpleNamespace(locked=lambda **kwargs: nullcontext(), bound=dict(path=marker_parent),
        control=SimpleNamespace(read_once=lambda name: (b"{}", {})))
    proof_pin = common.close_stage(state, SimpleNamespace(ledger=ledger))
    proof = value.read(proof_pin)
    assert common.checked_stage(proof, root=value.roots["auto"], provider_binding=value.bindings["auto"],
        stores_source=value.stores, read=read, read_authorities=None) == proof
    assert len(hashes) == len(allowed)
    science = value.read(proof["scientific_evidence"])
    assert science["source_verification"]["input_count"] == 1
    assert science["worker_scientific_evidence"] == worker_pin
    assert science["compute_evidence"]["native_launch_to_retirement_seconds"] == 2.
