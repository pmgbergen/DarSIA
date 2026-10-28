"""Synthetic common result/cap/policy wiring; no provider or production proof."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.pipeline import capacity, worker_io
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from darsia.fluidflower.pipeline.job_queue import LocalOutputBinding
from darsia.fluidflower.pipeline.store_runtime import ResolvedInput, resolve_catalog, InputHashMismatch
from darsia.fluidflower.job_filesystem import LocalFileAdmission
from darsia.fluidflower.store_map import parse_store_map
from darsia_cache import common_qualification as common
from darsia_cache.policy import CachePolicyError
from darsia_cache.worker_adapter import WorkerCacheAdapter
from .test_phase2_lean_capacity import _prepared
from .test_staged_data import phase1_data  # noqa: F401


def _pin(path, raw=b"synthetic metadata"):
    """Describe authored unit bytes without asserting an actual qualified source."""
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _write(path, value):
    """Create one exclusive JSON input and return its original byte identity."""
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return _pin(path, raw)


def _control(mode):
    """The positive explicit CPU/auto control used by this fixture."""
    return dict(format="darsia-opencl-n1-v1", mode=mode, device_id=None)


def _task(stage, config):
    """Bind a synthetic task to one genuine route and its explicit selection."""
    return dict(format="darsia-alias-task-v1", format_version=2 if stage == "phase2" else 1,
        pipeline=common.PIPELINES[stage], storage_profile=common.PROFILES[stage],
        run=config["run"], ordered_case_ids=[row["case_id"] for row in config.get("frames", ())] or ["case-one"],
        commit="a" * 40, backend="cpu", precision="float32", attempt_id="1" * 32,
        configuration=_pin("configuration.json"))


def _profile(task, stage, control):
    """Resolved-profile leaf fixture; authority resolution is tested separately."""
    producers, runtime = [_pin("producer.py")], [_pin("runtime.py")]
    return dict(format=common.JOB_PROFILE, purpose="jobs", provider=_pin("provider.json"),
        stores_source=_pin("stores.toml"), namespace_admission=_pin("namespace.json"),
        scope={key: task[key] for key in ("commit", "backend", "precision")},
        stages={stage: _pin("stage.json")},
        stage_routes={stage: dict(pipeline=common.PIPELINES[stage], storage_profile=common.PROFILES[stage])},
        device_identity=dict(synthetic=True), python_executable=_pin("python.exe"),
        producer_source_closure=producers, runtime_source_closure=runtime, module_closure=producers + runtime,
        compute_profiles={stage: dict(control=deepcopy(control), qualified_devices=[], cpu_fallback_qualified=True)},
        result_spaces={stage: dict(case_count=2, maximum_output_bytes=1200,
            configuration=deepcopy(task["configuration"]))},
        allocation_model="shared-service-only-no-claim-files-v1", additional_claim_cache_bytes=0,
        actual_writer_IO_proven=False, sampled_highwater_is_lifetime_peak=False,
        operational_service_entitlement_bytes=100)


def _phase1_layout(tmp_path, config):
    """Write the actual named layout JSON consumed by the result estimator."""
    rows = []

    def put(use, value):
        """Create one original bound metadata role without decoding an image."""
        bound = _write(tmp_path / (use + ".json"), value)
        rows.append(ResolvedInput(dict(sha256=bound["sha256"], bytes=bound["size_bytes"]),
            Path(bound["path"]), (use,)))
        return bound

    hw = [2, 2]
    config["native"]["configuration"] = put("native.configuration", dict(
        photometry=dict(native_shape_chw=[3, *hw], native_shape_hw=hw),
        source=dict(native_shape_hw=hw), extension=dict(native_shape_hw=hw), support=dict(native_shape_hw=hw)))
    config["phase"]["configuration"] = put("phase.configuration", dict(parameters=dict(
        geometry=dict(native_shape_hw=hw, source_box_xywh=[0, 0, 2, 2]),
        scientific=dict(native_shape_hw=hw), features=dict(alignment=dict(rgb_dtype="float32")))))
    for frame in config["frames"]:
        prefix = "frame." + frame["case_id"]
        frame["native_task"]["geometry"] = put(prefix + ".native_geometry", dict(native_shape_hw=hw))
        frame["geometry"] = put(prefix + ".rig_geometry", dict(source_native_shape_hw=hw, phase_shape_hw=hw))
        frame["mass_spec"]["photo_top"]["shape_hw"] = hw
    return rows


@pytest.mark.parametrize("mode", [None, "cpu", "auto"])
def test_phase2_lean_estimate_and_reservation_preserve_undeclared_record_shape(tmp_path, monkeypatch, mode):
    """The real estimator and immutable reservation add no absent-control field."""
    (tmp_path / "inputs").mkdir()
    task, config, rows = _prepared(tmp_path / "inputs")
    task.update(attempt_id="1" * 32, configuration=_pin("configuration.json"))
    if mode is not None:
        config["opencl_acceleration"] = _control(mode)
    monkeypatch.setattr(capacity, "_lean_writer_identity", lambda *args: [_pin("writer.py")])
    profile = _profile(task, "phase2", None if mode is None else _control(mode))
    estimate = capacity._common_result_estimate(None, "sample", task, config, profile,
        profile["provider"], repository=tmp_path, metadata_rows=rows)
    expected = deepcopy(estimate.estimate)
    if mode is None:
        assert "compute_control" not in expected
    else:
        assert expected.pop("compute_control") == _control(mode)
    # The estimator's other bytes remain identical to the undeclared route.
    plain = deepcopy(config)
    plain.pop("opencl_acceleration", None)
    baseline = capacity._phase2_lean_estimate(None, "sample", task, plain,
        repository=tmp_path, metadata_rows=rows)
    assert expected == baseline.estimate
    ledger = object.__new__(capacity.CapacityLedger)
    ledger.root = tmp_path / "ledger"
    ledger.root.mkdir()
    ledger.machine_id = ledger.host = "local"
    ledger.account, ledger.volume, ledger.remote_scope = "synthetic", dict(volume_id="synthetic"), None
    monkeypatch.setattr(ledger, "_common_binding", lambda: True)
    owner = dict(machine="local", pid=1, process_create_time="1.0")
    claim = SimpleNamespace(task=task, token=task["attempt_id"])
    root = tmp_path / "job"
    reservation = ledger.bind(root, claim, owner, estimate)
    actual = json.loads((reservation / "reserved.json").read_bytes())
    assert actual == dict(machine_id="local", physical_host="local", service_account_sid="synthetic",
        volume_id="synthetic", owner=owner, job_root=str(root.resolve()), run=task["run"],
        attempt_id=claim.token, case_count=estimate.case_count, ordered_case_ids=task["ordered_case_ids"],
        format="darsia-local-capacity-reservation-v1",
        products_per_frame_estimated_bytes=estimate.products_per_frame_estimated_bytes,
        transient_estimated_bytes=estimate.transient_estimated_bytes, estimate=asdict(estimate))
    if mode is None:
        assert "compute_control" not in actual["estimate"]["estimate"]


@pytest.mark.parametrize("mode", [None, "cpu", "auto"])
@pytest.mark.parametrize("count,whole_stage_bytes", [(1, 1200), (3, 2400)])
def test_common_phase1_result_estimate_uses_actual_layout_and_explicit_control(
        phase1_data, tmp_path, mode, count, whole_stage_bytes):
    """A positive P1 allowance keeps its full stage and the declared record shape."""
    original = phase1_data["frames"][0]
    previous = phase1_data["phase"]["initial_previous"]["source"]["source_sha256"]
    phase1_data["frames"] = []
    for index in range(1, count + 1):
        frame = deepcopy(original)
        digest = format(index, "064x")
        frame.update(case_id="case-" + str(index), frame_index=index, elapsed_h=float(index))
        frame["source"].update(work_index=index, elapsed_h=float(index), source_sha256=digest,
            previous_source_sha256=previous, source_binding=dict(source_sha256=digest))
        phase1_data["frames"].append(frame)
        previous = digest
    phase1_data["ordered_case_ids"] = [frame["case_id"] for frame in phase1_data["frames"]]
    rows = _phase1_layout(tmp_path, phase1_data)
    control = None if mode is None else _control(mode)
    if control is not None:
        phase1_data["statistics"]["opencl_acceleration"] = deepcopy(control)
    task = _task("phase1", phase1_data)
    profile = _profile(task, "phase1", control)
    layout = capacity._native_layout_for_task(None, "sample", task, phase1_data, rows)
    profile["result_spaces"]["phase1"]["measured_native_layout"] = layout
    before = deepcopy(phase1_data)
    estimate = capacity._common_result_estimate(None, "sample", task, phase1_data, profile,
        profile["provider"], repository=tmp_path, metadata_rows=rows)
    from darsia.fluidflower.pipeline.phase1_capacity import RUN_BYTES
    assert estimate.maximum_output_bytes == whole_stage_bytes + 4 * RUN_BYTES
    assert estimate.needed(reserve_bytes=0) == estimate.maximum_output_bytes
    scope = dict(stage="phase1", configuration=deepcopy(task["configuration"]))
    if control is not None:
        scope["compute_control"] = control
    assert estimate.scope == scope and phase1_data == before


@pytest.mark.parametrize("defect", ["missing", "format", "non_native"])
def test_common_phase1_requests_require_original_native_evidence_storage(phase1_data, defect):
    """The P1 selector cannot hide absent or malformed mandatory evidence requests."""
    from darsia.fluidflower.pipeline.phase1_capacity import validate_requests
    if defect == "missing":
        phase1_data.pop("storage")
    elif defect == "format":
        phase1_data["storage"]["format"] = "invalid-evidence-request"
    else:
        phase1_data["storage"]["resolution_mm"] = "invalid-native-resolution"
    before = deepcopy(phase1_data)
    with pytest.raises(ValueError):
        validate_requests(phase1_data)
    assert phase1_data == before


def test_common_calibration_result_estimate_keeps_exact_plan_and_no_compute_field(tmp_path):
    """C uses only its qualified exact plan, without a synthetic CPU control."""
    config = dict(run="sample")
    task = _task("calibration", config)
    profile = _profile(task, "calibration", None)
    estimate = capacity._common_result_estimate(None, "sample", task, config, profile,
        profile["provider"], repository=tmp_path, metadata_rows=())
    assert estimate.maximum_output_bytes == 1200
    assert estimate.scope == dict(stage="calibration", configuration=task["configuration"])
    changed = deepcopy(task)
    changed["configuration"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="actual qualified plan"):
        capacity._common_result_estimate(None, "sample", changed, config, profile,
            profile["provider"], repository=tmp_path, metadata_rows=())


def _adapter(task, profile, *, maximum=96 * 1024**3):
    """Isolate the already-resolved adapter leaf from OS/bootstrap ownership."""
    document = dict(format="synthetic-cap", maximum_size_bytes=maximum)
    cap = SimpleNamespace(maximum_size_bytes=maximum, document=lambda: deepcopy(document))
    profile["capacity_bound"] = dict(volume_cap_observation=dict(binding=deepcopy(document), virtual_size_bytes=maximum))
    adapter = object.__new__(WorkerCacheAdapter)
    adapter.runtime = SimpleNamespace(commit=task["commit"], backend="cpu")
    adapter.lease = SimpleNamespace(profile=dict(deepcopy(profile), format=common.IMPORT_PROFILE),
        profile_binding=deepcopy(profile["provider"]), source=deepcopy(profile["stores_source"]),
        document=dict(cache_bootstrap=dict(namespace_admission=deepcopy(profile["namespace_admission"]))),
        maximum=100, policy=SimpleNamespace(quota_bytes=maximum, volume_cap=cap))
    adapter.profiles = {}
    return adapter


@pytest.mark.parametrize("mode", [None, "cpu", "auto"])
def test_common_adapter_accepts_actual_nonhistorical_host_cap_and_remembers_exact_task(mode):
    """The positive leaf accepts the bound quota rather than a fixed capacity."""
    config = dict(run="sample", frames=[dict(case_id="case-one")])
    task = _task("phase1", config)
    control = None if mode is None else _control(mode)
    profile = _profile(task, "phase1", control)
    adapter = _adapter(task, profile)
    scope = {} if control is None else dict(compute_control=control)
    estimate = SimpleNamespace(scope=scope)
    measured = dict(complete=True, first_stop=None, cache_operational_reserve_crossed=False)
    completed = dict(complete=True, passed=True)
    assert adapter._verify_common_profile(task, estimate, profile["provider"], profile,
        measured, completed, remember=True)
    saved = adapter.profiles[task["attempt_id"]]
    assert saved["task"] == task and saved["qualification_kind"] == "common-phase-physical-bound"
    task["ordered_case_ids"].append("case-two")
    assert saved["task"]["ordered_case_ids"] == ["case-one"]


@pytest.mark.parametrize("defect", ["quota", "observed_size", "binding", "control", "completion", "closure"])
def test_common_adapter_refuses_mismatched_actual_cap_control_completion_and_sources(defect):
    """No positive adapter leaf can substitute a new cap or optional control."""
    task = _task("phase1", dict(run="sample", frames=[dict(case_id="case-one")]))
    profile = _profile(task, "phase1", _control("auto"))
    adapter = _adapter(task, profile)
    estimate = SimpleNamespace(scope=dict(compute_control=_control("auto")))
    completed = dict(complete=True, passed=True)
    if defect == "quota":
        adapter.lease.policy.quota_bytes += 512
    elif defect in {"observed_size", "binding"}:
        field = "virtual_size_bytes" if defect == "observed_size" else "binding"
        value = 1 if defect == "observed_size" else dict(different=True)
        profile["capacity_bound"]["volume_cap_observation"][field] = value
        adapter.lease.profile["capacity_bound"] = deepcopy(profile["capacity_bound"])
    elif defect == "control":
        estimate.scope["compute_control"] = _control("cpu")
    elif defect == "completion":
        completed["passed"] = False
    else:
        profile["runtime_source_closure"] = deepcopy(profile["producer_source_closure"])
        adapter.lease.profile["runtime_source_closure"] = deepcopy(profile["runtime_source_closure"])
    with pytest.raises(CachePolicyError):
        adapter._verify_common_profile(task, estimate, profile["provider"], profile,
            dict(complete=True, first_stop=None, cache_operational_reserve_crossed=False), completed, remember=True)
    assert adapter.profiles == {}


def _claim(tmp_path, mode):
    """Prepare a real named catalogue/output binding around tiny saved P2 inputs."""
    (tmp_path / "inputs").mkdir()
    task, config, _ = _prepared(tmp_path / "inputs")
    if mode is not None:
        config["opencl_acceleration"] = _control(mode)
    bound = _write(tmp_path / "run.json", config)
    manifest = _write(tmp_path / "manifest.json", dict(synthetic=True))
    execution = tmp_path / "execution.toml"
    text = '[[runs]]\nrun=' + json.dumps(config["run"]) + '\n[runs.configuration]\n'
    text += 'path=' + json.dumps(bound["path"]) + '\nsha256="' + bound["sha256"] + '"\n'
    text += 'size_bytes=' + str(bound["size_bytes"]) + '\n'
    with execution.open("xb") as stream:
        stream.write(text.encode())
    catalog = InputCatalog(dict(series=str(tmp_path)), config.get("data_aliases", ()))
    series_ref = catalog.add(_pin(execution, execution.read_bytes()), "configuration", "run.execution_series")
    manifest_ref = catalog.add(manifest, "configuration", "run.series_manifest")
    inputs = catalog.collect(bound, config)
    inputs["runtime_support"] = "darsia-role-io-runtime-v1"
    task.update(job_id="sample", attempt_id="1" * 32, legacy_paths=False, execution_ready=True,
        configuration=inputs["configuration"], series=series_ref, series_manifest=manifest_ref,
        input_catalog=inputs, calibration_sha256=config["calibration"]["sha256"], threads=1)
    results = tmp_path / "results"
    results.mkdir()
    stores = parse_store_map(('[machine]\nid="local"\n[stores.sample]\nseries=' + json.dumps(str(tmp_path)) +
        '\nresults=' + json.dumps(str(results)) + '\n').encode())
    rows = resolve_catalog(task, stores, "sample")
    admitted = LocalFileAdmission(True, "synthetic", "synthetic", "not a filesystem qualification", str(results))
    output = LocalOutputBinding("local", results, results / "sample" / task["run"] / ".attempts" / task["attempt_id"], admitted)
    return SimpleNamespace(task=task, token=task["attempt_id"], input_rows=rows, local_output=output), config


@pytest.mark.parametrize("route", ["default", "ordinary"])
def test_worker_prepare_attaches_only_the_admitted_policy_to_real_replayed_binding(tmp_path, route):
    """Actual input SHA admission and RunIO replay precede optional policy wiring."""
    claim, config = _claim(tmp_path, None if route == "default" else "auto")
    before = {row.path: row.path.read_bytes() for row in claim.input_rows}
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.commit, runtime.bindings = claim.task["commit"], {}
    profile = _profile(claim.task, "phase2", None if route == "default" else _control("auto"))
    profile["format"] = common.IMPORT_PROFILE
    devices = [dict(device_id=0, platform="synthetic", platform_version="1", name="synthetic",
        driver="1", global_memory_bytes=1024)]
    profile["compute_profiles"]["phase2"]["qualified_devices"] = deepcopy(devices)
    runtime.cache_adapter = SimpleNamespace(lease=SimpleNamespace(first_qualification=None,
        profile={} if route == "default" else profile))
    binding = runtime.prepare(claim)
    assert binding.configuration == config and binding.verified.run == claim.task["run"]
    assert binding is runtime.bindings[claim.token]
    if route == "ordinary":
        assert binding.opencl_policy == dict(qualified_devices=devices, cpu_fallback_qualified=True)
        assert binding.opencl_policy["qualified_devices"] is not profile["compute_profiles"]["phase2"]["qualified_devices"]
        assert not getattr(binding, "opencl_proof_released", False)
        assert not getattr(binding, "opencl_compare_cpu", False)
    else:
        assert not any(hasattr(binding, field) for field in
            ("opencl_policy", "opencl_proof_released", "opencl_compare_cpu", "common_closure_released"))
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_worker_prepare_requires_declared_control_for_the_released_common_proof(tmp_path):
    """An undeclared original configuration cannot silently become a proof arm."""
    claim, _ = _claim(tmp_path, None)
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.commit, runtime.bindings = claim.task["commit"], {}
    first = SimpleNamespace(common_mode=True, stage="phase2")
    runtime.cache_adapter = SimpleNamespace(lease=SimpleNamespace(first_qualification=first, profile={}))
    with pytest.raises(InputHashMismatch, match="declared CPU or auto control"):
        runtime.prepare(claim)
    assert runtime.bindings == {}


def test_worker_prepare_refuses_changed_bound_control_before_retaining_binding(tmp_path):
    """A valid input configuration still needs its actual qualified CPU/auto mode."""
    claim, _ = _claim(tmp_path, "auto")
    runtime = object.__new__(worker_io.WorkerRunIO)
    runtime.commit, runtime.bindings = claim.task["commit"], {}
    profile = _profile(claim.task, "phase2", _control("cpu"))
    profile["format"] = common.IMPORT_PROFILE
    runtime.cache_adapter = SimpleNamespace(lease=SimpleNamespace(first_qualification=None, profile=profile))
    with pytest.raises(InputHashMismatch, match="optional-compute control"):
        runtime.prepare(claim)
    assert runtime.bindings == {}
