"""The optional N1 device preserves CPU controls, fields and failure behavior."""
import numpy as np
import pytest
from types import SimpleNamespace

from darsia.fluidflower.mass import mass_opencl as backend
from darsia.fluidflower.mass.production_precision import PixelMath
from .test_staged_data import phase1_data


def operands(precision="float32", shape=(3, 7)):
    dtype = np.dtype(precision)
    prepared = np.linspace(-.2, 1.8, np.prod(shape), dtype=dtype).reshape(shape)
    return prepared, dict(gain=1.12, factor=.86,
        swi=np.full(shape, .23, dtype=dtype),
        aqueous_known=np.arange(prepared.size).reshape(shape) % 3 == 1,
        gas_known=np.arange(prepared.size).reshape(shape) % 3 == 2,
        solubility=np.full(shape, 1.6, dtype=dtype),
        gas_density=np.full(shape, 1.9, dtype=dtype))


def control(mode="auto", device_id=None):
    return dict(format=backend.FORMAT, mode=mode, device_id=device_id)


def identical(left, right):
    assert set(left) == set(right) == set(backend.FIELDS)
    for name in left:
        assert left[name].shape == right[name].shape
        assert left[name].dtype == right[name].dtype
        assert left[name].tobytes() == right[name].tobytes(), name


@pytest.fixture
def device_runtime(monkeypatch):
    clock = [0]
    monkeypatch.setattr(backend.time, "perf_counter_ns", lambda: clock[0])
    platform = SimpleNamespace(name="test", version="1")
    device = SimpleNamespace(name="test GPU", driver_version="1", global_mem_size=1024**3,
                             single_fp_config=7, double_fp_config=7)
    cl = SimpleNamespace(device_fp_config=SimpleNamespace(ROUND_TO_NEAREST=1, DENORM=2, INF_NAN=4))
    monkeypatch.setattr(backend, "_devices", lambda precision, requested: [(cl, 0, platform, device)])
    sessions = []

    class Session:
        mismatch = False
        failure = False
        elapsed = 10

        def __init__(self, entry, precision, size):
            self.dtype = np.dtype(precision)
            self.device = backend.device_identity(entry)
            self.buffer_bytes = 100
            self.closed = False
            assert all(value.closed for value in sessions), "Candidate allocations must never overlap"
            sessions.append(self)

        def tiles(self, values, gain, factor):
            clock[0] += self.elapsed
            if self.failure:
                raise RuntimeError("device unavailable")
            for start in range(0, values[0].size, 5):
                arrays = [np.asarray(value.flat[start:start + 5]) for value in values]
                result = original(PixelMath(self.dtype.name), arrays[0],
                    gain=gain, factor=factor, swi=arrays[1], solubility=arrays[2],
                    gas_density=arrays[3], aqueous_known=arrays[4], gas_known=arrays[5])
                if self.mismatch:
                    result["mass_total"][0] = np.nextafter(result["mass_total"][0], np.inf)
                yield start, result

        def close(self):
            self.closed = True

    monkeypatch.setattr(backend, "_Session", Session)
    Session.clock = clock
    original = PixelMath._selected_n1_fields_cpu

    def measured_cpu(self, *args, **kwargs):
        clock[0] += 100
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PixelMath, "_selected_n1_fields_cpu", measured_cpu)
    return Session, sessions


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_first_result_is_cpu_then_exact_faster_device_is_fixed(device_runtime, precision):
    _, sessions = device_runtime
    prepared, values = operands(precision)
    cpu = PixelMath(precision)
    expected = cpu.selected_n1_fields(prepared, **values)
    math = PixelMath(precision)
    math.configure_opencl(control())
    identical(expected, math.selected_n1_fields(prepared, **values))
    first = math.facts()["opencl_n1"]
    assert first["last_backend"] == "cpu" and first["used"] is False
    assert first["candidates"][0]["identical"] is True
    assert len(first["candidates"][0]["replay_ns"]) == 2
    identical(expected, math.selected_n1_fields(prepared, **values))
    facts = math.facts()["opencl_n1"]
    assert facts["last_backend"] == "opencl" and facts["accelerated_calls"] == 1
    assert facts["selected_device"]["device_id"] == 0
    assert facts["application_device_buffer_bytes"] == 100
    assert (facts["peak_process_memory_bytes"] is None
            or facts["peak_process_memory_bytes"] > 0)
    facts["selected_device"]["name"] = "changed copy"
    assert math.facts()["opencl_n1"]["selected_device"]["name"] == "test GPU"
    math.close()
    assert sessions[0].closed


def test_cpu_control_never_probes_a_device(monkeypatch):
    monkeypatch.setattr(backend, "_devices", lambda *args: pytest.fail("CPU mode must not probe"))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control("cpu"))
    identical(PixelMath("float32").selected_n1_fields(prepared, **values),
              math.selected_n1_fields(prepared, **values))
    assert math.facts()["opencl_n1"] == dict(control=control("cpu"), used=False)


@pytest.mark.parametrize("reason", ("missing", "mismatch", "slower"))
def test_unusable_inexact_or_slower_device_keeps_cpu(device_runtime, monkeypatch, reason):
    session, sessions = device_runtime
    if reason == "missing":
        monkeypatch.setattr(backend, "_devices", lambda *args: [])
    elif reason == "mismatch":
        session.mismatch = True
    else:
        session.elapsed = 200
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    expected = PixelMath("float32").selected_n1_fields(prepared, **values)
    identical(expected, math.selected_n1_fields(prepared, **values))
    identical(expected, math.selected_n1_fields(prepared, **values))
    assert math.facts()["opencl_n1"]["selected_device"] is None
    assert all(item.closed for item in sessions)


def test_optional_device_failure_returns_original_cpu_fields(device_runtime):
    session, _ = device_runtime
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    session.failure = True
    identical(PixelMath("float32").selected_n1_fields(prepared, **values),
              math.selected_n1_fields(prepared, **values))
    assert math.facts()["opencl_n1"]["last_backend"] == "cpu"
    assert math.facts()["opencl_n1"]["fallback"] == "RuntimeError"


def test_configuration_cannot_drift_after_first_call(device_runtime):
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    math.configure_opencl(control())
    with pytest.raises(RuntimeError, match="cannot change"):
        math.configure_opencl(control("cpu"))


@pytest.mark.parametrize("value", ({}, {"format": backend.FORMAT, "mode": "other"},
    dict(control(), device_id=True), dict(control(), device_id=-1),
    dict(control(), unexpected="value")))
def test_bad_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="control"):
        backend.configuration(value)


def test_gpu_discovery_failure_is_optional(device_runtime, monkeypatch):
    def missing(*args):
        raise ImportError("optional runtime unavailable")
    monkeypatch.setattr(backend, "_devices", missing)
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    identical(PixelMath("float32").selected_n1_fields(prepared, **values),
              math.selected_n1_fields(prepared, **values))
    assert math.facts()["opencl_n1"]["fallback"] == "ImportError"


@pytest.mark.parametrize("mode", ("cpu", "auto"))
def test_bound_control_survives_plan_mapping_and_phase_two_seal(tmp_path, phase1_data, mode):
    import json
    from .test_staged_data import _closure
    from .test_analysis_plan import _original
    from darsia.fluidflower import analysis_plan as plans
    from darsia.fluidflower.pipeline import staged_data as staged
    from darsia.fluidflower.mass.n1_pipeline import DataReader

    original = _original(phase1_data)
    original["mass"]["template"]["opencl_acceleration"] = control(mode)
    configured = plans.phase1_configuration(original, precision="float32")
    assert configured["statistics"]["opencl_acceleration"] == control(mode)
    staged.validate_phase1_run(configured)
    closure = _closure(tmp_path, configured)
    target = tmp_path / "mass.json"
    staged.seal_phase2_configuration(target, closure["completion"], closure["configuration"],
        expected_phase1_commit="1" * 40, fitted_calibration=closure["calibration"],
        fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=.0001), reader=DataReader())
    sealed = json.loads(target.read_bytes())
    assert sealed["opencl_acceleration"] == control(mode)
    assert staged.validate_phase2_run(sealed) is sealed


def test_unknown_device_or_driver_uses_admitted_cpu_before_allocation(device_runtime, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    _, sessions = device_runtime
    admitted = backend.device_identity(backend._devices("float32", None)[0])
    admitted["driver"] = "different"
    policy = dict(qualified_devices=[admitted], cpu_fallback_qualified=True)
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace(opencl_policy=policy))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    identical(PixelMath("float32").selected_n1_fields(prepared, **values),
              math.selected_n1_fields(prepared, **values))
    assert not sessions
    assert math.facts()["opencl_n1"]["candidates"][0]["fallback"] == "device_not_in_qualified_profile"


def test_known_device_can_be_reused_with_its_bound_driver(device_runtime, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    admitted = backend.device_identity(backend._devices("float32", None)[0])
    policy = dict(qualified_devices=[admitted], cpu_fallback_qualified=True)
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace(opencl_policy=policy))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    math.selected_n1_fields(prepared, **values)
    assert math.facts()["opencl_n1"]["last_backend"] == "opencl"
    assert math.facts()["opencl_n1"]["admitted_devices"] == [admitted]


def test_common_proof_compares_every_accelerated_call_and_fails_changed_pixels(device_runtime, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    session, _ = device_runtime
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace(
        opencl_compare_cpu=True, opencl_proof_released=True))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    math.selected_n1_fields(prepared, **values)
    facts = math.facts()["opencl_n1"]
    assert facts["cpu_comparison_calls"] == facts["accelerated_calls"] == 1
    assert facts["cpu_comparison_ns"] == 100 and facts["accelerated_ns"] == 10
    assert facts["peak_proof_duplicate_host_bytes"] == prepared.nbytes * 6
    assert facts["byte_verified_accelerated_calls"] == 1 and facts["guarded_accelerated_calls"] == 0
    session.mismatch = True
    with pytest.raises(ArithmeticError, match="proof differs"):
        math.selected_n1_fields(prepared, **values)
    assert math.facts()["opencl_n1"]["comparison_failures"] == 1


def test_common_proof_cannot_hide_initial_probe_pixel_mismatch(device_runtime, monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    session, _ = device_runtime
    session.mismatch = True
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace(
        opencl_compare_cpu=True, opencl_proof_released=True))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    with pytest.raises(ArithmeticError, match="pixels differ"):
        math.selected_n1_fields(prepared, **values)


def test_probe_mismatch_survives_candidate_cleanup_failure(device_runtime, monkeypatch):
    """A second device cleanup error cannot turn a proof mismatch into CPU fallback."""
    from darsia.fluidflower.pipeline import run_io
    session_type, sessions = device_runtime
    session_type.mismatch = True
    closed = []
    original_close = session_type.close

    def failing_close(self):
        """Record the original close attempt before simulating a queue failure."""
        closed.append(self)
        original_close(self)
        raise RuntimeError("device cleanup unavailable")

    monkeypatch.setattr(session_type, "close", failing_close)
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace(
        opencl_compare_cpu=True, opencl_proof_released=True))
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    with pytest.raises(backend._ProofMismatch, match="pixels differ"):
        math.selected_n1_fields(prepared, **values)
    assert closed == sessions and len(sessions) == 1 and sessions[0].closed
    facts = math.facts()["opencl_n1"]
    assert facts["comparison_failures"] == 1 and facts["accelerated_calls"] == 0
    assert facts["used"] is False and "fallback" not in facts


def test_candidate_cleanup_failure_without_mismatch_keeps_cpu(device_runtime, monkeypatch):
    """A cleanup-only runtime error still returns the original CPU fields."""
    session_type, sessions = device_runtime
    closed = []
    original_close = session_type.close

    def failing_close(self):
        """Record the original close attempt before simulating a queue failure."""
        closed.append(self)
        original_close(self)
        raise RuntimeError("device cleanup unavailable")

    monkeypatch.setattr(session_type, "close", failing_close)
    prepared, values = operands()
    expected = PixelMath("float32").selected_n1_fields(prepared, **values)
    math = PixelMath("float32")
    math.configure_opencl(control())
    identical(expected, math.selected_n1_fields(prepared, **values))
    identical(expected, math.selected_n1_fields(prepared, **values))
    assert closed == sessions and len(sessions) == 1 and sessions[0].closed
    facts = math.facts()["opencl_n1"]
    assert facts["fallback"] == "RuntimeError" and facts["comparison_failures"] == 0
    assert facts["selected_device"] is None and facts["used"] is False


def test_undeclared_control_keeps_cpu_facts_without_reading_policy(monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    monkeypatch.setattr(run_io, "current_binding", lambda: pytest.fail("Undeclared control reads no policy"))
    monkeypatch.setattr(backend, "_devices", lambda *args: pytest.fail("Undeclared control probes no device"))
    math = PixelMath("float32")
    expected = dict(precision="float32", backend="numpy_cpu", parameter_dtype="float64",
        reduction_dtype="float64", input_storage_dtype="frozen_asset_encoding",
        scientific_eligibility=False, production_permission_granted=False)
    math.configure_opencl(None)
    prepared, values = operands()
    math.selected_n1_fields(prepared, **values)
    assert math.facts() == expected
    assert backend.configuration(None) == control("cpu")


def test_undeclared_phase_one_sidecar_keeps_the_original_bytes(tmp_path, monkeypatch):
    import json
    from darsia import _opencv_runtime
    from darsia.fluidflower.pipeline import phase_statistics
    monkeypatch.setattr(_opencv_runtime, "require_ipp_off", lambda *args: None)
    monkeypatch.setattr(_opencv_runtime, "validate_observation", lambda *args: None)
    observation = dict(TESTONLY="unchanged observation")
    bound = phase_statistics._save_statistics(tmp_path,
        dict(statistics=dict(regions=[]), sandmap={}, opencv_ipp=observation))
    original = dict(format="darsia-native-statistics-runtime-v1", statistics=bound, opencv_ipp=observation)
    expected = json.dumps(original, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf8")
    assert (tmp_path / "native_statistics_runtime.json").read_bytes() == expected
    # The runner only supplies optional facts for an explicitly declared block.
    import ast
    from pathlib import Path
    source = ast.parse(Path(phase_statistics.__file__).read_text(encoding="utf8"))
    writes = [node for node in ast.walk(source) if isinstance(node, ast.If)
              and any(isinstance(child, ast.Assign) and any(
                  isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                  and target.slice.value == "pixel_backend_facts" for target in child.targets)
                  for child in node.body)]
    assert len(writes) == 1 and ast.unparse(writes[0].test) == "'opencl_acceleration' in statistics_config"


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("missing", (1, 2, 4))
def test_required_fp_behavior_is_checked_before_allocation(device_runtime, precision, missing):
    _, sessions = device_runtime
    entry = backend._devices(precision, None)[0]
    setattr(entry[3], "double_fp_config" if precision == "float64" else "single_fp_config", 7 ^ missing)
    prepared, values = operands(precision)
    math = PixelMath(precision)
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    facts = math.facts()["opencl_n1"]
    assert not sessions and facts["candidates"][0]["fallback"] == "required_floating_point_behavior_unavailable"


@pytest.mark.parametrize("key", ("prepared", "swi", "solubility", "gas_density", "gain", "factor"))
def test_later_nonfinite_operands_keep_the_original_cpu_bytes(device_runtime, key):
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    if key == "prepared":
        prepared.flat[0] = np.nan
    elif key in {"gain", "factor"}:
        values[key] = np.inf
    else:
        values[key].flat[0] = np.nan
    with np.errstate(all="ignore"):
        expected = PixelMath("float32").selected_n1_fields(prepared, **values)
        identical(expected, math.selected_n1_fields(prepared, **values))
    facts = math.facts()["opencl_n1"]
    assert facts["nonfinite_cpu_calls"] == 1 and facts["accelerated_calls"] == 0
    assert facts["last_backend"] == "cpu"


def test_probe_times_full_output_allocation_and_records_host_duplicates(device_runtime, monkeypatch):
    session_type, sessions = device_runtime
    original = np.empty_like
    # Each device field allocation now costs more than the CPU's whole call.
    def expensive_output(value):
        session_type.clock[0] += 200
        return original(value)
    monkeypatch.setattr(backend.np, "empty_like", expensive_output)
    prepared, values = operands()
    values = {key: value.astype(np.float64) if key in {"swi", "solubility", "gas_density"} else value
              for key, value in values.items()}
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    facts = math.facts()["opencl_n1"]
    assert facts["selected_device"] is None and all(session.closed for session in sessions)
    assert facts["peak_probe_duplicate_host_bytes"] == prepared.nbytes * 6
    assert facts["peak_operand_cast_host_bytes"] == prepared.nbytes * 3
    assert facts["peak_application_device_buffer_bytes"] == 100


def test_bound_auto_without_policy_or_release_is_refused(monkeypatch):
    from darsia.fluidflower.pipeline import run_io
    monkeypatch.setattr(run_io, "current_binding", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="admitted policy"):
        PixelMath("float32").configure_opencl(control())


def test_session_build_disables_the_python_compiler_cache():
    calls, released = [], []
    class Program:
        def __init__(self, context, source):
            pass
        def build(self, **kwargs):
            calls.append(kwargs)
            return self
    class Buffer:
        def __init__(self, *args):
            pass
        def release(self):
            released.append(self)
    cl = SimpleNamespace(Context=lambda *args: object(),
        CommandQueue=lambda *args: SimpleNamespace(finish=lambda: None), Program=Program,
        Kernel=lambda *args: object(), Buffer=Buffer, mem_flags=SimpleNamespace(READ_ONLY=1, WRITE_ONLY=2))
    entry = (cl, 0, SimpleNamespace(name="test", version="1"), SimpleNamespace(
        name="test GPU", driver_version="1", global_mem_size=1024**3, max_mem_alloc_size=1024**2))
    session = backend._Session(entry, "float32", 21)
    assert calls == [dict(options=["-cl-opt-disable"], cache_dir=False)]
    session.close()
    assert len(released) == 12 and session.context is None


def test_close_waits_for_the_same_provider_compute_lock(device_runtime, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    session_type, sessions = device_runtime
    prepared, values = operands()
    math = PixelMath("float32")
    math.configure_opencl(control())
    math.selected_n1_fields(prepared, **values)
    entered, release, closing = Event(), Event(), Event()
    original = session_type.tiles
    def blocked_tiles(self, *args):
        entered.set()
        assert release.wait(2)
        yield from original(self, *args)
    def close():
        closing.set()
        math.close()
    monkeypatch.setattr(session_type, "tiles", blocked_tiles)
    with ThreadPoolExecutor(max_workers=2) as pool:
        computing = pool.submit(math.selected_n1_fields, prepared, **values)
        try:
            assert entered.wait(2)
            closing_future = pool.submit(close)
            assert closing.wait(2)
            assert not sessions[-1].closed
        finally:
            release.set()
        computing.result(timeout=2)
        closing_future.result(timeout=2)
    assert sessions[-1].closed
